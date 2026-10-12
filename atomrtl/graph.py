"""RTL-creation graph: static Python nodes do the deterministic work, LLM nodes plan/generate/fix.

parse_spec → plan → generate → write_code → check → route ─┬─ pass / out of budget → finalize → END
                        ▲            ▲                      ├─ errors → fix → write_code
                        │            └──────────────────────┤
                        └─────── stuck (same errors) ───────┤
                                                            └─ pass, simulation on → simulate → route_sim ─┐
     ┌──────────────────────────────────────────────────────────────────────────────────────────────────┘
     ├─ sim pass / out of sim budget / stuck → finalize (keeps the best simulated code)
     └─ mismatches → sim_fix → write_code (then check, and fix if the sim fix broke the build)
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from atomrtl import prompts
from atomrtl.eda import (
    Diagnostic,
    SimResult,
    check_interface,
    error_signature,
    fix_hints,
    format_diagnostics,
    lint_slang,
    run_iverilog,
    sim_signature,
    simulate,
)
from atomrtl.llm import make_llm
from atomrtl.spec import parse_spec
from atomrtl.state import RTLState, RunContext
from atomrtl.vcd import mismatch_table

_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.MULTILINE)


def extract_code(response: str) -> str | None:
    """The last fenced block that contains a full module (fix replies may quote old code first)."""
    blocks = [body for _, body in _FENCE_RE.findall(response) if "endmodule" in body]
    return blocks[-1].strip() + "\n" if blocks else None


def _call_llm(node: str, messages: list[BaseMessage], state: RTLState, config: RunnableConfig, ctx: RunContext) -> AIMessage:
    llm = make_llm(ctx.config, ctx.config.node_llm(node))
    # Passing the node's config propagates the transcript callback; metadata tags the attempt.
    return llm.with_config(metadata={"attempt": state.get("attempts", 0)}).invoke(messages, config)


def _usage(msg: AIMessage) -> dict:
    u = msg.usage_metadata or {}
    return {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "llm_calls": 1}


# --- static nodes -------------------------------------------------------------------------------


def parse_spec_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    spec = parse_spec(state["spec"], top=state.get("top"))
    runtime.context.transcript.event(
        "state",
        "parse_spec",
        title=f"module={spec.module_name} ports={len(spec.ports) if spec.ports else None}",
        sections={"HEADER": spec.header or "(none: interface not found in spec)"},
    )
    return {"module_name": spec.module_name, "ports": spec.ports, "header": spec.header, "attempts": 0}


def write_code_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    code = extract_code(state["response"])
    if code is None:
        ctx.transcript.event("state", "write_code", title="no code block in response", attempt=state["attempts"])
        return {"code": None, "code_path": None, "code_unchanged": False}

    name = state.get("module_name") or (m.group(1) if (m := _MODULE_RE.search(code)) else "design")
    path = ctx.workdir / f"{name}.sv"
    path.write_text(code)
    # A "fix" that returns the failing code verbatim gets called out explicitly (see check_node).
    unchanged = code.strip() == (state.get("code") or "").strip()
    title = f"wrote {path.name} ({len(code)} chars)" + (" UNCHANGED from the previous attempt" if unchanged else "")
    ctx.transcript.event("state", "write_code", title=title, attempt=state["attempts"])
    return {"code": code, "code_path": str(path), "code_unchanged": unchanged}


def check_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    ctx, t, attempt = runtime.context, runtime.context.transcript, state["attempts"]
    if not state.get("code_path"):
        diags = [Diagnostic("agent", "no_code", "error",
                            "Your response did not contain a ```systemverilog code block with a complete module.")]
    else:
        path = Path(state["code_path"])
        # Relative to the workdir so tool output shows "TopModule.sv:3", not a long absolute path.
        res, diags = run_iverilog([Path(path.name)], ctx.workdir, ctx.config.eda_timeout)
        t.event("tool", "check", title=f"$ {res.cmdline} -> exit {res.returncode}", attempt=attempt,
                duration_s=res.duration_s, sections={"OUTPUT": res.output or "(no output)"},
                tool="iverilog", returncode=res.returncode)

        lint = lint_slang([path], as_errors=ctx.config.blocking_warnings)
        n_err = sum(d.severity == "error" for d in lint.diagnostics)
        t.event("tool", "check", title=f"pyslang lint -> {n_err} errors, {len(lint.diagnostics) - n_err} warnings",
                attempt=attempt, duration_s=lint.duration_s, sections={"OUTPUT": lint.report or "(clean)"},
                tool="slang", modules={k: [asdict(p) for p in v] for k, v in lint.modules.items()})

        iface = check_interface(state.get("module_name"), state.get("ports"), lint.modules)
        t.event("tool", "check", title=f"interface check -> {'OK' if not iface else f'{len(iface)} problem(s)'}",
                attempt=attempt, sections={"OUTPUT": "\n".join(d.message for d in iface) or "OK"}, tool="interface")
        diags = diags + lint.diagnostics + iface
        if state.get("code_unchanged") and any(d.severity == "error" for d in diags):
            diags.insert(0, Diagnostic("agent", "no_change", "error",
                                       "Your corrected code is identical to the code that failed, so every error "
                                       "below is still there. You must actually change the code."))

    errors = sum(d.severity == "error" for d in diags)
    text = format_diagnostics(diags, state.get("code") or "", ctx.config.max_diagnostics)
    return {
        "diagnostics": diags,
        "feedback": [{"source": "check", "kind": "diagnostics", "text": text}] if diags else [],
        "history": [{
            "attempt": attempt,
            "node": state.get("response_node", "generate"),
            "signature": error_signature(diags),
            "errors": errors,
            "warnings": len(diags) - errors,
        }],
    }


def route_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    cfg = runtime.context.config
    history, attempts = state["history"], state["attempts"]
    errors = history[-1]["errors"]
    recent = [h["signature"] for h in history[-cfg.stuck_after:]]

    if errors == 0 and cfg.simulation.enabled and runtime.context.testbench:
        route, reason = "simulate", "clean: simulating against the testbench"
    elif errors == 0:
        route, reason = "finalize", "pass: no errors"
    elif attempts >= cfg.max_attempts:
        route, reason = "finalize", f"fail: budget exhausted ({attempts}/{cfg.max_attempts} attempts)"
    elif len(recent) == cfg.stuck_after and len(set(recent)) == 1:
        route, reason = "generate", f"stuck: same errors {cfg.stuck_after}x in a row (signature {recent[0]}), fresh restart"
    else:
        route, reason = "fix", f"{errors} error(s), attempt {attempts}/{cfg.max_attempts}"

    runtime.context.transcript.event("route", "route", title=f"-> {route}: {reason}", attempt=attempts)
    update = {"route": route, "route_reason": reason}
    if route == "generate":
        errs = [d.message for d in state["diagnostics"] if d.severity == "error"][:3]
        hints = fix_hints(state["diagnostics"])
        update["restart_hint"] = (
            "A previous implementation kept failing with these errors despite fixes:\n- "
            + "\n- ".join(errs)
            + ("\nHow to avoid them:\n- " + "\n- ".join(hints) if hints else "")
            + "\nWrite a fresh implementation that avoids the constructs that caused them."
        )
    return update


def finalize_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    status = "pass" if state["history"] and state["history"][-1]["errors"] == 0 else "fail"
    update = {}
    sim = state.get("sim")
    if sim is not None:
        # With simulation on, only a matching simulation is a pass. Otherwise keep the best simulated code
        # (a later sim fix may have made it worse or broken the build).
        status = "pass" if status == "pass" and sim["status"] == "pass" else "fail"
        best = state.get("best_code")
        if status == "fail" and best is not None:
            path = ctx.workdir / f"{state.get('module_name') or 'design'}.sv"
            restored = not path.exists() or path.read_text() != best
            path.write_text(best)
            update = {"code": best, "code_path": str(path)}
            if restored:
                ctx.transcript.event("state", "finalize", title=f"restored the best simulated code "
                                     f"({state['best_mismatches']} mismatches) to {path.name}")
    result = {
        "status": status,
        "reason": state.get("route_reason"),
        "module_name": state.get("module_name"),
        "code_path": update.get("code_path", state.get("code_path")),
        "attempts": state["attempts"],
        "history": state["history"],
        "final_diagnostics": [asdict(d) for d in state.get("diagnostics", [])],
        "tokens": state.get("tokens", {}),
    }
    if sim is not None:
        result["simulation"] = {
            "last": sim,
            "best_mismatches": state.get("best_mismatches"),
            "sim_fixes": state.get("sim_attempts", 0),
            "history": state.get("sim_history", []),
        }
    (ctx.workdir / "result.json").write_text(json.dumps(result, indent=2))
    ctx.transcript.event("state", "finalize", title=f"status={status}", sections={"RESULT": json.dumps(result, indent=2)})
    return {"status": status, **update}


# --- simulation (optional) ----------------------------------------------------------------------


def _sim_feedback(res: SimResult, ctx: RunContext) -> str:
    """What sim_fix sees: per-output mismatch summary plus the waveform table (never the testbench/reference source)."""
    cfg = ctx.config.simulation
    if res.status == "compile_error":
        out = res.log[-1].split("\n", 1)[-1].replace(str(ctx.testbench), "testbench.sv").replace(str(ctx.reference), "reference.sv")
        return "The testbench failed to compile together with your design:\n" + "\n".join(out.strip().splitlines()[:30])
    if res.status == "sim_timeout":
        return ("The simulation never finished (timeout). The usual cause is a combinational loop: a signal "
                "that depends on itself through assign/always_comb logic, or two blocks that trigger each other.")
    lines = [f"- {name}: {n} mismatches, first at time {t}" if n else f"- {name}: OK (no mismatches)"
             for name, (n, t) in res.outputs.items()]
    lines.append(f"Total: {res.mismatches} of {res.samples} samples have at least one mismatching output.")
    table = mismatch_table(res.vcd, res.outputs, cfg) if res.vcd else ""
    if table:
        lines += ["", "### Waveform around the first mismatch",
                  "Each row shows the values just before a clock edge, which is when the testbench compares them "
                  "(times in ps; up to 8 bits as binary, wider as hex)."
                  + (" MISMATCH marks rows where `got` differs from `expected`." if cfg.show_expected else ""),
                  "", table]
    return "\n".join(lines)


def simulate_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    ctx, cfg = runtime.context, runtime.context.config.simulation
    simdir = ctx.workdir / "sim"  # keeps wave.vcd/sim.vvp out of the directory the scorer globs *.sv in
    simdir.mkdir(exist_ok=True)
    res = simulate(ctx.testbench, ctx.reference, [Path("..") / Path(state["code_path"]).name], simdir, cfg.timeout)
    feedback = _sim_feedback(res, ctx)
    counts = f" {res.mismatches}/{res.samples} mismatches" if res.mismatches is not None else ""
    ctx.transcript.event("tool", "simulate", title=f"simulate -> {res.status}{counts}", attempt=state["attempts"],
                         duration_s=res.duration_s, sections={"OUTPUT": "\n\n".join(res.log), "FEEDBACK": feedback},
                         tool="sim", sim=res.summary())

    update = {
        "sim": res.summary(),
        "sim_feedback": feedback,
        "sim_history": [{"sim_attempt": state.get("sim_attempts", 0), "status": res.status,
                         "mismatches": res.mismatches, "samples": res.samples, "signature": sim_signature(res)}],
    }
    best = state.get("best_mismatches")
    if res.mismatches is not None and (best is None or res.mismatches < best):
        update |= {"best_code": state["code"], "best_mismatches": res.mismatches, "best_feedback": feedback}
    return update


def route_sim_node(state: RTLState, runtime: Runtime[RunContext]) -> dict:
    cfg = runtime.context.config.simulation
    sim, fixes = state["sim"], state.get("sim_attempts", 0)
    recent = [h["signature"] for h in state["sim_history"][-cfg.stuck_after:]]
    best = state.get("best_mismatches")
    update = {"sim_note": None}

    if sim["status"] == "pass":
        route, reason = "finalize", "pass: simulation matches the testbench"
    elif sim["status"] == "unknown":
        route, reason = "finalize", "fail: could not parse the simulation result"
    elif fixes >= cfg.max_fixes:
        route, reason = "finalize", f"fail: simulation fix budget exhausted ({fixes}/{cfg.max_fixes}), best {best} mismatches"
    elif len(recent) == cfg.stuck_after and len(set(recent)) == 1:
        route, reason = "finalize", f"fail: same simulation mismatches {cfg.stuck_after}x in a row, best {best} mismatches"
    else:
        route = "sim_fix"
        reason = f"{sim['status']}" + (f" ({sim['mismatches']} mismatches)" if sim["mismatches"] is not None else "")
        reason += f", sim fix {fixes + 1}/{cfg.max_fixes}"
        if best is not None and (sim["mismatches"] is None or sim["mismatches"] > best):
            # The last sim fix made things worse: discard it and fix the best version instead.
            now = sim["mismatches"] if sim["mismatches"] is not None else sim["status"]
            update |= {"code": state["best_code"], "sim_feedback": state["best_feedback"],
                       "sim_note": f"Your last change made the simulation worse ({best} -> {now} mismatches), so it "
                                   "was discarded. Below is the best version so far and its simulation result; "
                                   "try a different fix."}
            reason += f", reverted to the best version ({best} mismatches)"
        elif len(recent) >= 2 and recent[-1] == recent[-2]:
            update["sim_note"] = ("Your last change did not change the simulation result at all: every output "
                                  "still mismatches in exactly the same places, so the cause is somewhere else. "
                                  "Re-read the spec and compare it with the expected values in the table before "
                                  "changing the code.")

    runtime.context.transcript.event("route", "route_sim", title=f"-> {route}: {reason}", attempt=state["attempts"])
    return {"route": route, "route_reason": reason, **update}


# --- LLM nodes ----------------------------------------------------------------------------------


def plan_node(state: RTLState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    msg = _call_llm("plan", prompts.build_plan_messages(state), state, config, runtime.context)
    return {"plan": msg.text, "tokens": _usage(msg)}


def generate_node(state: RTLState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    state = {**state, "attempts": state["attempts"] + 1}
    msg = _call_llm("generate", prompts.build_generate_messages(state), state, config, runtime.context)
    return {"response": msg.text, "response_node": "generate", "attempts": state["attempts"], "restart_hint": None,
            "tokens": _usage(msg)}


def fix_node(state: RTLState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    state = {**state, "attempts": state["attempts"] + 1}
    msg = _call_llm("fix", prompts.build_fix_messages(state), state, config, runtime.context)
    return {"response": msg.text, "response_node": "fix", "attempts": state["attempts"], "tokens": _usage(msg)}


def sim_fix_node(state: RTLState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    # Not counted in `attempts` (the static-fix budget); sim fixes have their own budget.
    msg = _call_llm("sim_fix", prompts.build_sim_fix_messages(state), state, config, runtime.context)
    return {"response": msg.text, "response_node": "sim_fix", "sim_attempts": state.get("sim_attempts", 0) + 1,
            "tokens": _usage(msg)}


# --- graph --------------------------------------------------------------------------------------


def build_graph():
    g = StateGraph(RTLState, context_schema=RunContext)
    g.add_node("parse_spec", parse_spec_node)
    g.add_node("plan", plan_node)
    g.add_node("generate", generate_node)
    g.add_node("write_code", write_code_node)
    g.add_node("check", check_node)
    g.add_node("route", route_node)
    g.add_node("fix", fix_node)
    g.add_node("finalize", finalize_node)
    g.add_node("simulate", simulate_node)
    g.add_node("route_sim", route_sim_node)
    g.add_node("sim_fix", sim_fix_node)

    g.add_edge(START, "parse_spec")
    g.add_edge("parse_spec", "plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "write_code")
    g.add_edge("write_code", "check")
    g.add_edge("check", "route")
    g.add_conditional_edges("route", lambda s: s["route"], ["finalize", "fix", "generate", "simulate"])
    g.add_edge("fix", "write_code")
    g.add_edge("simulate", "route_sim")
    g.add_conditional_edges("route_sim", lambda s: s["route"], ["finalize", "sim_fix"])
    g.add_edge("sim_fix", "write_code")
    g.add_edge("finalize", END)
    return g.compile()
