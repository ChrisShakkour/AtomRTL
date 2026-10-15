"""Run the RTL-creation agent on one spec.

    python -m atomrtl.cli --spec path/to/spec.txt [--top TopModule] [--config cfg.yaml]
                          [--mode agent|oneshot] [--oneshot-prompt agent|verilog_eval] [--provider ollama|anthropic] [--model NAME]
                          [--sim-tb TB.sv --sim-ref REF.sv] [--workdir DIR] [--no-server] [-v]

Creates runs/<timestamp>/<spec-name>/ with the generated .sv, result.json and the full transcript.
With --sim-tb/--sim-ref, a design that passes the static checks is also simulated against that
testbench and reference, and an LLM fixes mismatches (see `simulation` in atomrtl/config.py).
With the ollama provider, starts the Ollama server if it isn't running (and stops it afterwards), unless
--no-server. The anthropic provider needs ANTHROPIC_API_KEY.
--mode oneshot is the baseline: one LLM call with the spec, the reply written to the file unchecked.
Exit code: 0 = pass (oneshot: code written), 1 = finished without a passing design, 2 = crashed.
"""

import argparse
import shlex
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from atomrtl.config import MODES, ONESHOT_PROMPTS, PROVIDERS, AgentConfig, config_to_dict, load_config
from atomrtl.graph import build_graph
from atomrtl.llm import llm_backend
from atomrtl.state import RunContext
from atomrtl.transcript import Transcript, TranscriptCallback


def new_run_dir(runs_dir: Path) -> Path:
    run_dir = runs_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_info(run_dir: Path, cfg: AgentConfig) -> None:
    (run_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))


def run_agent(spec: str, workdir: Path, cfg: AgentConfig, top: str | None = None, verbose: bool = False,
              testbench: Path | None = None, reference: Path | None = None) -> dict:
    """Run the graph on one spec inside `workdir` (expects the LLM backend to be up). Returns the final state.

    `testbench`/`reference`: simulate the design against them once it passes the static checks
    (only when `cfg.simulation.enabled`).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "prompt.txt").write_text(spec)
    transcript = Transcript(workdir, echo=verbose)
    start = time.monotonic()
    try:
        return build_graph(cfg.mode).invoke(
            {"spec": spec, "top": top},
            config={
                "callbacks": [TranscriptCallback(transcript)],
                # ~4 graph steps per attempt (fix, write_code, check, route) plus setup, and ~6 per sim fix
                # (sim_fix, write_code, check, route, simulate, route_sim).
                "recursion_limit": 10 + 5 * cfg.max_attempts + 7 * (cfg.simulation.max_fixes + 1),
            },
            context=RunContext(config=cfg, workdir=workdir, transcript=transcript,
                               testbench=testbench and testbench.resolve(), reference=reference and reference.resolve()),
        )
    except Exception as e:
        transcript.event("error", "agent", title=repr(e))
        raise
    finally:
        transcript.event("state", "agent", title="run finished", duration_s=time.monotonic() - start)
        transcript.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path, help="spec text file")
    ap.add_argument("--top", help="required top module name (default: taken from the spec)")
    ap.add_argument("--config", type=Path, help="YAML config (see atomrtl/config.py)")
    ap.add_argument("--mode", choices=MODES,
                    help="agent (full graph) or oneshot (single LLM call, no checks) (default: from the config, else agent)")
    ap.add_argument("--oneshot-prompt", choices=ONESHOT_PROMPTS,
                    help="mode oneshot: agent (the agent's generate prompt, no plan) or verilog_eval "
                         "(verilog-eval's own prompt) (default: from the config, else agent)")
    ap.add_argument("--provider", choices=PROVIDERS, help="LLM backend (default: from the config, else ollama)")
    ap.add_argument("--model", help="override the active provider's model")
    ap.add_argument("--sim-tb", type=Path, help="reference testbench to simulate the design against (enables simulation)")
    ap.add_argument("--sim-ref", type=Path, help="reference model the testbench instantiates (with --sim-tb)")
    ap.add_argument("--runs-dir", type=Path, help="where to create the run directory (default: runs/)")
    ap.add_argument("--workdir", type=Path, help="run directly in this directory instead of a new runs/<timestamp>/")
    ap.add_argument("--no-server", action="store_true",
                    help="only connect to an already-running Ollama server; never start or stop one "
                         "(used by the benchmark runner, which manages the shared server)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each transcript event live")
    args = ap.parse_args()

    cfg = load_config(args.config, runs_dir=args.runs_dir, provider=args.provider, mode=args.mode,
                      oneshot_prompt=args.oneshot_prompt)
    if args.model:
        cfg = cfg.with_model(args.model)
    if bool(args.sim_tb) != bool(args.sim_ref):
        ap.error("--sim-tb and --sim-ref go together")
    if args.sim_tb and cfg.mode == "oneshot":
        ap.error("--mode oneshot never simulates; drop --sim-tb/--sim-ref")
    if args.sim_tb:
        cfg = replace(cfg, simulation=replace(cfg.simulation, enabled=True))
    elif cfg.simulation.enabled:
        ap.error("simulation is enabled in the config, but no --sim-tb/--sim-ref was given")

    if args.workdir:
        workdir = args.workdir
    else:
        run_dir = new_run_dir(cfg.runs_dir)
        write_run_info(run_dir, cfg)
        workdir = run_dir / args.spec.name.removesuffix(".txt").removesuffix("_prompt")

    try:
        with nullcontext() if args.no_server else llm_backend(cfg):
            final = run_agent(args.spec.read_text(), workdir, cfg, top=args.top, verbose=args.verbose,
                              testbench=args.sim_tb, reference=args.sim_ref)
    except Exception:
        traceback.print_exc()
        sys.exit(2)  # crashed, distinct from 1 = finished without a passing design

    print(f"\n{final['status'].upper()}: {final.get('route_reason')}")
    print(f"code:       {final.get('code_path')}")
    print(f"transcript: {workdir / 'transcript.log'}")
    sys.exit(0 if final["status"] in ("pass", "unchecked") else 1)


if __name__ == "__main__":
    main()
