"""Prompt builders, one per LLM node. Each takes only the state fields it needs (no shared history)."""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from atomrtl.state import RTLState

SYSTEM_PROMPT = """\
You are an expert SystemVerilog RTL designer. You write clean, synthesizable SystemVerilog-2012 that \
compiles with Icarus Verilog (iverilog -g2012) and passes a strict linter.

Rules:
- Use always_ff for sequential logic, and always_comb or assign for combinational logic. Use \
non-blocking assignments (<=) inside always_ff, and blocking assignments (=) inside always_comb.
- Declare every port and internal signal as logic. Never declare a signal inside an always block.
- Unless the spec says otherwise, sequential logic triggers on the positive edge of clk.
- Synchronous reset: only the clock is in the sensitivity list (@(posedge clk)), and reset is checked \
inside the block. Use an asynchronous reset only when the spec explicitly asks for one.
- Use exactly the module name, port names, directions and widths from the spec. Never rename, add or \
remove ports.
- Inside the module you may freely declare helper logic signals, localparam constants and functions \
for intermediate values, counters, state registers or sub-expressions. Declare them at module level \
(not inside an always block) and use clear names. Only the port list must match the spec exactly.
- For state machines, encode states with localparam constants on a logic vector, not enum types \
(Icarus rejects assigning plain constants to enum variables).
- Watch bit widths: size every constant and make both sides of each assignment the same width.
- When the spec contains a Karnaugh map, truth table or waveform, read the row/column labels and \
their ordering carefully before deriving the logic.
- If the spec is ambiguous, pick the most standard interpretation and state the assumption in one line.\
"""

_CODE_FORMAT = (
    "Reply with the complete design in a single ```systemverilog code block containing every module "
    "needed. Do not include a testbench. Do not output any other code block."
)


def _spec_section(state: RTLState) -> str:
    parts = [f"## Spec\n{state['spec'].strip()}"]
    if state.get("header"):
        parts.append(
            "## Required interface (use this module header exactly)\n"
            f"```systemverilog\n{state['header']}\n```"
        )
    elif state.get("module_name"):
        parts.append(f"## Required module name\n{state['module_name']}")
    return "\n\n".join(parts)


def build_plan_messages(state: RTLState) -> list[BaseMessage]:
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"{_spec_section(state)}\n\n"
            "Write a short implementation plan (3-6 bullet points): the registers and state needed, "
            "the logic equations or state transitions, and any tricky details in the spec. "
            "Do not write any code yet."
        ),
    ]


def build_generate_messages(state: RTLState) -> list[BaseMessage]:
    parts = [_spec_section(state)]
    if state.get("plan"):
        parts.append(f"## Plan\n{state['plan'].strip()}")
    if state.get("restart_hint"):
        parts.append(f"## Note\n{state['restart_hint']}")
    parts.append(f"Implement the design.\n{_CODE_FORMAT}")
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage("\n\n".join(parts))]


def build_fix_messages(state: RTLState) -> list[BaseMessage]:
    code = state.get("code")
    current = (
        f"## Current code\n```systemverilog\n{code.rstrip()}\n```"
        if code
        else f"## Your previous response (no usable code block was found in it)\n{state.get('response', '').strip()}"
    )
    feedback = "\n\n".join(f"### From {f['source']} ({f['kind']})\n{f['text']}" for f in state.get("feedback", []))
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"{_spec_section(state)}\n\n{current}\n\n## Problems found\n{feedback}\n\n"
            "First explain the cause of each error in 2-4 short bullet points. Then fix every error "
            "while keeping the required interface and the existing signal names.\n"
            f"{_CODE_FORMAT}"
        ),
    ]
