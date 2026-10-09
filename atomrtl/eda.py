"""EDA checks as plain Python (no LLM): iverilog compile, pyslang lint, interface check.

Each check returns structured Diagnostics. The full raw tool output is kept on the CmdResult /
LintResult for the transcript; the LLM only ever sees format_diagnostics()'s capped summary.
"""

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from collections.abc import Iterable
from pathlib import Path

import pyslang
from pyslang import analysis, ast, syntax

from atomrtl.spec import Port


@dataclass(frozen=True)
class Diagnostic:
    tool: str  # iverilog | slang | interface | agent
    category: str  # syntax | elab | lint | interface | no_code
    severity: str  # error | warning
    message: str
    line: int | None = None
    code: str = ""  # tool-specific id, e.g. slang's "WidthTruncate"


@dataclass
class CmdResult:
    cmd: list[str]
    returncode: int | None
    output: str  # stdout + stderr, untruncated
    duration_s: float
    timed_out: bool = False

    @property
    def cmdline(self) -> str:
        return " ".join(self.cmd)


def run_cmd(cmd: list[str], cwd: Path, timeout: float) -> CmdResult:
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return CmdResult(cmd, p.returncode, p.stdout + p.stderr, time.monotonic() - start)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return CmdResult(cmd, None, out, time.monotonic() - start, timed_out=True)


# --- iverilog -----------------------------------------------------------------------------------

_IVERILOG_CONT_RE = re.compile(r"^[^:\s][^:]*:\d+:\s+:")
_IVERILOG_LINE_RE = re.compile(r"^(?P<file>[^:\s][^:]*):(?P<line>\d+):\s*(?P<kind>error|warning|sorry)?:?\s*(?P<msg>.*)$")


def run_iverilog(files: list[Path], cwd: Path, timeout: float = 30) -> tuple[CmdResult, list[Diagnostic]]:
    """Compile and elaborate with Icarus (the same compiler used for scoring), no simulation output."""
    res = run_cmd(["iverilog", "-g2012", "-t", "null", *map(str, files)], cwd, timeout)
    if res.timed_out:
        return res, [Diagnostic("iverilog", "elab", "error", f"iverilog timed out after {timeout}s")]

    diags = []
    for raw in res.output.splitlines():
        m = _IVERILOG_LINE_RE.match(raw.strip())
        if not m:
            continue
        kind, msg = m["kind"], m["msg"].strip()
        if _IVERILOG_CONT_RE.match(raw) and diags and diags[-1].line == int(m["line"]):
            # Continuation line ("file:5:      : Expression is: ...") of the previous message.
            prev = diags.pop()
            diags.append(Diagnostic(prev.tool, prev.category, prev.severity, f"{prev.message} {msg}", prev.line))
            continue
        # The exit code decides: with exit 0 everything printed is advisory, including "sorry:"
        # limitation notices such as "constant selects in always_* processes are not currently supported".
        if res.returncode == 0 or kind == "warning":
            severity = "warning"
        else:
            severity = "error"
        category = "syntax" if "syntax error" in msg.lower() or "syntax error" in raw.lower() else "elab"
        diags.append(Diagnostic("iverilog", category, severity, msg or "syntax error", int(m["line"])))

    # iverilog prints a bare "syntax error" plus a specific message for the same line; keep the specific one.
    specific = {d.line for d in diags if d.message != "syntax error"}
    diags = [d for d in diags if not (d.message == "syntax error" and d.line in specific)]

    if res.returncode and not any(d.severity == "error" for d in diags):
        # Failed without a file:line message (e.g. "Unknown module type" summaries, missing top).
        diags.append(Diagnostic("iverilog", "elab", "error", res.output.strip() or f"exit code {res.returncode}"))
    return res, diags


# --- pyslang ------------------------------------------------------------------------------------

# Extra guidance appended to warnings promoted to errors.
_PROMOTED_HINTS = {
    "WidthTruncate": (
        " (treated as an error: high bits are silently lost, which usually means the logic is wrong. "
        "Recheck the logic; if dropping the high bits is really intended, make it explicit with a slice, "
        "e.g. out = sum[1:0])"
    ),
}

_TRUNC_FROM_RE = re.compile(r"truncates from (\d+) to")


def _is_int_truncation(code: str, message: str) -> bool:
    """A truncation from 32 bits almost always comes from integer arithmetic (`i - 1` with an int
    loop variable, unsized literals) and is intended, so it stays a warning even when promoted."""
    m = _TRUNC_FROM_RE.search(message)
    return code == "WidthTruncate" and m is not None and m.group(1) == "32"


_DIRECTIONS = {"In": "input", "Out": "output", "InOut": "inout", "Ref": "ref"}


@dataclass
class LintResult:
    report: str  # slang's own formatted output (with source carets), untruncated
    diagnostics: list[Diagnostic]
    modules: dict[str, tuple[Port, ...]] = field(default_factory=dict)  # top-level modules → ports
    duration_s: float = 0.0


def lint_slang(files: list[Path], as_errors: Iterable[str] = ()) -> LintResult:
    """Parse + elaborate with slang (stricter than Icarus) and extract the top-level module ports.

    `as_errors`: slang warning codes (e.g. "WidthTruncate") to report as errors, so they must be fixed.
    """
    as_errors = set(as_errors)
    start = time.monotonic()
    sm = pyslang.SourceManager()
    comp = ast.Compilation()
    for f in files:
        comp.addSyntaxTree(syntax.SyntaxTree.fromFile(str(f), sm))

    engine = pyslang.DiagnosticEngine(sm)
    engine.setWarningOptions(["everything", "no-newline-eof"])
    client = pyslang.TextDiagnosticClient()
    engine.addClient(client)

    raw = list(comp.getAllDiagnostics())
    # slang 10+ runs its dataflow checks (multiple drivers, inferred latches) in a separate analysis
    # pass. Only run it on code that compiles; on broken code it would just add follow-on noise.
    if not any(d.isError() for d in raw):
        comp.freeze()
        manager = analysis.AnalysisManager()
        manager.analyze(comp)
        raw += list(manager.getDiagnostics())

    diags = []
    for d in raw:
        engine.issue(d)
        sev = engine.getSeverity(d.code, d.location)
        code = str(d.code).removeprefix("DiagCode(").removesuffix(")")
        message = engine.formatMessage(d)
        if sev in (pyslang.DiagnosticSeverity.Error, pyslang.DiagnosticSeverity.Fatal):
            severity = "error"
        elif sev == pyslang.DiagnosticSeverity.Warning:
            severity = "warning"
            if code in as_errors and not _is_int_truncation(code, message):
                severity = "error"
                message += _PROMOTED_HINTS.get(code, " (treated as an error)")
        else:
            continue  # notes / ignored
        diags.append(Diagnostic("slang", "lint", severity, message, sm.getLineNumber(d.location), code))

    modules = {}
    for inst in comp.getRoot().topInstances:
        ports = []
        for p in inst.body.portList:
            direction = _DIRECTIONS.get(str(p.direction).split(".")[-1], str(p.direction))
            width = getattr(getattr(p, "type", None), "bitWidth", 0) or 0
            ports.append(Port(p.name, direction, width))
        modules[inst.name] = tuple(ports)

    return LintResult(client.getString(), diags, modules, time.monotonic() - start)


# --- interface check ----------------------------------------------------------------------------


def check_interface(
    expected_name: str | None, expected_ports: tuple[Port, ...] | None, modules: dict[str, tuple[Port, ...]]
) -> list[Diagnostic]:
    """Compare the elaborated top module against the spec's name and ports."""
    if not expected_name:
        return []

    def err(msg: str) -> Diagnostic:
        return Diagnostic("interface", "interface", "error", msg)

    if expected_name not in modules:
        found = ", ".join(sorted(modules)) or "none"
        return [err(f"No top-level module named '{expected_name}' (found: {found}). The module must be named "
                    f"exactly '{expected_name}' and must not be instantiated by another module.")]
    if not expected_ports:
        return []

    actual = {p.name: p for p in modules[expected_name]}
    diags = []
    for p in expected_ports:
        a = actual.get(p.name)
        if a is None:
            diags.append(err(f"Missing port '{p.name}' ({p.direction}, {p.width} bit{'s' if p.width > 1 else ''})."))
            continue
        if a.direction != p.direction:
            # A warning, not an error: specs occasionally get a direction wrong (verilog-eval Prob031
            # lists a flip-flop's q as an input), and forcing it would break otherwise-correct code.
            diags.append(Diagnostic("interface", "interface", "warning",
                                    f"Port '{p.name}' is declared as {a.direction}, but the spec lists it as "
                                    f"{p.direction}. Double-check the spec."))
        if a.width != p.width:
            diags.append(err(f"Port '{p.name}' must be {p.width} bit(s) wide, but is {a.width} bit(s)."))
    for name in actual.keys() - {p.name for p in expected_ports}:
        diags.append(err(f"Unexpected port '{name}': the spec's interface does not have it."))
    return diags


# --- formatting for the LLM ---------------------------------------------------------------------


def error_signature(diags: list[Diagnostic]) -> str:
    """Stable id of the set of errors (ignoring line numbers), used to detect a stuck fix loop.

    The "code unchanged" note is excluded so an unchanged fix still counts as the same errors.
    """
    keys = sorted({f"{d.tool}:{d.message}" for d in diags if d.severity == "error" and d.category != "no_change"})
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()[:12] if keys else ""


@dataclass(frozen=True)
class FixHint:
    """How to fix a known diagnostic (tool messages say what is wrong, not how to fix it).

    Same idea as RTLFixer's hand-written compiler-error guidance: a deterministic lookup, no LLM.
    Matches on slang's diagnostic `code` when given (exact), else on a message regex (iverilog has no codes).
    """

    hint: str
    code: str | None = None
    pattern: re.Pattern | None = None

    def matches(self, d: Diagnostic) -> bool:
        return (self.code is not None and d.code == self.code) or (
            self.pattern is not None and self.pattern.search(d.message) is not None
        )


FIX_HINTS: list[FixHint] = [
    FixHint("A replication inside a concatenation needs its own braces: write {{24{in[7]}}, in}, not {24{in[7]}, in}.",
            pattern=re.compile(r"repeat concatenation")),
    FixHint("Assigning a plain constant to an enum variable needs a cast (state <= state_t'(2'b01)); simpler: "
            "declare the state as a logic vector and the states as localparam constants.",
            pattern=re.compile(r"no implicit conversion from .* to '\w+'|requires an explicit cast")),
    FixHint("Declare every internal signal (as logic) before it is used, and check its spelling against the ports.",
            pattern=re.compile(r"Unable to bind wire/reg/memory|use of undeclared identifier")),
    FixHint("Only outputs and internal logic signals can be assigned; never assign an input port.",
            pattern=re.compile(r"is not a valid l-value|cannot assign to input")),
    # slang codes made blocking via AgentConfig.blocking_warnings:
    FixHint("Each variable may be assigned in only one always block. Put all assignments to it in one always_ff "
            "(registers) or one always_comb (next-state / output logic), never both.",
            code="MultipleAlwaysAssigns"),
    FixHint("Give every variable assigned in an always_comb a value on every path: assign a default at the top "
            "of the block (e.g. next_state = state;) or add a default: branch to every case.",
            code="InferredLatch"),
    FixHint("Two case items have the same value, so the second can never be taken; make every item distinct.",
            code="CaseDup"),
    FixHint("A sized literal has more digits than its width (e.g. 4'b10101); fix the width or the digits.",
            code="VectorLiteralOverflow"),
    FixHint("A constant changes value when converted to the target width; size the constant to match it.",
            code="ConstantConversion"),
    FixHint("An undeclared name was silently turned into a 1-bit wire, usually a typo: declare it as logic "
            "with the right width, or fix the spelling.",
            code="ImplicitNet"),
    FixHint("An input port is being driven inside the module; inputs are read-only.",
            code="InputPortCoercion"),
]


def fix_hints(diags: list[Diagnostic]) -> list[str]:
    """The fix suggestions that apply to these diagnostics' errors, each listed once."""
    errors = [d for d in diags if d.severity == "error"]
    return [h.hint for h in FIX_HINTS if any(h.matches(d) for d in errors)]


def format_diagnostics(diags: list[Diagnostic], code: str, limit: int = 8) -> str:
    """Errors first, deduplicated, capped, each with the offending source lines, then fix hints."""
    seen, error_lines, unique = set(), set(), []
    for d in sorted(diags, key=lambda d: (d.severity != "error", d.line or 0)):
        key = (d.tool, d.message, d.line)
        if key in seen:
            continue
        seen.add(key)
        if d.severity == "error" and d.line is not None:
            # Only the first error per tool and line: the rest are parser follow-on errors
            # ("expected ';'", "unexpected '}'") caused by it, which just add noise.
            if (d.tool, d.line) in error_lines:
                continue
            error_lines.add((d.tool, d.line))
        unique.append(d)

    lines = code.splitlines()
    blocks = []
    for d in unique[:limit]:
        loc = f" line {d.line}" if d.line else ""
        block = f"[{d.severity.upper()}] {d.tool}{loc}: {d.message}"
        if d.line and 1 <= d.line <= len(lines):
            lo, hi = max(1, d.line - 2), min(len(lines), d.line + 2)
            block += "\n" + "\n".join(
                f"{'>>' if n == d.line else '  '} {n:4d} | {lines[n - 1]}" for n in range(lo, hi + 1)
            )
        blocks.append(block)

    if len(unique) > limit:
        rest = unique[limit:]
        n_err = sum(d.severity == "error" for d in rest)
        blocks.append(f"... ({n_err} more error(s), {len(rest) - n_err} more warning(s) omitted)")
    if hints := fix_hints(diags):
        blocks.append("Fix suggestions:\n" + "\n".join(f"- {h}" for h in hints))
    return "\n\n".join(blocks)
