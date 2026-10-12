"""Compact waveform view of a failing simulation, for the sim_fix LLM node.

Reads the wave.vcd that verilog-eval testbenches dump (inputs, `<out>_ref` and `<out>_dut` in scope `tb`,
plus the clock) and renders the values the testbench compared at each clock edge around the first mismatch.
The VCD parser follows TigressRTL's src/vcd.py.
"""

import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

from atomrtl.config import SimConfig

_VAR_RE = re.compile(r"^\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)")
_SCALAR_RE = re.compile(r"^([01xXzZ])(\S+)$")
_VECTOR_RE = re.compile(r"^[bB]([01xXzZ]+)\s+(\S+)$")


@dataclass
class Signal:
    name: str  # dotted hierarchical name, e.g. "tb.out_ref", "tb.stim1.clk"
    width: int
    times: list[int] = field(default_factory=list)  # change times, ascending
    values: list[str] = field(default_factory=list)  # bit strings, padded to `width`

    def before(self, t: int) -> str:
        """Value in effect just before time t: what a process triggered at t reads (the testbench's compare)."""
        i = bisect_left(self.times, t)
        return self.values[i - 1] if i else "x" * self.width


def _pad(bits: str, width: int) -> str:
    """VCD drops leading zeros; extend per the VCD rules (x/z extend as themselves, 0/1 with 0)."""
    bits = bits.lower()
    fill = bits[0] if bits[0] in "xz" else "0"
    return bits.rjust(width, fill)[-width:]


def parse_vcd(path: Path) -> dict[str, Signal]:
    signals: dict[str, Signal] = {}
    by_id: dict[str, list[Signal]] = {}
    scope: list[str] = []
    now = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("$scope"):
                scope.append(line.split()[2])
            elif line.startswith("$upscope"):
                scope = scope[:-1]
            elif line.startswith("$var"):
                if m := _VAR_RE.match(line):
                    width, var_id, name = int(m[1]), m[2], ".".join([*scope, m[3]])
                    sig = signals.setdefault(name, Signal(name, width))
                    by_id.setdefault(var_id, []).append(sig)
            elif line.startswith("#"):
                now = int(line[1:]) if line[1:].isdigit() else now
            elif not line.startswith("$"):
                m = _VECTOR_RE.match(line) or _SCALAR_RE.match(line)
                if not m:
                    continue  # real-valued changes etc.
                for sig in by_id.get(m[2], []):
                    value = _pad(m[1], sig.width)
                    if sig.times and sig.times[-1] == now:
                        sig.values[-1] = value
                    else:
                        sig.times.append(now)
                        sig.values.append(value)
    return signals


def format_value(bits: str, width: int) -> str:
    """1 bit as-is, up to 8 bits as sized binary, wider as sized hex (binary if it has x/z bits)."""
    if width == 1:
        return bits
    if set(bits) <= {"x"}:
        return "x"
    if width <= 8 or set(bits) - {"0", "1"}:
        return f"{width}'b{bits}"
    return f"{width}'h{int(bits, 2):0{(width + 3) // 4}x}"


def _mismatch(expected: str, got: str) -> bool:
    """The testbench's rule: an x bit in the reference matches anything, otherwise bits must be identical."""
    return any(e in "01" and e != g for e, g in zip(expected, got))


def mismatch_table(vcd: Path, outputs: dict[str, tuple[int, int | None]], cfg: SimConfig) -> str:
    """Table of inputs and failing outputs at each clock edge around the earliest first mismatch ("" if n/a)."""
    failing = [o for o, (n, t) in outputs.items() if n and t is not None]
    if not failing:
        return ""
    signals = parse_vcd(vcd)
    top = {name.split(".", 1)[1]: s for name, s in signals.items() if name.count(".") == 1 and name.startswith("tb.")}
    failing = [o for o in failing if f"{o}_ref" in top and f"{o}_dut" in top]
    if not failing:
        return ""
    first = min(outputs[o][1] for o in failing)

    clock = top.get("clk") or signals.get("tb.stim1.clk")
    if clock is not None and len(clock.times) > 2:
        edges = clock.times
    else:  # no clock: every time anything changed
        clock = None
        edges = sorted({t for s in top.values() for t in s.times} | {first})
    i = bisect_left(edges, first)
    lo, hi = max(0, i - 2 * cfg.cycles_before), min(len(edges), i + 2 * cfg.cycles_after + 1)
    lo = max(lo, hi - cfg.max_rows)  # keep the rows closest to (and after) the first mismatch
    times = edges[lo:hi]

    inputs = [n for n in top if not n.endswith(("_ref", "_dut")) and n not in ("tb_mismatch", "clk")]
    header = ["time"] + (["clk"] if clock else []) + inputs
    for o in failing:
        header += [f"{o} expected", f"{o} got"] if cfg.show_expected else [f"{o} got"]
    rows, unknown = [], set()
    for t in times:
        row = [str(t)]
        if clock:
            row.append("rise" if clock.before(t + 1) == "1" else "fall")
        row += [format_value(top[n].before(t), top[n].width) for n in inputs]
        bad = False
        for o in failing:
            ref, dut = top[f"{o}_ref"], top[f"{o}_dut"]
            exp, got = ref.before(t), dut.before(t)
            bad |= _mismatch(exp, got)
            if "x" in got and "x" not in exp and t >= first:
                unknown.add(o)
            row += [format_value(exp, ref.width), format_value(got, dut.width)] if cfg.show_expected else [format_value(got, dut.width)]
        note = ("MISMATCH" if bad and cfg.show_expected else "") + ("  <- first mismatch" if t == first else "")
        rows.append(row + [note.strip()])

    widths = [max(len(r[c]) for r in [header + [""], *rows]) for c in range(len(header) + 1)]
    lines = [" | ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in [header + [""], *rows]]
    lines.insert(1, "-" * len(lines[0]))
    if unknown:
        lines += ["", f"Note: {', '.join(sorted(unknown))} is x (unknown) where a value is expected. The usual causes: "
                      "the signal is never driven, or it is only set in a declaration initializer "
                      "(`logic v = expr;` is evaluated once at time 0, not continuously: use `assign` or always_comb), "
                      "or it depends on a register that is never reset."]
    return "\n".join(lines)
