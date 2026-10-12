from pathlib import Path

from atomrtl.config import SimConfig
from atomrtl.eda import simulate
from atomrtl.vcd import format_value, mismatch_table, parse_vcd

FIXTURES = Path(__file__).parent / "fixtures"
NAND = "module TopModule(input logic a, input logic b, output logic out);\n  assign out = ~(a & b);\nendmodule\n"


def _failing_sim(tmp_path):
    (tmp_path / "TopModule.sv").write_text(NAND)
    res = simulate(FIXTURES / "nor_test.sv", FIXTURES / "nor_ref.sv", [Path("TopModule.sv")], tmp_path)
    assert res.status == "fail"
    return res


def test_parse_vcd_pads_and_scopes(tmp_path):
    vcd = tmp_path / "w.vcd"
    vcd.write_text("$scope module tb $end\n$var wire 4 ! q [3:0] $end\n$upscope $end\n$enddefinitions $end\n"
                   "#0\n$dumpvars\nbx !\n$end\n#5\nb10 !\n#10\nb1x !\n")
    q = parse_vcd(vcd)["tb.q"]
    assert q.values == ["xxxx", "0010", "001x"]
    assert q.before(5) == "xxxx" and q.before(6) == "0010" and q.before(100) == "001x"


def test_format_value():
    assert format_value("1", 1) == "1"
    assert format_value("0101", 4) == "4'b0101"
    assert format_value("0000000011111111", 16) == "16'h00ff"
    assert format_value("xxxx", 4) == "x"
    assert format_value("0000000x11111111", 16) == "16'b0000000x11111111"


def test_table_marks_first_mismatch(tmp_path):
    res = _failing_sim(tmp_path)
    first = res.outputs["out"][1]
    table = mismatch_table(res.vcd, res.outputs, SimConfig())
    header, rows = table.splitlines()[0], table.splitlines()[2:]
    assert [c.strip() for c in header.split("|")][:6] == ["time", "clk", "a", "b", "out expected", "out got"]
    first_row = next(r for r in rows if "<- first mismatch" in r)
    assert first_row.startswith(str(first)) and "MISMATCH" in first_row
    # Our own expected-vs-got comparison agrees with the testbench: no MISMATCH row before the first one.
    assert not any("MISMATCH" in r for r in rows[: rows.index(first_row)])


def test_table_window_and_hidden_expected(tmp_path):
    res = _failing_sim(tmp_path)
    table = mismatch_table(res.vcd, res.outputs, SimConfig(show_expected=False, cycles_before=1, cycles_after=1))
    assert "expected" not in table and "MISMATCH" not in table and "<- first mismatch" in table
    assert len(table.splitlines()) - 2 <= 5  # 2 edges per cycle before/after + the mismatch edge
    capped = mismatch_table(res.vcd, res.outputs, SimConfig(max_rows=3))
    assert len(capped.splitlines()) - 2 == 3


def test_table_empty_without_failures(tmp_path):
    assert mismatch_table(tmp_path / "missing.vcd", {"out": (0, None)}, SimConfig()) == ""


def test_table_notes_unknown_outputs(tmp_path):
    (tmp_path / "TopModule.sv").write_text(
        "module TopModule(input logic a, input logic b, output logic out);\n  logic n = ~(a | b);\n  assign out = n;\nendmodule\n")
    res = simulate(FIXTURES / "nor_test.sv", FIXTURES / "nor_ref.sv", [Path("TopModule.sv")], tmp_path)
    assert res.status == "fail"
    assert "Note: out is x (unknown)" in mismatch_table(res.vcd, res.outputs, SimConfig())
    assert "Note:" not in mismatch_table(_failing_sim(tmp_path).vcd, _failing_sim(tmp_path).outputs, SimConfig())
