from atomrtl.eda import check_interface, error_signature, format_diagnostics, lint_slang, run_iverilog
from atomrtl.config import AgentConfig
from atomrtl.spec import Port

GOOD = """module TopModule (
  input  logic       clk,
  input  logic [7:0] in,
  output logic [7:0] out
);
  always_ff @(posedge clk) out <= in;
endmodule
"""
PORTS = (Port("clk", "input"), Port("in", "input", 8), Port("out", "output", 8))


def _write(tmp_path, code):
    p = tmp_path / "TopModule.sv"
    p.write_text(code)
    return p


def test_good_code_is_clean(tmp_path):
    p = _write(tmp_path, GOOD)
    res, diags = run_iverilog([p], tmp_path)
    assert res.returncode == 0 and diags == []
    lint = lint_slang([p])
    assert not [d for d in lint.diagnostics if d.severity == "error"]
    assert lint.modules["TopModule"] == PORTS
    assert check_interface("TopModule", PORTS, lint.modules) == []


def test_syntax_error_reported_once_with_line(tmp_path):
    p = _write(tmp_path, GOOD.replace("out <= in;", "out <= {24{in[7]}, in};"))
    res, diags = run_iverilog([p], tmp_path)
    assert res.returncode != 0
    errs = [d for d in diags if d.severity == "error"]
    assert errs and all(d.line == 6 for d in errs)
    assert not any(d.message == "syntax error" for d in errs)  # bare duplicate dropped


def test_slang_catches_enum_assignment(tmp_path):
    code = GOOD.replace(
        "  always_ff", "  typedef enum logic [1:0] {A, B} st_t;\n  st_t s;\n  always_ff @(posedge clk) s <= 2'b01;\n  always_ff"
    )
    lint = lint_slang([_write(tmp_path, code)])
    assert any(d.severity == "error" and "conversion" in d.message for d in lint.diagnostics)


def test_interface_mismatches():
    mods = {"TopModule": (Port("clk", "input"), Port("in", "input", 4), Port("q", "output", 8), Port("out", "input", 8))}
    msgs = {d.severity + ": " + d.message for d in check_interface("TopModule", PORTS, mods)}
    assert any(m.startswith("error: Port 'in' must be 8 bit") for m in msgs)
    assert any(m.startswith("error: Unexpected port 'q'") for m in msgs)
    assert any(m.startswith("warning: Port 'out' is declared as input") for m in msgs)
    assert check_interface("TopModule", PORTS, {"top": ()})[0].message.startswith("No top-level module named 'TopModule'")
    assert check_interface(None, None, {}) == []


def test_format_and_signature(tmp_path):
    p = _write(tmp_path, GOOD.replace("out <= in;", "out <= missing_sig;"))
    _, diags = run_iverilog([p], tmp_path)
    text = format_diagnostics(diags, p.read_text(), limit=1)
    assert ">>    6 |" in text
    # Same errors on a different line → same signature (line numbers are ignored).
    _, diags2 = run_iverilog([_write(tmp_path, "\n" + GOOD.replace("out <= in;", "out <= missing_sig;"))], tmp_path)
    assert error_signature(diags) == error_signature(diags2) != ""


def test_iverilog_continuation_lines_merged(tmp_path):
    p = _write(tmp_path, "module TopModule (input logic [2:0] in, output logic [1:0] out);\n"
                         "  assign out = '{in[0], in[1], in[2]};\nendmodule\n")
    _, diags = run_iverilog([p], tmp_path)
    assert any("assignment patterns" in d.message and "Expression is:" in d.message for d in diags)
    assert not any(d.message.startswith("Expression is:") for d in diags)


def test_width_truncate_can_be_promoted_to_error(tmp_path):
    p = _write(tmp_path, "module TopModule (input logic [2:0] in, output logic [1:0] out);\n"
                         "  assign out = {in[0], in[1], in[2]};\nendmodule\n")
    plain = lint_slang([p]).diagnostics
    assert [(d.severity, d.code) for d in plain] == [("warning", "WidthTruncate")]
    promoted = lint_slang([p], as_errors=["WidthTruncate"]).diagnostics
    assert [(d.severity, d.code) for d in promoted] == [("error", "WidthTruncate")]
    assert "explicit with a slice" in promoted[0].message
    # An explicit slice is intended truncation: no diagnostic at all.
    p.write_text("module TopModule (input logic [2:0] in, output logic [1:0] out);\n"
                 "  logic [2:0] s;\n  assign s = in;\n  assign out = s[1:0];\nendmodule\n")
    assert not lint_slang([p], as_errors=["WidthTruncate"]).diagnostics


def test_iverilog_sorry_with_exit_0_is_a_warning(tmp_path):
    # Legal code; Icarus prints a "sorry:" limitation notice but compiles it (exit 0).
    p = _write(tmp_path, "module TopModule (input logic [3:0] y, input logic w, output logic z);\n"
                         "  always_comb z = y[1] & w;\nendmodule\n")
    res, diags = run_iverilog([p], tmp_path)
    assert res.returncode == 0
    assert [(d.severity, d.line) for d in diags] == [("warning", 2)]
    assert "constant selects" in diags[0].message


def test_truncation_from_32_bits_is_not_promoted(tmp_path):
    # `i - 1` with an int is 32-bit arithmetic; truncating it to 4 bits is the intended wrap-around.
    p = _write(tmp_path, "module TopModule (input logic [3:0] a, output logic [3:0] out);\n"
                         "  int i;\n  assign i = a;\n  assign out = i - 1;\nendmodule\n")
    diags = lint_slang([p], as_errors=["WidthTruncate"]).diagnostics
    trunc = [d for d in diags if d.code == "WidthTruncate"]
    assert trunc and all(d.severity == "warning" for d in trunc)


def test_format_collapses_follow_on_errors_and_adds_fix_hints(tmp_path):
    p = _write(tmp_path, GOOD.replace("out <= in;", "out <= {24{in[7]}, in};"))
    _, iv = run_iverilog([p], tmp_path)
    lint = lint_slang([p])
    text = format_diagnostics(iv + lint.diagnostics, p.read_text())
    # slang reports 4 parser errors on line 6; only its first is shown (plus iverilog's one).
    assert text.count("[ERROR] slang line 6") == 1
    assert text.count("[ERROR] iverilog line 6") == 1
    assert "Fix suggestions:\n- A replication inside a concatenation needs its own braces" in text


BLOCKING = AgentConfig().blocking_warnings


def test_analysis_pass_catches_multiple_drivers(tmp_path):
    p = _write(tmp_path, "module TopModule (input logic clk, input logic a, output logic q);\n"
                         "  logic n;\n"
                         "  always_comb n = a;\n"
                         "  always_ff @(posedge clk) begin n <= ~a; q <= n; end\n"
                         "endmodule\n")
    diags = lint_slang([p], as_errors=BLOCKING).diagnostics
    assert any(d.code == "MultipleAlwaysAssigns" and d.severity == "error" for d in diags)
    assert "only one always block" in format_diagnostics(diags, p.read_text())


def test_analysis_pass_catches_latch_but_not_default_assignment(tmp_path):
    latch = ("module TopModule (input logic [1:0] s, input logic a, output logic y);\n"
             "  always_comb begin\n    case (s)\n      2'd0: y = a;\n      2'd1: y = ~a;\n    endcase\n  end\nendmodule\n")
    diags = lint_slang([_write(tmp_path, latch)], as_errors=BLOCKING).diagnostics
    assert any(d.code == "InferredLatch" and d.severity == "error" for d in diags)
    clean = latch.replace("  always_comb begin\n", "  always_comb begin\n    y = 1'b0;\n")
    assert not [d for d in lint_slang([_write(tmp_path, clean)], as_errors=BLOCKING).diagnostics
                if d.severity == "error"]


def test_analysis_skipped_on_code_that_does_not_compile(tmp_path):
    p = _write(tmp_path, GOOD.replace("out <= in;", "out <= {24{in[7]}, in};"))
    diags = lint_slang([p], as_errors=BLOCKING).diagnostics
    assert not any(d.code in ("MultipleAlwaysAssigns", "InferredLatch") for d in diags)


def test_fix_hints_match_on_code_and_message():
    from atomrtl.eda import Diagnostic, fix_hints
    by_code = fix_hints([Diagnostic("slang", "lint", "error", "anything", 3, "CaseDup")])
    by_msg = fix_hints([Diagnostic("iverilog", "syntax", "error", "Syntax error ... of repeat concatenation.", 3)])
    warn_only = fix_hints([Diagnostic("slang", "lint", "warning", "x", 3, "CaseDup")])
    assert by_code and "case items" in by_code[0]
    assert by_msg and "own braces" in by_msg[0]
    assert warn_only == []  # hints are only for errors
