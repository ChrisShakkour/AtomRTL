"""Graph flow with a scripted fake LLM (no Ollama needed)."""

import json
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import atomrtl.graph as graph
from atomrtl.cli import run_agent
from atomrtl.config import AgentConfig

SPEC = """I would like you to implement a module named TopModule with the following
interface. All input and output ports are one bit unless otherwise specified.

 - input  a
 - input  b
 - output out

The module should implement a NOR gate.
"""
GOOD = "```systemverilog\nmodule TopModule(input logic a, input logic b, output logic out);\n  assign out = ~(a | b);\nendmodule\n```"
SYNTAX_ERR = "```systemverilog\nmodule TopModule(input logic a, input logic b, output logic out);\n  assign out = ~(a | b)\nendmodule\n```"
WRONG_IFACE = "```systemverilog\nmodule TopModule(input logic x, input logic b, output logic out);\n  assign out = ~(x | b);\nendmodule\n```"
PLAN = "- out = NOR(a, b)"


@pytest.fixture
def scripted_llm(monkeypatch):
    """Replace the Ollama client: each LLM node call returns the next scripted response."""
    calls = []

    def install(*responses):
        fake = FakeListChatModel(responses=list(responses))

        def make_llm(ollama_cfg, node_cfg):
            calls.append(node_cfg)
            return fake

        monkeypatch.setattr(graph, "make_llm", make_llm)
        return calls

    return install


def _run(tmp_path, **cfg):
    return run_agent(SPEC, tmp_path, replace(AgentConfig(), **cfg))


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]


def test_pass_first_try(tmp_path, scripted_llm):
    scripted_llm(PLAN, GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 1
    assert (tmp_path / "TopModule.sv").read_text().startswith("module TopModule")
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "pass"


def test_fix_after_syntax_error(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, "- missing semicolon\n" + GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2
    assert [h["node"] for h in final["history"]] == ["generate", "fix"]
    assert final["history"][0]["errors"] > 0


def test_no_code_block_goes_to_fix(tmp_path, scripted_llm):
    scripted_llm(PLAN, "I will write a NOR gate.", GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2
    assert final["history"][0]["signature"] != ""


def test_interface_error_is_fixed(tmp_path, scripted_llm):
    scripted_llm(PLAN, WRONG_IFACE, GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2


def test_stuck_restarts_with_fresh_generate(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, SYNTAX_ERR, GOOD)
    final = _run(tmp_path, stuck_after=2)
    assert final["status"] == "pass"
    assert [h["node"] for h in final["history"]] == ["generate", "fix", "generate"]


def test_budget_exhausted_fails(tmp_path, scripted_llm):
    scripted_llm(PLAN, *[SYNTAX_ERR] * 10)
    final = _run(tmp_path, max_attempts=3, stuck_after=99)
    assert final["status"] == "fail" and final["attempts"] == 3
    assert "budget exhausted" in final["route_reason"]


def test_per_node_llm_settings(tmp_path, scripted_llm):
    from atomrtl.config import NodeLLMConfig

    calls = scripted_llm(PLAN, GOOD)
    _run(tmp_path, nodes={"plan": NodeLLMConfig(temperature=0.3)})
    assert [c.temperature for c in calls] == [0.3, 0.0]


def test_transcript_records_everything(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, GOOD)
    _run(tmp_path)
    events = _events(tmp_path)
    llm = [e for e in events if e["kind"] == "llm"]
    assert [e["node"] for e in llm] == ["plan", "generate", "fix"]
    assert llm[1]["sections"]["RESPONSE"] == SYNTAX_ERR  # raw response, before extraction
    assert any(k.startswith("SYSTEM") for k in llm[0]["sections"])
    tools = [e["tool"] for e in events if e["kind"] == "tool"]
    assert tools == ["iverilog", "slang", "interface"] * 2
    routes = [e["title"] for e in events if e["kind"] == "route"]
    assert routes[0].startswith("-> fix") and routes[1].startswith("-> finalize")
    log = (tmp_path / "transcript.log").read_text()
    assert "$ iverilog -g2012 -t null" in log and "--- RESPONSE ---" in log


def test_graph_shape():
    mermaid = graph.build_graph().get_graph().draw_mermaid()
    for edge in ["parse_spec --> plan", "plan --> generate", "generate --> write_code", "write_code --> check",
                 "check --> route", "fix --> write_code", "finalize --> __end__"]:
        assert edge in mermaid


def test_unchanged_fix_is_called_out(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, SYNTAX_ERR, GOOD)
    final = _run(tmp_path, stuck_after=99)
    assert final["status"] == "pass" and final["attempts"] == 3
    fix_prompts = [e["sections"]["HUMAN [1]"] for e in _events(tmp_path) if e["kind"] == "llm" and e["node"] == "fix"]
    assert "identical to the code that failed" not in fix_prompts[0]
    assert "identical to the code that failed" in fix_prompts[1]
    # Same compiler errors → same signature, so stuck detection still sees the repeat.
    assert final["history"][0]["signature"] == final["history"][1]["signature"]
