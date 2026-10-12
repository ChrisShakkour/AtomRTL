"""Provider selection: Ollama (default) vs Claude over the Anthropic API. No network: clients are only built."""

from dataclasses import replace

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from atomrtl.config import AgentConfig, NodeLLMConfig, load_config
from atomrtl.llm import llm_backend, make_llm

CLAUDE = AgentConfig(provider="anthropic")


def test_default_provider_is_ollama():
    cfg = AgentConfig()
    assert cfg.provider == "ollama"
    llm = make_llm(cfg, cfg.node_llm("generate"))
    assert isinstance(llm, ChatOllama)
    assert llm.model == cfg.ollama.model and llm.temperature == 0.0


def test_anthropic_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = make_llm(CLAUDE, CLAUDE.node_llm("generate"))
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-sonnet-5-5"
    assert llm.max_tokens == 16000
    assert llm.temperature is None  # Claude Sonnet 5.5 rejects non-default sampling
    assert llm.reasoning_effort == "medium"
    assert llm.betas == ["server-side-fallback-2026-07-01"]
    assert llm.model_kwargs == {"fallbacks": "default"}


def test_anthropic_ignores_temperatures_and_ollama_sim_cap():
    cfg = replace(CLAUDE, temperature=0.3, nodes={"plan": NodeLLMConfig(temperature=0.7)})
    assert cfg.node_llm("plan").temperature is None
    assert cfg.node_llm("sim_fix").max_tokens is None  # falls back to anthropic.max_tokens
    cfg = replace(cfg, nodes={"sim_fix": NodeLLMConfig(max_tokens=4000)})
    assert cfg.node_llm("sim_fix").max_tokens == 4000


def test_anthropic_without_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = replace(CLAUDE, anthropic=replace(CLAUDE.anthropic, refusal_fallback=False))
    llm = make_llm(cfg, cfg.node_llm("fix"))
    assert llm.betas is None and llm.model_kwargs == {}


def test_with_model_targets_active_provider():
    assert AgentConfig().with_model("qwen3").ollama.model == "qwen3"
    cfg = CLAUDE.with_model("claude-opus-5-5")
    assert cfg.anthropic.model == "claude-opus-5-5" and cfg.model == "claude-opus-5-5"
    assert cfg.ollama.model == AgentConfig().ollama.model


def test_load_config_provider(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("provider: anthropic\nanthropic: {model: claude-sonnet-5-5, effort: high}\n")
    cfg = load_config(path)
    assert cfg.provider == "anthropic" and cfg.anthropic.effort == "high"
    assert load_config(path, provider="ollama").provider == "ollama"  # CLI flag wins
    with pytest.raises(ValueError, match="unknown provider"):
        AgentConfig(provider="openai")


def test_anthropic_backend_needs_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        with llm_backend(CLAUDE):
            pass
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with llm_backend(CLAUDE) as keeper:
        keeper.ensure()  # no server to start
