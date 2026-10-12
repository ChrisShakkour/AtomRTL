"""One shared Ollama server for all agents, and a factory for chat-model clients (Ollama or Anthropic API)."""

import os
import shutil
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from atomrtl.config import DEFAULT, AgentConfig, AnthropicConfig, NodeLLMConfig, OllamaConfig


def is_running(cfg: OllamaConfig = DEFAULT) -> bool:
    try:
        urllib.request.urlopen(f"{cfg.url}/api/version", timeout=2)
        return True
    except OSError:
        return False


def _start_server(cfg: OllamaConfig) -> subprocess.Popen | None:
    """Start `ollama serve` and wait until it answers. Returns None if another process won the race."""
    binary = shutil.which("ollama")
    if binary is None:
        raise RuntimeError("'ollama' not found on PATH; see README for install steps")

    env = os.environ | {
        "OLLAMA_HOST": cfg.url.split("://", 1)[1],
        "OLLAMA_NUM_PARALLEL": str(cfg.num_parallel),
        "OLLAMA_CONTEXT_LENGTH": str(cfg.num_ctx),
        "OLLAMA_KEEP_ALIVE": cfg.keep_alive,
    }
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        proc = subprocess.Popen(
            [binary, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

    deadline = time.monotonic() + cfg.startup_timeout
    while not is_running(cfg):
        if proc.poll() is not None:
            if is_running(cfg):  # another process started a server at the same moment
                break
            raise RuntimeError(f"ollama serve exited with code {proc.returncode}; see {log_path}")
        if time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"ollama serve did not start within {cfg.startup_timeout}s; see {log_path}")
        time.sleep(0.2)
    return proc if proc.poll() is None else None


class ServerKeeper:
    """Keeps an Ollama server available for as long as it's needed.

    ensure() reuses a running server or (re)starts one if it's down, e.g. because whoever started the
    server we were sharing has exited. close() stops only the servers this keeper started.
    Thread-safe, so parallel workers can call ensure() before each task.
    """

    def __init__(self, cfg: OllamaConfig = DEFAULT):
        self.cfg = cfg
        self._started: list[subprocess.Popen] = []
        self._lock = threading.Lock()

    def ensure(self) -> None:
        with self._lock:
            if not is_running(self.cfg):
                if proc := _start_server(self.cfg):
                    self._started.append(proc)

    def close(self) -> None:
        for proc in self._started:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


@contextmanager
def ollama_server(cfg: OllamaConfig = DEFAULT):
    """Use the running Ollama server, or start one for the duration of the block.

    A server this block starts is stopped on exit; an already-running one is left alone
    (and keeps whatever settings it was started with). Yields the ServerKeeper, whose ensure()
    restarts the server if it goes away mid-block.
    """
    keeper = ServerKeeper(cfg)
    keeper.ensure()
    try:
        yield keeper
    finally:
        keeper.close()


class _NoServer:
    """Stands in for ServerKeeper when the model is behind an API: nothing to start or keep alive."""

    def ensure(self) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def llm_backend(cfg: AgentConfig):
    """What the agents need running for `cfg.provider`: the shared Ollama server, or nothing (Anthropic API).

    Yields an object with ensure(), like ollama_server().
    """
    if cfg.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("provider is anthropic but ANTHROPIC_API_KEY is not set")
        yield _NoServer()
    else:
        with ollama_server(cfg.ollama) as keeper:
            yield keeper


def make_llm(cfg: AgentConfig, node: NodeLLMConfig | None = None, **kwargs) -> BaseChatModel:
    """Chat model for the configured provider.

    `node` applies per-node overrides (model, temperature, max_tokens) — see AgentConfig.node_llm().
    """
    if cfg.provider == "anthropic":
        return _make_anthropic(cfg.anthropic, node, **kwargs)
    return _make_ollama(cfg.ollama, node, **kwargs)


def _make_ollama(cfg: OllamaConfig = DEFAULT, node: NodeLLMConfig | None = None, **kwargs) -> ChatOllama:
    """Client for the shared server; always uses the shared num_ctx."""
    if node is not None:
        kwargs.setdefault("temperature", node.temperature)
        kwargs.setdefault("num_predict", node.max_tokens)
    model = node.model if node is not None and node.model else cfg.model
    return ChatOllama(model=model, base_url=cfg.url, num_ctx=cfg.num_ctx, **kwargs)


def _make_anthropic(cfg: AnthropicConfig, node: NodeLLMConfig | None = None, **kwargs):
    """Claude over the API. No temperature: Claude Sonnet 5.5 rejects non-default sampling and always thinks
    (adaptive); `effort` controls how much."""
    # Imported here so an Ollama-only environment doesn't need the package.
    from langchain_anthropic import ChatAnthropic

    model = node.model if node is not None and node.model else cfg.model
    max_tokens = node.max_tokens if node is not None and node.max_tokens else cfg.max_tokens
    if cfg.refusal_fallback:
        kwargs.setdefault("betas", ["server-side-fallback-2026-07-01"])
        kwargs.setdefault("model_kwargs", {"fallbacks": "default"})
    # An organization-level key (not scoped to a workspace) must name the workspace on every request.
    if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        kwargs.setdefault("default_headers", {"anthropic-workspace-id": workspace})
    return ChatAnthropic(model=model, max_tokens=max_tokens, reasoning_effort=cfg.effort,
                         max_retries=cfg.max_retries, timeout=cfg.timeout, **kwargs)
