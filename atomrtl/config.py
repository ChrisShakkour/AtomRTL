"""Shared settings. Every agent and server launch reads these, so they always agree."""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "devstral-small-2"
    # Must be identical for every client: a different num_ctx forces Ollama to reload the model.
    num_ctx: int = 16384
    # host:port, overridable with OLLAMA_HOST (e.g. to use a server on another node).
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    # Concurrent requests the server handles; each slot needs its own num_ctx of GPU memory.
    num_parallel: int = 2
    # -1 keeps the model loaded until the server stops (default would unload after 5 min idle).
    keep_alive: str = "-1"
    log_file: Path = REPO_ROOT / "logs" / "ollama.log"
    # Seconds to wait for a newly started server to answer. It doesn't answer until GPU discovery
    # finishes, which has taken 13-30s on the 3090 machine.
    startup_timeout: float = 120

    @property
    def url(self) -> str:
        return self.host if "://" in self.host else f"http://{self.host}"


DEFAULT = OllamaConfig()


@dataclass(frozen=True)
class AnthropicConfig:
    """Claude over the Anthropic API (provider: anthropic). The key comes from ANTHROPIC_API_KEY, never from here."""

    model: str = "claude-sonnet-5-5"
    # Covers thinking + the reply. Claude Sonnet 5.5 always thinks (adaptive), so keep this well above the
    # ~1K tokens a reply needs.
    max_tokens: int = 16000
    # output_config.effort: low | medium | high | xhigh | max. Controls how much the model thinks.
    effort: str = "medium"
    # Concurrent problems in the benchmark runner (no GPU to share; bounded by the API rate limits).
    num_parallel: int = 8
    max_retries: int = 4
    # Seconds per request.
    timeout: float = 600
    # On a safety-classifier refusal, let the API rerun the request on a fallback model it picks
    # (fallbacks: "default", beta server-side-fallback-2026-07-01). Claude API only.
    refusal_fallback: bool = True


@dataclass(frozen=True)
class NodeLLMConfig:
    """Per-node LLM overrides; None means use the default (the active provider's model)."""

    model: str | None = None
    # Ignored by the anthropic provider (Claude Sonnet 5.5 rejects non-default sampling).
    temperature: float | None = None
    # Max tokens to generate (Ollama num_predict / Anthropic max_tokens); None = provider default.
    max_tokens: int | None = None


@dataclass(frozen=True)
class SimConfig:
    """Optional simulation against a reference testbench once the design passes the static checks.

    The testbench/reference files are per problem, so they come from the CLI (--sim-tb/--sim-ref), not from here.
    """

    enabled: bool = False
    # sim_fix attempts (LLM debugs a failing simulation), on top of max_attempts for static fixes.
    max_fixes: int = 4
    # Show the reference's expected output values in the waveform table (off: only inputs + the DUT's outputs).
    show_expected: bool = True
    # Waveform window around the first mismatch, in clock cycles (one row per clock edge), and a row cap.
    cycles_before: int = 6
    cycles_after: int = 2
    max_rows: int = 24
    # Give up early when the same mismatches (per output: count and first time) repeat this many times in a row.
    # One repeat only adds a "your fix changed nothing" note to the next sim_fix prompt.
    stuck_after: int = 3
    # Compile + simulate timeout, seconds.
    timeout: float = 60
    # Output cap for sim_fix replies (a diagnosis + the design is well under 1K tokens; at temperature 0 the
    # model occasionally loops until num_ctx). nodes.sim_fix.max_tokens overrides it.
    max_tokens: int | None = 2048


PROVIDERS = ("ollama", "anthropic")
# agent: the full graph (plan, generate, check, fix, optional simulation).
# oneshot: baseline. The spec goes to the LLM once and the reply is written to the file unchecked:
# no plan, no build, no lint, no fixes, no simulation.
MODES = ("agent", "oneshot")
# Prompt for mode oneshot. agent: the agent's own generate prompt (system rules, spec, parsed interface header,
# code-block instruction), without a plan. verilog_eval: verilog-eval's spec-to-rtl prompt, the spec verbatim.
ONESHOT_PROMPTS = ("agent", "verilog_eval")


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "agent"
    oneshot_prompt: str = "agent"
    # Which LLM backend every node uses: a local Ollama server, or Claude over the Anthropic API.
    provider: str = "ollama"
    ollama: OllamaConfig = DEFAULT
    anthropic: AnthropicConfig = AnthropicConfig()
    # Default sampling for every LLM node, overridable per node in `nodes`.
    temperature: float = 0.0
    nodes: dict[str, NodeLLMConfig] = field(default_factory=dict)
    # Total code attempts (first generate + fixes + restarts) before giving up.
    max_attempts: int = 6
    # Restart from a fresh generate when the same error signature repeats this many times in a row.
    stuck_after: int = 2
    # slang warnings treated as errors (must be fixed before the design passes): ones that almost always
    # mean a real bug and have a mechanical fix (see FIX_HINTS in eda.py). WidthTruncate: high bits
    # silently dropped (an explicit slice is fine); MultipleAlwaysAssigns / InferredLatch come from
    # slang's analysis pass; the rest are value-changing literals/conversions, dead case items,
    # typo'd names (implicit nets) and driven inputs.
    blocking_warnings: tuple[str, ...] = (
        "WidthTruncate", "MultipleAlwaysAssigns", "InferredLatch", "CaseDup",
        "VectorLiteralOverflow", "ConstantConversion", "ImplicitNet", "InputPortCoercion",
    )
    # Max diagnostics shown to the LLM per attempt (the transcript always keeps everything).
    max_diagnostics: int = 8
    # Per-command timeout for EDA tools, seconds.
    eda_timeout: float = 30
    simulation: SimConfig = SimConfig()
    runs_dir: Path = REPO_ROOT / "runs"

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; expected one of {PROVIDERS}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")
        if self.oneshot_prompt not in ONESHOT_PROMPTS:
            raise ValueError(f"unknown oneshot_prompt {self.oneshot_prompt!r}; expected one of {ONESHOT_PROMPTS}")
        if self.mode == "oneshot" and self.simulation.enabled:
            raise ValueError("mode oneshot never simulates; turn simulation off")

    @property
    def model(self) -> str:
        """The active provider's default model."""
        return self.anthropic.model if self.provider == "anthropic" else self.ollama.model

    @property
    def num_parallel(self) -> int:
        return self.anthropic.num_parallel if self.provider == "anthropic" else self.ollama.num_parallel

    def with_model(self, model: str) -> "AgentConfig":
        """Copy with the active provider's model replaced (the --model flag)."""
        if self.provider == "anthropic":
            return dataclasses.replace(self, anthropic=dataclasses.replace(self.anthropic, model=model))
        return dataclasses.replace(self, ollama=dataclasses.replace(self.ollama, model=model))

    def node_llm(self, node: str) -> NodeLLMConfig:
        """Resolved LLM settings for a node: per-node overrides on top of the defaults."""
        o = self.nodes.get(node, NodeLLMConfig())
        if self.provider == "anthropic":
            # No sampling settings, and no sim_fix cap: simulation.max_tokens guards against Ollama looping, but
            # here thinking counts against max_tokens, so 2048 would truncate replies.
            return NodeLLMConfig(model=o.model or self.model, temperature=None, max_tokens=o.max_tokens)
        return NodeLLMConfig(
            model=o.model or self.model,
            temperature=self.temperature if o.temperature is None else o.temperature,
            max_tokens=o.max_tokens if o.max_tokens or node != "sim_fix" else self.simulation.max_tokens,
        )


def load_config(path: str | Path | None = None, **overrides) -> AgentConfig:
    """AgentConfig from an optional YAML file (same keys as the dataclasses), then keyword overrides.

    YAML example:
        mode: agent                 # or: oneshot (single LLM call, no checks; see MODES)
        oneshot_prompt: agent       # or: verilog_eval (only used by mode oneshot)
        provider: ollama            # or: anthropic
        ollama: {model: devstral-small-2, num_ctx: 16384}
        anthropic: {model: claude-sonnet-5-5, effort: medium}
        temperature: 0.0
        max_attempts: 6
        nodes:
          plan: {temperature: 0.3}
        simulation: {enabled: true, max_fixes: 4}
    """
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = (data or {}) | {k: v for k, v in overrides.items() if v is not None}

    known = {f.name for f in dataclasses.fields(AgentConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    if "ollama" in data:
        data["ollama"] = OllamaConfig(**data["ollama"])
    if "anthropic" in data:
        data["anthropic"] = AnthropicConfig(**data["anthropic"])
    if "simulation" in data:
        data["simulation"] = SimConfig(**data["simulation"])
    if "nodes" in data:
        data["nodes"] = {name: NodeLLMConfig(**cfg) for name, cfg in data["nodes"].items()}
    if "blocking_warnings" in data:
        data["blocking_warnings"] = tuple(data["blocking_warnings"])
    if "runs_dir" in data:
        data["runs_dir"] = Path(data["runs_dir"])
    return AgentConfig(**data)


def config_to_dict(cfg: AgentConfig) -> dict:
    """Plain dict for writing the resolved config next to a run."""

    def convert(v):
        if dataclasses.is_dataclass(v):
            return {f.name: convert(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple):
            return [convert(x) for x in v]
        return v

    return convert(cfg)
