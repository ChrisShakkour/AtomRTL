"""Graph state and run context for the RTL-creation agent."""

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from atomrtl.config import AgentConfig
from atomrtl.eda import Diagnostic
from atomrtl.spec import Port
from atomrtl.transcript import Transcript


class Feedback(TypedDict):
    """One piece of feedback for the next fix: from static checks now, from LLM reviewers later."""

    source: str  # e.g. "check"
    kind: str  # e.g. "diagnostics"
    text: str


class AttemptRecord(TypedDict):
    attempt: int
    node: str  # generate | fix
    signature: str  # error signature ("" when clean)
    errors: int
    warnings: int


def _add_tokens(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}


class RTLState(TypedDict, total=False):
    # Input
    spec: str
    top: str | None
    # parse_spec
    module_name: str | None
    ports: tuple[Port, ...] | None
    header: str | None
    # LLM nodes
    plan: str
    response: str  # last raw generate/fix response
    response_node: str  # which node produced `response`: generate | fix
    restart_hint: str | None  # set by route when a fresh generate replaces a stuck fix loop
    # write_code / check
    code: str | None
    code_path: str | None
    code_unchanged: bool  # the new code is identical to the previous attempt's
    diagnostics: list[Diagnostic]
    feedback: list[Feedback]  # for the current attempt; the fix node renders all of it
    attempts: int
    history: Annotated[list[AttemptRecord], operator.add]
    # route / finalize
    route: str
    route_reason: str
    status: str  # pass | fail
    tokens: Annotated[dict, _add_tokens]


@dataclass
class RunContext:
    """Run-scoped, non-state inputs available to every node via `runtime.context`."""

    config: AgentConfig
    workdir: Path
    transcript: Transcript
