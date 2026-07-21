"""Scorer interface.

Anything that takes (context, claim) and returns a supported/not decision with a
confidence implements `Scorer`: the zero-shot NLI baseline, the fine-tune, HHEM,
MiniCheck, Granite Guardian, a hosted LLM judge. One interface is what makes the
leaderboard apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from groundcontrol.data.base import Example, Label3


@dataclass(slots=True)
class PassageScore:
    """Per-passage support, used by the injection canary and v2 attribution."""

    passage_id: int
    supported: bool
    score: float


@dataclass(slots=True)
class ClaimVerdict:
    """Per-claim support, used by the demo's sentence-level highlighting."""

    claim: str
    supported: bool
    score: float


@dataclass(slots=True)
class Verdict:
    supported: bool
    """Binary head. This is what the leaderboard scores."""

    score: float
    """Calibrated P(supported), in [0, 1]."""

    label3: Label3 | None = None
    """3-way internal prediction, when the scorer exposes one."""

    p_label3: dict[str, float] | None = None
    """Full 3-way distribution. The injection canary needs P(contradicted) specifically:
    a passage that merely fails to mention the claim is neutral, not contradicting, and
    `1 - P(supported)` cannot tell those apart."""

    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    passage_scores: list[PassageScore] = field(default_factory=list)
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(slots=True)
class EfficiencyProfile:
    """Static cost facts a scorer knows about itself.

    Measured latency/throughput are the runner's job, not the scorer's; this is only
    what the scorer can state without being run. `hosted` scorers report cost instead
    of footprint, which is why the footprint fields are optional.
    """

    hosted: bool = False
    params_m: float | None = None
    size_mb: float | None = None
    cost_per_1k_usd: float | None = None


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, context: str, claim: str) -> Verdict: ...

    def score_batch(self, items: list[Example]) -> list[Verdict]: ...

    def efficiency(self) -> EfficiencyProfile: ...
