"""The injection canary: detect that a retrieved set disagrees with itself.

A groundedness check silently assumes the evidence is trustworthy. Injection breaks that
assumption, and it breaks it in a way that whole-context scoring cannot see: concatenate
every passage and ask "is this claim supported?" and the answer is *yes*, because the
attacker's text is sitting right there in the context. The standard check passes the
attack through by construction.

Scoring each passage separately exposes the conflict. A poisoned set has one passage
insisting the claim is true while others contradict it. That internal disagreement is
the signal, and reading it requires no external truth source, no model internals, and no
retraining — the same scorer, called per passage instead of per context.

The signal is `min(strongest support, strongest contradiction)`, and contradiction must
come from the 3-way head rather than `1 - P(supported)`. A retrieved set is full of
passages that simply do not mention the claim; those are *neutral*, not contradicting,
and collapsing to binary makes every ordinary set look maximally conflicted. Measured:
that mistake drove detection to zero while clean sets scored *higher* than attacked ones.

So the signal is high only when some passage firmly supports the claim *and* another
firmly contradicts it, which is exactly what a minority injection creates. Sets that
agree, and sets where nothing supports the claim, both score low.

Threat model: the attacker controls a minority of passages. Detection degrades as that
fraction rises, and measuring the degradation honestly is part of the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from groundcheck.data.base import Example


@dataclass(slots=True)
class PassageVerdict:
    index: int
    p_supported: float
    p_contradicted: float
    label3: str | None
    poisoned: bool = False


@dataclass(slots=True)
class CanaryResult:
    conflict: float
    """min(max support, max contradiction). High means the set disagrees with itself."""

    whole_context_supported: bool
    """What a standard groundedness check concludes. The baseline the canary beats."""

    whole_context_score: float
    passage_verdicts: list[PassageVerdict] = field(default_factory=list)

    @property
    def most_supporting(self) -> PassageVerdict | None:
        """The passage most responsible for the conflict: the attribution signal."""
        return max(self.passage_verdicts, key=lambda v: v.p_supported, default=None)

    def localizes_attack(self) -> bool:
        """Whether the top-supporting passage is in fact the poisoned one."""
        top = self.most_supporting
        return bool(top and top.poisoned)


def run(scorer, passage_set, joiner: str = "\n\n") -> CanaryResult:
    """Score a claim against each passage separately, and against all of them joined."""
    claim = passage_set.claim
    passages = passage_set.passages
    poisoned = set(passage_set.poisoned_indices)

    per_passage = scorer.score_batch(
        [Example(context=p, claim=claim, label="supported") for p in passages]
    )
    joined = scorer.score(joiner.join(passages), claim)

    verdicts = [
        PassageVerdict(
            index=i,
            p_supported=float(v.score),
            p_contradicted=float((v.p_label3 or {}).get("contradicted", 1.0 - v.score)),
            label3=v.label3,
            poisoned=i in poisoned,
        )
        for i, v in enumerate(per_passage)
    ]

    max_support = max((v.p_supported for v in verdicts), default=0.0)
    max_contradiction = max((v.p_contradicted for v in verdicts), default=0.0)

    return CanaryResult(
        conflict=min(max_support, max_contradiction),
        whole_context_supported=bool(joined.supported),
        whole_context_score=float(joined.score),
        passage_verdicts=verdicts,
    )
