import pytest

from groundcontrol.data.base import Example
from groundcontrol.registry import (
    available_scorers,
    get_scorer,
    register_scorer,
)
from groundcontrol.scorers.base import EfficiencyProfile, Scorer, Verdict


class ConstantScorer:
    """Minimal conforming scorer, used to pin the interface without pulling torch."""

    name = "constant"

    def __init__(self, score: float = 0.5):
        self._score = score

    def score(self, context: str, claim: str) -> Verdict:
        return Verdict(supported=self._score >= 0.5, score=self._score)

    def score_batch(self, items: list[Example]) -> list[Verdict]:
        return [self.score(i.context, i.claim) for i in items]

    def efficiency(self) -> EfficiencyProfile:
        return EfficiencyProfile(hosted=False, params_m=0.0, size_mb=0.0)


def test_a_plain_class_satisfies_the_protocol():
    # Protocols, not inheritance: a scorer is anything with the right shape.
    assert isinstance(ConstantScorer(), Scorer)


def test_verdict_rejects_out_of_range_scores():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Verdict(supported=True, score=bad)


def test_score_batch_matches_score():
    s = ConstantScorer(0.8)
    items = [Example(context="c", claim="q", label="supported") for _ in range(3)]
    batch = s.score_batch(items)
    assert len(batch) == 3
    assert all(v.score == s.score("c", "q").score for v in batch)


def test_hosted_scorers_report_cost_instead_of_footprint():
    prof = EfficiencyProfile(hosted=True, cost_per_1k_usd=0.42)
    assert prof.params_m is None and prof.size_mb is None
    assert prof.cost_per_1k_usd == 0.42


def test_registry_roundtrip_and_duplicate_guard():
    register_scorer("test-constant", ConstantScorer)
    assert "test-constant" in available_scorers()
    assert isinstance(get_scorer("test-constant", score=0.9), ConstantScorer)

    with pytest.raises(ValueError, match="already registered"):
        register_scorer("test-constant", ConstantScorer)

    with pytest.raises(KeyError, match="unknown scorer"):
        get_scorer("nope")


def test_registry_resolves_import_paths_lazily():
    register_scorer("test-by-path", "tests.test_scorer_interface:ConstantScorer")
    assert isinstance(get_scorer("test-by-path"), ConstantScorer)
