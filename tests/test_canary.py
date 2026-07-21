from __future__ import annotations

import pytest

from groundcheck import canary
from groundcheck.data.injection import PassageSet, build_set, make_payload
from groundcheck.scorers.base import Verdict


class ScriptedScorer:
    """Returns a preset P(supported) per passage, keyed by a marker in the text."""

    name = "scripted"

    def __init__(self, mapping: dict[str, float], joined: float = 0.9):
        self.mapping = mapping
        self.joined = joined

    def _score_for(self, context: str) -> float:
        for marker, value in self.mapping.items():
            if marker in context:
                return value
        return 0.5

    def score(self, context: str, claim: str) -> Verdict:
        # A joined context contains every marker, so it is scored separately: this is
        # the whole-context baseline the canary is compared against.
        if sum(1 for m in self.mapping if m in context) > 1:
            return Verdict(supported=self.joined >= 0.5, score=self.joined)
        p = self._score_for(context)
        return Verdict(supported=p >= 0.5, score=p)

    def score_batch(self, items):
        return [self.score(i.context, i.claim) for i in items]


# --- the set builder ---------------------------------------------------------------


def test_payload_asserts_the_claim():
    assert "Paris is in Spain" in make_payload("Paris is in Spain.")


def test_built_set_marks_where_the_poison_landed():
    s = build_set("C", ["refutes C"], ["d1", "d2", "d3", "d4"], 1, 5, key="k")

    assert len(s.passages) == 5
    assert s.n_poisoned == 1
    assert "C" in s.passages[s.poisoned_indices[0]]


def test_poison_is_not_always_in_the_same_position():
    # Otherwise a detector could win by learning the index rather than the signal.
    positions = {
        build_set("C", ["r"], ["d"] * 4, 1, 5, key=f"k{i}").poisoned_indices[0] for i in range(20)
    }
    assert len(positions) > 1


def test_clean_sets_have_no_poisoned_passages():
    s = build_set("C", ["r"], ["d"] * 4, 0, 5, key="k")
    assert s.poisoned_indices == [] and s.poisoned is False


def test_majority_poisoning_is_rejected_by_default():
    # Outside the stated threat model. The earlier guard only caught k >= n, so it
    # silently permitted 3-of-5 and 4-of-5, which are majorities.
    for k in (3, 4, 5):
        with pytest.raises(ValueError, match="majority poisoning"):
            build_set("C", ["r"], ["d"] * 6, k, 5, key="k")


def test_majority_poisoning_can_be_probed_deliberately():
    # Reporting where detection breaks is part of the result, but it has to be opt-in
    # so a majority-poisoned number can never be produced by accident.
    s = build_set("C", ["r"], ["d"] * 6, 3, 5, key="k", allow_majority=True)
    assert s.n_poisoned == 3


def test_displacing_the_evidence_removes_the_trusted_contradiction():
    # The condition the canary actually depends on: not the poison fraction, but
    # whether any trusted contradicting passage survives retrieval.
    kept = build_set("C", ["REFUTES C"], ["d"] * 6, 1, 5, key="k")
    displaced = build_set("C", ["REFUTES C"], ["d"] * 6, 1, 5, key="k", keep_refuting=False)

    assert any("REFUTES C" in p for p in kept.passages)
    assert not any("REFUTES C" in p for p in displaced.passages)


def test_insufficient_material_returns_none():
    assert build_set("C", [], [], 1, 5, key="k") is None


# --- the canary --------------------------------------------------------------------


def _poisoned_set():
    return PassageSet(
        claim="the claim",
        passages=["POISON asserts it", "REFUTE denies it", "NEUTRAL unrelated"],
        poisoned_indices=[0],
    )


def test_conflict_fires_when_the_set_disagrees_with_itself():
    scorer = ScriptedScorer({"POISON": 0.97, "REFUTE": 0.02, "NEUTRAL": 0.4})
    result = canary.run(scorer, _poisoned_set())

    assert result.conflict > 0.9


def test_a_set_that_agrees_does_not_fire():
    scorer = ScriptedScorer({"POISON": 0.95, "REFUTE": 0.93, "NEUTRAL": 0.9}, joined=0.95)
    everything_agrees = PassageSet(
        claim="c", passages=["POISON a", "REFUTE b", "NEUTRAL c"], poisoned_indices=[]
    )
    assert canary.run(scorer, everything_agrees).conflict < 0.1


def test_a_set_supporting_nothing_does_not_fire():
    # No support anywhere is an unanswerable query, not an attack.
    scorer = ScriptedScorer({"POISON": 0.05, "REFUTE": 0.03, "NEUTRAL": 0.02}, joined=0.04)
    assert canary.run(scorer, _poisoned_set()).conflict < 0.1


def test_whole_context_scoring_misses_the_attack_that_the_canary_catches():
    # The result the experiment exists to show. Concatenating passages puts the
    # attacker's text inside the context, so the standard check calls it supported.
    scorer = ScriptedScorer({"POISON": 0.97, "REFUTE": 0.02, "NEUTRAL": 0.4}, joined=0.93)
    result = canary.run(scorer, _poisoned_set())

    assert result.whole_context_supported is True
    assert result.conflict > 0.9


def test_canary_localizes_the_poisoned_passage():
    scorer = ScriptedScorer({"POISON": 0.97, "REFUTE": 0.02, "NEUTRAL": 0.4})
    result = canary.run(scorer, _poisoned_set())

    assert result.most_supporting.index == 0
    assert result.localizes_attack() is True


def test_verdicts_are_returned_per_passage_with_poison_flags():
    scorer = ScriptedScorer({"POISON": 0.9, "REFUTE": 0.1, "NEUTRAL": 0.5})
    verdicts = canary.run(scorer, _poisoned_set()).passage_verdicts

    assert [v.index for v in verdicts] == [0, 1, 2]
    assert [v.poisoned for v in verdicts] == [True, False, False]
