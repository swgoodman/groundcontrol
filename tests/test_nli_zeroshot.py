"""Zero-shot scorer logic, without loading a model.

The part worth testing hermetically is the label mapping. NLI checkpoints disagree on
index order, and reading it wrong inverts every prediction while still producing
numbers that look reasonable. `test_nli_zeroshot_live.py` checks a real checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from groundcheck.scorers.nli_zeroshot import NLIZeroShot


def _scorer_with_label_order(order: list[str], threshold: float = 0.5) -> NLIZeroShot:
    scorer = NLIZeroShot(threshold=threshold)
    scorer._label_index = {name: i for i, name in enumerate(order)}
    return scorer


ENTAILMENT_FIRST = ["entailment", "neutral", "contradiction"]
CONTRADICTION_FIRST = ["contradiction", "entailment", "neutral"]


def test_probabilities_are_read_by_label_name_not_position():
    # The same distribution under two checkpoint conventions must yield the same
    # verdict. MoritzLaurer/DeBERTa-v3-base-mnli is entailment-first;
    # cross-encoder/nli-deberta-v3-base is contradiction-first.
    entail_first = _scorer_with_label_order(ENTAILMENT_FIRST)
    contra_first = _scorer_with_label_order(CONTRADICTION_FIRST)

    a = entail_first._verdict_from_probs(np.array([0.9, 0.05, 0.05]))
    b = contra_first._verdict_from_probs(np.array([0.05, 0.9, 0.05]))

    assert a.supported is b.supported is True
    assert a.score == pytest.approx(b.score) == pytest.approx(0.9)
    assert a.label3 == b.label3 == "supported"


@pytest.mark.parametrize(
    ("probs", "expected3"),
    [
        ([0.8, 0.1, 0.1], "supported"),
        ([0.1, 0.8, 0.1], "neutral"),
        ([0.1, 0.1, 0.8], "contradicted"),
    ],
)
def test_three_way_label_follows_the_argmax(probs, expected3):
    verdict = _scorer_with_label_order(ENTAILMENT_FIRST)._verdict_from_probs(np.array(probs))
    assert verdict.label3 == expected3


def test_binary_head_thresholds_entailment_rather_than_taking_the_argmax():
    # Entailment can win a 3-way argmax while sitting below the threshold. The
    # leaderboard scores the thresholded probability, so the two views can disagree
    # and that is intended, not a bug.
    verdict = _scorer_with_label_order(ENTAILMENT_FIRST)._verdict_from_probs(
        np.array([0.45, 0.30, 0.25])
    )
    assert verdict.label3 == "supported"
    assert verdict.supported is False
    assert verdict.score == pytest.approx(0.45)


def test_threshold_is_configurable():
    scorer = _scorer_with_label_order(ENTAILMENT_FIRST, threshold=0.4)
    assert scorer._verdict_from_probs(np.array([0.45, 0.30, 0.25])).supported is True


def test_default_checkpoint_avoids_training_on_an_evaluation_dataset():
    # A "zero-shot" baseline trained on FEVER is not zero-shot on FEVER.
    assert NLIZeroShot().training_corpora == ("mnli",)
    variant = NLIZeroShot(model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    assert "fever" in variant.training_corpora


def test_empty_batch_returns_no_verdicts_without_loading_a_model():
    assert NLIZeroShot().score_batch([]) == []
