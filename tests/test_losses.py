"""Loss behaviour, checked on hand-built tensors rather than by training."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from groundcheck.losses import (  # noqa: E402
    CONTRADICTED,
    NEUTRAL,
    SUPPORTED,
    class_weights_from_labels,
    masked_three_way_loss,
)


def _loss(logits, labels, coarse, weights=None):
    return masked_three_way_loss(
        torch.tensor(logits, dtype=torch.float),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(coarse, dtype=torch.bool),
        class_weights=weights,
    ).item()


CONFIDENT_SUPPORTED = [8.0, 0.0, 0.0]
CONFIDENT_CONTRADICTED = [0.0, 8.0, 0.0]
CONFIDENT_NEUTRAL = [0.0, 0.0, 8.0]


def test_fine_rows_are_supervised_on_the_full_three_way_distinction():
    right = _loss([CONFIDENT_CONTRADICTED], [CONTRADICTED], [False])
    wrong = _loss([CONFIDENT_NEUTRAL], [CONTRADICTED], [False])
    assert right < 0.01 < wrong


def test_coarse_rows_do_not_distinguish_contradicted_from_neutral():
    # The whole point. Both predictions place all mass on not-supported, and a coarse
    # row has no opinion about which. Supervising it either way would invent a label.
    as_contradicted = _loss([CONFIDENT_CONTRADICTED], [NEUTRAL], [True])
    as_neutral = _loss([CONFIDENT_NEUTRAL], [NEUTRAL], [True])
    assert as_contradicted == pytest.approx(as_neutral, abs=1e-6)


def test_the_same_rows_are_distinguished_when_marked_fine():
    as_contradicted = _loss([CONFIDENT_CONTRADICTED], [NEUTRAL], [False])
    as_neutral = _loss([CONFIDENT_NEUTRAL], [NEUTRAL], [False])
    assert as_contradicted > as_neutral + 1.0


def test_coarse_rows_still_supervise_the_collapsed_decision():
    # Silent about which kind of unsupported, not silent about whether it is supported.
    correct = _loss([CONFIDENT_NEUTRAL], [NEUTRAL], [True])
    incorrect = _loss([CONFIDENT_SUPPORTED], [NEUTRAL], [True])
    assert correct < 0.01 < incorrect


def test_coarse_supported_rows_are_supervised_normally():
    correct = _loss([CONFIDENT_SUPPORTED], [SUPPORTED], [True])
    incorrect = _loss([CONFIDENT_CONTRADICTED], [SUPPORTED], [True])
    assert correct < 0.01 < incorrect


def test_probability_mass_splits_across_both_unsupported_classes():
    # A coarse row is satisfied by any split of the remainder, so a model hedging
    # between contradicted and neutral pays no penalty over committing to one.
    split = _loss([[0.0, 4.0, 4.0]], [NEUTRAL], [True])
    committed = _loss([[0.0, 8.0, 0.0]], [NEUTRAL], [True])
    assert split == pytest.approx(committed, abs=0.01)


def test_mixed_batches_combine_both_supervision_modes():
    logits = [CONFIDENT_SUPPORTED, CONFIDENT_CONTRADICTED]
    clean = _loss(logits, [SUPPORTED, CONTRADICTED], [False, True])
    assert clean < 0.01

    broken = _loss(logits, [SUPPORTED, SUPPORTED], [False, True])
    assert broken > 1.0


def test_loss_is_a_mean_so_the_coarse_fraction_does_not_change_scale():
    one = _loss([CONFIDENT_SUPPORTED], [CONTRADICTED], [True])
    four = _loss([CONFIDENT_SUPPORTED] * 4, [CONTRADICTED] * 4, [True] * 4)
    assert one == pytest.approx(four, abs=1e-6)


def test_gradients_flow_through_both_paths():
    logits = torch.tensor(
        [CONFIDENT_SUPPORTED, CONFIDENT_SUPPORTED], dtype=torch.float, requires_grad=True
    )
    masked_three_way_loss(
        logits,
        torch.tensor([CONTRADICTED, NEUTRAL]),
        torch.tensor([False, True]),
    ).backward()
    assert (logits.grad[0] != 0).any()
    assert (logits.grad[1] != 0).any()


def test_class_weights_raise_the_cost_of_the_rare_class():
    weights = torch.tensor([0.5, 2.0, 2.0])
    unweighted = _loss([CONFIDENT_SUPPORTED], [CONTRADICTED], [False])
    weighted = _loss([CONFIDENT_SUPPORTED], [CONTRADICTED], [False], weights=weights)
    assert weighted > unweighted


def test_class_weights_apply_to_coarse_rows_too():
    weights = torch.tensor([0.5, 2.0, 2.0])
    unweighted = _loss([CONFIDENT_SUPPORTED], [NEUTRAL], [True])
    weighted = _loss([CONFIDENT_SUPPORTED], [NEUTRAL], [True], weights=weights)
    assert weighted > unweighted


def test_inverse_frequency_weights_favour_the_rare_class():
    labels = torch.tensor([SUPPORTED] * 8 + [CONTRADICTED, NEUTRAL])
    weights = class_weights_from_labels(labels)
    assert weights[CONTRADICTED] > weights[SUPPORTED]
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)


def test_absent_classes_do_not_produce_infinite_weights():
    weights = class_weights_from_labels(torch.tensor([SUPPORTED, SUPPORTED]))
    assert torch.isfinite(weights).all()


def test_malformed_input_is_rejected():
    with pytest.raises(ValueError, match=r"\(batch, 3\)"):
        masked_three_way_loss(torch.zeros(2, 2), torch.zeros(2).long(), torch.zeros(2).bool())
    with pytest.raises(ValueError, match="same length"):
        masked_three_way_loss(torch.zeros(2, 3), torch.zeros(3).long(), torch.zeros(2).bool())
