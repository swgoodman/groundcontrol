from __future__ import annotations

import warnings

import numpy as np
import pytest

from groundcheck.calibration import (
    apply_temperature,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)
from groundcheck.eval.metrics import expected_calibration_error


def _overconfident(n=400, seed=0):
    """Logits that are directionally right but far too sharp, as a trained net's are."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    margin = rng.normal(4.0, 1.5, size=n)
    logits = np.zeros((n, 2))
    correct = rng.random(n) < 0.75
    for i, (label, m, ok) in enumerate(zip(labels, margin, correct, strict=True)):
        winner = label if ok else 1 - label
        logits[i, winner] = m
    return logits, labels


def test_softmax_rows_are_distributions():
    probs = softmax(np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.allclose(probs[1], 1 / 3)


def test_softmax_is_numerically_stable_on_large_logits():
    probs = softmax(np.array([[1000.0, 999.0]]))
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(), 1.0)


def test_higher_temperature_softens_and_lower_sharpens():
    logits = np.array([[3.0, 0.0]])
    assert softmax(logits, 5.0).max() < softmax(logits, 1.0).max() < softmax(logits, 0.5).max()


def test_temperature_never_changes_a_prediction():
    # The property that makes post-hoc scaling safe: dividing every logit by the same
    # positive scalar cannot move the argmax, so accuracy is untouched.
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 3)) * 5
    baseline = softmax(logits, 1.0).argmax(axis=1)
    for t in (0.2, 0.7, 2.0, 9.0):
        assert (softmax(logits, t).argmax(axis=1) == baseline).all()


def test_fitting_softens_an_overconfident_model():
    logits, labels = _overconfident()
    assert fit_temperature(logits, labels) > 1.0


def test_fitting_sharpens_an_underconfident_model():
    logits, labels = _overconfident()
    assert fit_temperature(logits / 8.0, labels) < 1.0


def test_fitted_temperature_minimizes_validation_nll():
    logits, labels = _overconfident()
    best = fit_temperature(logits, labels)
    at_best = negative_log_likelihood(logits, labels, best)
    for other in (best * 0.5, best * 0.8, best * 1.25, best * 2.0):
        assert at_best <= negative_log_likelihood(logits, labels, other) + 1e-9


def test_calibration_improves_ece_without_touching_accuracy():
    # The deliverable in one assertion: same decisions, honest confidence.
    logits, labels = _overconfident()
    t = fit_temperature(logits, labels)

    before = softmax(logits, 1.0)[:, 1]
    after = apply_temperature(logits, t)[:, 1]
    truth = labels.astype(bool)

    ece_before, _ = expected_calibration_error(truth, before)
    ece_after, _ = expected_calibration_error(truth, after)

    assert ece_after < ece_before
    assert ((before >= 0.5) == (after >= 0.5)).all()


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        softmax(np.array([[1.0, 0.0]]), 0.0)


def test_fit_rejects_malformed_input():
    with pytest.raises(ValueError, match="2-D logits"):
        fit_temperature(np.array([1.0, 0.0]), np.array([0]))
    with pytest.raises(ValueError, match="same length"):
        fit_temperature(np.zeros((3, 2)), np.array([0]))
    with pytest.raises(ValueError, match="zero examples"):
        fit_temperature(np.zeros((0, 2)), np.array([], dtype=int))


def test_temperature_resting_on_a_bound_warns():
    # A clamped value is not a fit. Returning it silently would be the failure mode
    # this project exists to catch: a check that passes because it is blind.
    logits, labels = _overconfident()
    with pytest.warns(RuntimeWarning, match="clamp, not an optimum"):
        fit_temperature(logits, labels, lower=0.05, upper=1.5)


def test_no_warning_when_the_optimum_is_interior():
    logits, labels = _overconfident()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_temperature(logits, labels)


def test_collapse_marginalizes_rather_than_taking_the_larger_class():
    # logsumexp keeps both unsupported classes' mass. Taking the max would discard
    # one, understating P(not supported) when the model hedges between them.
    from groundcheck.calibration import collapse_to_binary_logits

    hedged = np.array([[0.0, 2.0, 2.0]])
    binary = collapse_to_binary_logits(hedged, supported_index=0)
    p_supported = softmax(binary)[0, 1]

    three_way_marginal = softmax(hedged)[0, 0]
    assert p_supported == pytest.approx(three_way_marginal, abs=1e-9)


def test_collapse_then_scale_preserves_decisions():
    from groundcheck.calibration import collapse_to_binary_logits

    rng = np.random.default_rng(3)
    logits = rng.normal(size=(300, 3)) * 4
    binary = collapse_to_binary_logits(logits)

    baseline = softmax(binary, 1.0)[:, 1] >= 0.5
    for t in (0.3, 3.0, 30.0):
        assert ((softmax(binary, t)[:, 1] >= 0.5) == baseline).all()


def test_scaling_the_three_vector_would_move_decisions():
    # Documents why the collapse happens first: this is the failure it prevents.
    # Three classes flatten toward 1/3, which is below the 0.5 threshold, so a
    # confident "supported" becomes "not supported" purely from softening.
    logits = np.array([[2.0, 0.0, 0.0]])
    assert softmax(logits, 1.0)[0, 0] > 0.5
    assert softmax(logits, 20.0)[0, 0] < 0.5


def test_collapse_rejects_wrong_shapes():
    from groundcheck.calibration import collapse_to_binary_logits

    with pytest.raises(ValueError, match=r"\(batch, 3\)"):
        collapse_to_binary_logits(np.zeros((2, 2)))
