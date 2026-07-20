import numpy as np
import pytest

from groundcheck.eval import metrics


def test_perfect_confident_predictions_are_calibrated():
    y = np.array([True, True, False, False])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    m = metrics.compute(y, p)
    assert m.balanced_acc == pytest.approx(1.0)
    assert m.f1_notsup == pytest.approx(1.0)
    assert m.ece == pytest.approx(0.0)


def test_confidently_wrong_is_maximally_miscalibrated():
    y = np.array([True, True, False, False])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    m = metrics.compute(y, p)
    assert m.balanced_acc == pytest.approx(0.0)
    assert m.ece == pytest.approx(1.0)


def test_metrics_target_the_not_supported_class():
    # 3 not-supported, 1 supported. The scorer catches 2 of 3 hallucinations and
    # raises no false alarms, so P=1.0 and R=2/3 on the class of interest.
    y = np.array([False, False, False, True])
    p = np.array([0.1, 0.2, 0.9, 0.9])
    m = metrics.compute(y, p)
    assert m.precision_notsup == pytest.approx(1.0)
    assert m.recall_notsup == pytest.approx(2 / 3)
    assert m.f1_notsup == pytest.approx(0.8)


def test_single_class_yields_nan_not_a_misleading_number():
    # A smoke slice can easily be all-supported. Better to surface NaN than to report
    # a balanced accuracy that silently means nothing.
    y = np.array([True, True, True])
    p = np.array([0.9, 0.8, 0.7])
    m = metrics.compute(y, p)
    assert np.isnan(m.balanced_acc)
    assert np.isnan(m.pr_auc_notsup)


def test_every_sample_lands_in_exactly_one_reliability_bin():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = rng.random(500) < p
    ece, bins = metrics.expected_calibration_error(y, p, n_bins=10)
    assert sum(b.count for b in bins) == 500
    assert 0.0 <= ece <= 1.0


def test_confidence_is_symmetric_around_the_decision_boundary():
    # p=0.02 predicting not-supported is just as confident as p=0.98 predicting
    # supported. ECE must treat them alike, or the rare class looks miscalibrated.
    ece_low, _ = metrics.expected_calibration_error(np.array([False]), np.array([0.02]))
    ece_high, _ = metrics.expected_calibration_error(np.array([True]), np.array([0.98]))
    assert ece_low == pytest.approx(ece_high)


def test_shape_and_emptiness_guards():
    with pytest.raises(ValueError, match="same shape"):
        metrics.compute(np.array([True]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="zero examples"):
        metrics.compute(np.array([], dtype=bool), np.array([]))
