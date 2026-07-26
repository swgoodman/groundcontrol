from __future__ import annotations

import numpy as np
import pytest

from groundcontrol.eval import detection


def scores(poisoned, clean, name="d", higher=True):
    return detection.DetectorScores(
        name=name, poisoned=np.array(poisoned), clean=np.array(clean), higher_is_attack=higher
    )


# --- thresholds ---------------------------------------------------------------------


def test_threshold_spends_the_budget_on_clean_traffic():
    clean = np.linspace(0.0, 1.0, 101)
    t = detection.threshold_at_fpr(clean, 0.10, higher_is_attack=True)
    assert detection.flags(clean, t, True).mean() == pytest.approx(0.10, abs=0.02)


def test_threshold_flips_with_the_direction():
    # A detector reading P(supported) flags the *low* tail, so the same budget lands at
    # the opposite end of the distribution.
    clean = np.linspace(0.0, 1.0, 101)
    high = detection.threshold_at_fpr(clean, 0.10, higher_is_attack=True)
    low = detection.threshold_at_fpr(clean, 0.10, higher_is_attack=False)
    assert low < high
    assert detection.flags(clean, low, False).mean() == pytest.approx(0.10, abs=0.02)


def test_a_score_exactly_on_the_threshold_is_not_an_alarm():
    assert not detection.flags(np.array([0.5]), 0.5, True)[0]
    assert not detection.flags(np.array([0.5]), 0.5, False)[0]


def test_ties_are_reported_rather_than_smoothed_over():
    # Clean scores pinned at one value cannot be cut at an arbitrary quantile. The
    # achieved rate has to show that, not silently claim the target was met.
    d = scores(poisoned=[0.2] * 50, clean=[0.99] * 100, higher=False)
    result = detection.evaluate([d], target_fpr=0.10, n_boot=200)
    assert result.detections["d"].achieved_fpr == 0.0
    assert result.detections["d"].target_fpr == 0.10


# --- matched comparison -------------------------------------------------------------


def test_both_detectors_are_held_to_the_same_budget():
    rng = np.random.default_rng(0)
    a = scores(rng.normal(0.8, 0.1, 200), rng.normal(0.2, 0.1, 200), name="canary", higher=True)
    b = scores(rng.normal(0.7, 0.1, 200), rng.normal(0.9, 0.1, 200), name="whole", higher=False)

    result = detection.evaluate([a, b], target_fpr=0.10, n_boot=200)
    for det in result.detections.values():
        assert det.achieved_fpr == pytest.approx(0.10, abs=0.03)


def test_detectors_must_be_scored_on_the_same_sets():
    a = scores([0.9] * 10, [0.1] * 10, name="a")
    b = scores([0.9] * 9, [0.1] * 10, name="b")
    with pytest.raises(ValueError, match="same sets"):
        detection.evaluate([a, b], n_boot=10)


def test_duplicate_detector_names_are_rejected():
    a = scores([0.9] * 10, [0.1] * 10, name="a")
    with pytest.raises(ValueError, match="unique"):
        detection.evaluate([a, a], n_boot=10)


def test_edge_reports_a_finite_difference_when_the_ratio_blows_up():
    # The regime the README lives in: the baseline detects almost nothing, so the ratio
    # is enormous and uninformative while the difference stays readable.
    canary = scores([0.9] * 100, [0.1] * 100, name="canary", higher=True)
    whole = scores([0.99] * 100, [0.99] * 100, name="whole", higher=False)
    result = detection.evaluate([canary, whole], target_fpr=0.10, n_boot=200)

    edge = result.edges["whole"]
    assert edge.difference == pytest.approx(1.0)
    assert np.isinf(edge.ratio)


def test_reference_defaults_to_the_first_detector():
    a = scores([0.9] * 20, [0.1] * 20, name="canary")
    b = scores([0.5] * 20, [0.1] * 20, name="whole")
    result = detection.evaluate([a, b], n_boot=100)
    assert result.reference == "canary"
    assert set(result.edges) == {"whole"}


# --- intervals ----------------------------------------------------------------------


def test_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    d = scores(rng.normal(0.8, 0.15, 300), rng.normal(0.2, 0.15, 300), name="canary")
    det = detection.evaluate([d], n_boot=2000).detections["canary"]
    assert det.detection_ci.lo <= det.detection_rate <= det.detection_ci.hi


def test_smaller_samples_give_wider_intervals():
    rng = np.random.default_rng(2)
    big = scores(rng.normal(0.8, 0.2, 1000), rng.normal(0.2, 0.2, 1000), name="d")
    rng = np.random.default_rng(2)
    small = scores(rng.normal(0.8, 0.2, 40), rng.normal(0.2, 0.2, 40), name="d")

    wide = detection.evaluate([small], n_boot=2000).detections["d"].detection_ci
    tight = detection.evaluate([big], n_boot=2000).detections["d"].detection_ci
    assert (wide.hi - wide.lo) > (tight.hi - tight.lo)


def test_the_interval_carries_threshold_uncertainty():
    # Refitting the threshold inside each replicate has to widen the interval versus
    # holding the point-estimate threshold fixed, since the threshold is itself measured.
    rng = np.random.default_rng(3)
    poisoned, clean = rng.normal(0.7, 0.25, 60), rng.normal(0.3, 0.25, 60)
    d = scores(poisoned, clean, name="d")

    refit = detection.evaluate([d], n_boot=4000).detections["d"].detection_ci

    fixed_threshold = detection.threshold_at_fpr(clean, 0.10, True)
    idx = np.random.default_rng(0).integers(0, 60, size=(4000, 60))
    fixed = detection.flags(poisoned[idx], fixed_threshold, True).mean(axis=1)
    fixed_width = np.quantile(fixed, 0.975) - np.quantile(fixed, 0.025)

    assert (refit.hi - refit.lo) > fixed_width


def test_evaluation_is_reproducible_from_the_seed():
    d = scores(np.linspace(0.4, 1.0, 50), np.linspace(0.0, 0.6, 50), name="d")
    a = detection.evaluate([d], n_boot=500, seed=7).detections["d"].detection_ci
    b = detection.evaluate([d], n_boot=500, seed=7).detections["d"].detection_ci
    assert (a.lo, a.hi) == (b.lo, b.hi)


# --- auroc --------------------------------------------------------------------------


def test_auroc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    for _ in range(5):
        p, c = rng.normal(1, 1, 120), rng.normal(0, 1, 200)
        y = np.r_[np.ones(120), np.zeros(200)]
        assert detection.auroc(p, c) == pytest.approx(roc_auc_score(y, np.r_[p, c]))

    # Conflict scores tie often, so a tie must count as half a win, not a whole one.
    tied_p, tied_c = np.array([1.0, 1.0, 2.0]), np.array([1.0, 0.0])
    expected = roc_auc_score([1, 1, 1, 0, 0], np.r_[tied_p, tied_c])
    assert detection.auroc(tied_p, tied_c) == pytest.approx(expected)


def test_auroc_respects_the_attack_direction():
    # A detector whose low tail means attack scores well only once the sign is flipped.
    poisoned, clean = np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9])
    assert detection.auroc(poisoned, clean, higher_is_attack=False) == 1.0
    assert detection.auroc(poisoned, clean, higher_is_attack=True) == 0.0


def test_auroc_survives_a_threshold_that_moves():
    # The reason it is reported at all: clean scores here are bimodal, so the fitted
    # threshold lands on a cliff and the detection rate swings with it. Ranking does not.
    rng = np.random.default_rng(4)
    clean = np.r_[rng.normal(0.02, 0.01, 700), rng.normal(0.9, 0.03, 100)]
    poisoned = rng.normal(0.85, 0.15, 800)
    d = scores(poisoned, clean, name="canary")

    tight = detection.evaluate([d], target_fpr=0.10, n_boot=1000).detections["canary"]
    loose = detection.evaluate([d], target_fpr=0.20, n_boot=1000).detections["canary"]

    assert abs(tight.detection_rate - loose.detection_rate) > 0.1
    assert tight.auroc == pytest.approx(loose.auroc)


def test_auroc_edge_is_reported_against_the_reference():
    rng = np.random.default_rng(5)
    strong = scores(rng.normal(1.0, 1, 200), rng.normal(0, 1, 200), name="canary")
    weak = scores(rng.normal(0.2, 1, 200), rng.normal(0, 1, 200), name="whole")

    result = detection.evaluate([strong, weak], n_boot=500)
    edge = result.edges["whole"]
    assert edge.auroc_difference > 0
    assert edge.auroc_difference_ci.lo <= edge.auroc_difference <= edge.auroc_difference_ci.hi


def test_auroc_of_indistinguishable_detectors_is_a_coin_flip():
    rng = np.random.default_rng(6)
    d = scores(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000), name="d")
    det = detection.evaluate([d], n_boot=200).detections["d"]
    assert det.auroc == pytest.approx(0.5, abs=0.05)


# --- proportions --------------------------------------------------------------------


def test_wilson_interval_stays_inside_the_unit_range_near_one():
    # The case that motivates Wilson: localization runs ~0.96, where the normal
    # approximation reports an upper bound above 1.
    ci = detection.proportion_ci(117, 120)
    assert 0.0 <= ci.lo <= ci.hi <= 1.0
    assert ci.hi < 1.0


def test_wilson_interval_handles_a_perfect_rate():
    ci = detection.proportion_ci(120, 120)
    assert ci.hi == 1.0
    assert ci.lo < 1.0


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        detection.proportion_ci(5, 0)
    with pytest.raises(ValueError):
        detection.proportion_ci(11, 10)
