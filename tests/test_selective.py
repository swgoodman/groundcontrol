from __future__ import annotations

import numpy as np
import pytest

from groundcheck.eval.selective import risk_coverage, summarize


def test_full_coverage_risk_equals_the_base_rate():
    # Accept everything and you inherit exactly the population's error rate.
    y = np.array([True, True, False, False, False])
    rc = risk_coverage(y, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))

    assert rc.base_rate == pytest.approx(0.6)
    assert rc.points[-1].coverage == 1.0
    assert rc.points[-1].risk == pytest.approx(0.6)


def test_a_perfect_ranking_admits_every_grounded_answer_before_any_bad_one():
    y = np.array([True, True, True, False, False])
    rc = risk_coverage(y, np.array([0.99, 0.98, 0.97, 0.02, 0.01]))

    assert rc.risk_at_coverage(0.6) == 0.0
    assert rc.coverage_at_risk(0.0) == pytest.approx(0.6)
    # Normalized against the achievable optimum, not zero: past 60% coverage even a
    # flawless ranking has to start admitting the ungrounded answers.
    assert rc.lift_over_random == pytest.approx(1.0)
    assert rc.optimal_aurc > 0.0


def test_uninformative_confidence_is_flat_at_the_base_rate():
    # The failure the whole module exists to expose: a gate that is exactly equivalent
    # to sampling at random. Every coverage level carries the population error rate.
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.7
    scores = rng.random(2000)  # unrelated to the labels

    rc = risk_coverage(y, scores)

    assert rc.aurc == pytest.approx(rc.base_rate, abs=0.03)
    assert abs(rc.lift_over_random) < 0.1


def test_an_informative_scorer_bends_below_the_random_line():
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.7
    # Confidence correlated with correctness, plus noise.
    scores = np.where(y, rng.normal(0.8, 0.15, 2000), rng.normal(0.3, 0.15, 2000)).clip(0, 1)

    rc = risk_coverage(y, scores)

    assert rc.aurc < rc.base_rate
    assert rc.lift_over_random > 0.5
    assert rc.risk_at_coverage(0.5) < rc.risk_at_coverage(1.0)


def test_inverted_confidence_is_worse_than_random():
    y = np.array([True, True, False, False])
    rc = risk_coverage(y, np.array([0.1, 0.2, 0.9, 0.8]))

    assert rc.lift_over_random < 0


def test_coverage_at_risk_answers_the_deployment_question():
    # 8 grounded, 2 not. The two bad ones score lowest, so 80% of traffic can be
    # auto-accepted with zero misses.
    y = np.array([True] * 8 + [False] * 2)
    scores = np.array([0.95] * 8 + [0.10] * 2)

    rc = risk_coverage(y, scores)

    assert rc.coverage_at_risk(0.0) == pytest.approx(0.8)
    assert rc.risk_at_coverage(0.8) == 0.0


def test_thresholds_are_never_placed_inside_a_tie():
    # A threshold that splits equal scores cannot be implemented: the gate has no way
    # to prefer one of two identically-scored answers.
    y = np.array([True, False, True, False])
    rc = risk_coverage(y, np.array([0.5, 0.5, 0.5, 0.5]))

    assert len(rc.points) == 1
    assert rc.points[0].coverage == 1.0


def test_missed_counts_track_ungrounded_answers_that_were_accepted():
    y = np.array([True, False, True])
    rc = risk_coverage(y, np.array([0.9, 0.8, 0.7]))

    assert [p.n_missed for p in rc.points] == [0, 1, 1]
    assert [p.n_accepted for p in rc.points] == [1, 2, 3]


def test_summary_exposes_the_operational_numbers():
    y = np.array([True] * 8 + [False] * 2)
    out = summarize(risk_coverage(y, np.array([0.9] * 8 + [0.1] * 2)))

    assert out["base_rate"] == pytest.approx(0.2)
    assert out["risk_at_coverage"][0.8] == 0.0
    assert out["coverage_at_risk"][0.01] == pytest.approx(0.8)


def test_input_guards():
    with pytest.raises(ValueError, match="same shape"):
        risk_coverage(np.array([True]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="zero examples"):
        risk_coverage(np.array([], dtype=bool), np.array([]))
