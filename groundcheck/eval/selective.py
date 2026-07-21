"""Risk-coverage analysis: what the scorer is worth as a gate.

Accuracy answers "how often is it right." A deployment asks something narrower: if I
auto-accept the answers this scorer is most confident are grounded, and send the rest
to review, how many ungrounded answers still reach a user, and how much review do I pay
for that?

    coverage — the fraction of traffic auto-accepted
    risk     — of what was accepted, the fraction that was actually not supported

Sweeping the threshold traces the trade-off. The reference to read it against is a
scorer whose confidence carries no information: its risk is flat at the base rate no
matter where the threshold sits, because every slice looks like a random sample. A
useful gate bends below that line, and the gap is the whole value proposition.

This is the operational form of the calibration argument. A scorer can post respectable
balanced accuracy and still be worthless here, and nothing in the accuracy column would
show it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(slots=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    n_accepted: int
    n_missed: int
    """Accepted answers that were not supported: the ones that reach a user."""


@dataclass(slots=True)
class RiskCoverage:
    points: list[RiskCoveragePoint] = field(default_factory=list)
    base_rate: float = 0.0
    """Fraction not supported overall. A no-information scorer sits flat here."""

    aurc: float = 0.0
    """Area under the risk-coverage curve. Lower is better; base_rate is no-information."""

    optimal_aurc: float = 0.0
    """AURC of a perfect ranking on these labels. Not zero: past the grounded
    fraction, even flawless ordering must start admitting ungrounded answers."""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        return d

    def risk_at_coverage(self, coverage: float) -> float:
        """Missed-hallucination rate when auto-accepting this fraction of traffic."""
        eligible = [p for p in self.points if p.coverage <= coverage and p.n_accepted]
        return eligible[-1].risk if eligible else 0.0

    def coverage_at_risk(self, max_risk: float) -> float:
        """Most traffic auto-acceptable while holding missed hallucinations under a bound.

        The operational question: 'how much can I stop reviewing and still keep the
        error rate under 1%?'
        """
        eligible = [p for p in self.points if p.risk <= max_risk and p.n_accepted]
        return max((p.coverage for p in eligible), default=0.0)

    @property
    def lift_over_random(self) -> float:
        """Where this scorer sits between no-information and the best possible ranking.

        1.0 is perfect ordering, 0.0 is no better than random, negative is worse than
        random. Normalized against `optimal_aurc` rather than zero, because the
        unreachable ceiling depends on the base rate and would otherwise make the
        number incomparable across datasets.
        """
        span = self.base_rate - self.optimal_aurc
        if span <= 0:
            return 0.0
        return (self.base_rate - self.aurc) / span


def risk_coverage(y_true: np.ndarray, p_supported: np.ndarray) -> RiskCoverage:
    """Trace risk against coverage by sweeping the auto-accept threshold.

    Answers are accepted in descending order of P(supported), so coverage grows by
    admitting progressively less confident answers. Every distinct score becomes a
    threshold, which avoids the arbitrary binning that would smooth over the cliff
    edges a deployment actually cares about.
    """
    y_true = np.asarray(y_true, dtype=bool)
    p = np.asarray(p_supported, dtype=float)
    if y_true.shape != p.shape:
        raise ValueError("y_true and p_supported must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot compute risk-coverage over zero examples")

    order = np.argsort(-p, kind="stable")
    return _trace(y_true, p, order, optimal_aurc=_optimal_aurc(y_true), collapse_ties=True)


def _optimal_aurc(y_true: np.ndarray) -> float:
    """AURC achievable by a flawless ranking, used to normalize `lift_over_random`.

    Ties are not collapsed here: the ideal ordering is a construction used as a
    reference point, not a threshold any gate would implement.
    """
    perfect = np.argsort(~y_true, kind="stable")
    return _trace(y_true, y_true.astype(float), perfect, optimal_aurc=0.0, collapse_ties=False).aurc


def _trace(
    y_true: np.ndarray,
    p: np.ndarray,
    order: np.ndarray,
    optimal_aurc: float,
    collapse_ties: bool,
) -> RiskCoverage:
    not_supported = (~y_true)[order]
    scores = p[order]

    cumulative_missed = np.cumsum(not_supported)
    n = y_true.size

    points: list[RiskCoveragePoint] = []
    for i in range(n):
        # Only cut between distinct scores: splitting a tie would report a threshold
        # no gate could implement.
        if collapse_ties and i + 1 < n and scores[i] == scores[i + 1]:
            continue
        accepted = i + 1
        missed = int(cumulative_missed[i])
        points.append(
            RiskCoveragePoint(
                threshold=float(scores[i]),
                coverage=accepted / n,
                risk=missed / accepted,
                n_accepted=accepted,
                n_missed=missed,
            )
        )

    base_rate = float((~y_true).mean())
    # Mean risk across coverage levels, weighted by the span each point covers.
    if points:
        coverages = np.array([p_.coverage for p_ in points])
        risks = np.array([p_.risk for p_ in points])
        widths = np.diff(np.concatenate([[0.0], coverages]))
        aurc = float((risks * widths).sum())
    else:
        aurc = base_rate

    return RiskCoverage(points=points, base_rate=base_rate, aurc=aurc, optimal_aurc=optimal_aurc)


def summarize(rc: RiskCoverage, coverages=(0.5, 0.8, 0.9, 1.0)) -> dict:
    """The handful of numbers a deployment decision actually turns on."""
    return {
        "base_rate": rc.base_rate,
        "aurc": rc.aurc,
        "optimal_aurc": rc.optimal_aurc,
        "lift_over_random": rc.lift_over_random,
        "risk_at_coverage": {c: rc.risk_at_coverage(c) for c in coverages},
        "coverage_at_risk": {r: rc.coverage_at_risk(r) for r in (0.01, 0.05, 0.10)},
    }
