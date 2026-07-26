"""Compare detectors at a matched false-positive budget, with bootstrap intervals.

Two ways a detection comparison lies, both of which this module exists to close.

**Unmatched operating points.** A detector that flags 10% of clean traffic and one that
flags 3% are not comparable, and the difference flatters whichever one is allowed to
alarm more. A fitted threshold and an off-the-shelf 0.5 cutoff are exactly that mismatch.
Fixing the budget for both and refitting each threshold on clean traffic alone is the
only version of the comparison that means anything.

Ties make the budget approximate rather than exact. A whole-context score pinned near
1.0 for most clean sets cannot be cut at an arbitrary quantile: any threshold in that
mass flags either almost none of them or almost all. So the achieved rate is reported
next to the target and never assumed to equal it.

**A point estimate with no interval.** A detection rate on a few hundred sets carries
several points of sampling error, often more than the gap being described. The bootstrap
here resamples poisoned and clean sets together and *refits the threshold inside each
replicate*, so the interval carries threshold-estimation noise too; pretending the
operating point was known in advance would understate the spread.

The false-positive rate is then read on the clean sets that replicate held out, not on
the ones its threshold was cut from. Scoring a fitted quantile on its own fitting sample
returns the target back to you no matter how thin the evidence, which is a restatement
of the budget rather than a measurement of it.

**A ratio with an infinite tail, reported as though it were finite.** When the baseline
detects almost nothing, whole replicates come back with a zero denominator, and quietly
dropping those describes the experiment conditional on the baseline firing — an interval
that is both too narrow and biased toward the reference. They are kept in the ordering
here, so a ratio that is genuinely unbounded reports itself as one.

Detectors are resampled on shared indices, because they score the same sets. That makes
the interval on their ratio a paired one, which is both tighter and correct; resampling
each independently would describe an experiment nobody ran.

**A rate at one threshold, when the threshold sits on a cliff.** AUROC is reported next
to every detection rate because in this data they disagree, and the disagreement is the
point. Ordinary clean retrieval sets are not uniformly agreeable: about an eighth of them
contain a passage that genuinely contradicts the claim, so the clean conflict scores are
bimodal and a 10% budget forces the threshold up into that mass, where a two-point move
in the budget swings detection by thirty. AUROC asks whether attacked sets *rank* above
clean ones at all, which is the property the method either has or does not, and which no
choice of threshold can manufacture or destroy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(slots=True)
class Interval:
    """A percentile bootstrap interval. `level` is the nominal coverage, e.g. 0.95."""

    lo: float
    hi: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"[{self.lo:.3f}, {self.hi:.3f}]"


@dataclass(slots=True)
class DetectorScores:
    """One detector's raw scores on the same poisoned and clean sets as its rivals.

    `higher_is_attack` records which tail means "attack". The canary reads conflict, so
    high scores are suspicious. The whole-context check reads P(supported) on the joined
    context, so a *low* score is what flags. Encoding the direction here is what lets one
    threshold routine serve both without a caller getting the sign backwards.
    """

    name: str
    poisoned: np.ndarray
    clean: np.ndarray
    higher_is_attack: bool = True

    def __post_init__(self) -> None:
        self.poisoned = np.asarray(self.poisoned, dtype=float)
        self.clean = np.asarray(self.clean, dtype=float)
        if self.poisoned.ndim != 1 or self.clean.ndim != 1:
            raise ValueError(f"{self.name}: scores must be 1-D")
        if self.poisoned.size == 0 or self.clean.size == 0:
            raise ValueError(f"{self.name}: need at least one poisoned and one clean set")


@dataclass(slots=True)
class Detection:
    name: str
    threshold: float
    detection_rate: float
    detection_ci: Interval
    """Refits the threshold in every replicate: the spread of the whole experiment,
    re-run from scratch. This is the interval to quote."""

    detection_ci_given_threshold: Interval
    """Holds the threshold fixed and resamples only attacked sets. Narrower, and it
    answers a different question: how precise the detection rate is *once* the operating
    point is known. Reported alongside because the gap between the two is diagnostic —
    when it is large, the experiment is limited by how few clean sets pin the threshold,
    not by how few attacks were measured, and the cheap fix is more clean controls."""

    target_fpr: float
    achieved_fpr: float
    """What the fitted threshold actually costs on clean traffic. Equals `target_fpr`
    only when the clean scores are free enough of ties to cut at that quantile."""

    achieved_fpr_ci: Interval
    """How far that cost could travel on clean traffic the threshold was not fitted to.

    Measured out-of-bag: each replicate refits the threshold on its resample and then
    scores it on the clean sets that resample left out. Reading the rate off the same
    sets the quantile was cut from would pin it at the target by construction — the
    interval would then track tie and discretization noise only, could never exceed the
    budget however few clean controls there were, and would answer "did the quantile
    land where I asked" rather than the question anyone has, which is "what does this
    operating point cost tomorrow".

    Runs wider than a textbook binomial spread on `n_clean`, and not only by a little:
    each replicate fits on the ~63% of clean sets it drew and scores on the ~37% it did
    not, so the interval carries the noise of a sample that size rather than the full
    one. Read it as an upper bound on the drift. Erring wide is the right direction for
    a false-positive budget, and the cheap fix for a width that hurts is the same one
    the fixed-threshold gap points at: more clean controls."""

    auroc: float
    """Probability that a random attacked set outranks a random clean one. Threshold-free,
    so it survives an operating point this data cannot pin down. 0.5 is no signal."""

    auroc_ci: Interval
    n_poisoned: int
    n_clean: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Edge:
    """One detector against the reference, at the same false-positive budget."""

    name: str
    ratio: float
    ratio_ci: Interval
    """Unbounded above whenever the baseline detects nothing in more than a tail's worth
    of replicates, which in this data is the common case rather than the corner. A `hi`
    of `inf` is the honest reading of that, and the cue to quote `difference_ci`
    instead — see `ratio_unbounded` for how much of the bootstrap went that way."""

    ratio_unbounded: float
    """Fraction of replicates in which the baseline detected nothing, leaving the ratio
    infinite (or, if the reference also detected nothing, undefined). Reported because
    the ratio's point estimate looks equally authoritative at 0% and at 36%, and only
    this number tells the two apart. Above roughly a couple of percent, `ratio_ci` is
    unbounded and the ratio has stopped being a summary of anything."""

    difference: float
    """Percentage points. Reported alongside the ratio because the ratio is unstable
    when the baseline detects almost nothing: a near-zero denominator turns a modest
    gap into a huge multiple, and only the difference stays finite and readable."""

    difference_ci: Interval
    auroc_difference: float
    """Reference AUROC minus this one. The comparison that does not depend on where
    either threshold was put, and the one to lead with when the operating point moves."""

    auroc_difference_ci: Interval

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Comparison:
    target_fpr: float
    n_boot: int
    reference: str
    detections: dict[str, Detection] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_fpr": self.target_fpr,
            "n_boot": self.n_boot,
            "reference": self.reference,
            "detections": {k: v.to_dict() for k, v in self.detections.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
        }


def threshold_at_fpr(clean: np.ndarray, target_fpr: float, higher_is_attack: bool) -> float:
    """The cutoff that spends `target_fpr` of the clean traffic, and no attacked set.

    Fitted on clean scores alone. A threshold chosen with any view of attacked traffic
    is a threshold no deployment could have set in advance.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    q = 1.0 - target_fpr if higher_is_attack else target_fpr
    return float(np.quantile(clean, q))


def flags(scores: np.ndarray, threshold: float, higher_is_attack: bool) -> np.ndarray:
    """Strict comparison in both directions, so a score exactly on the threshold is not
    an alarm. Being strict on both sides is what keeps the achieved FPR at or under the
    budget rather than one tie-block above it."""
    return scores > threshold if higher_is_attack else scores < threshold


def auroc(poisoned: np.ndarray, clean: np.ndarray, higher_is_attack: bool = True) -> float:
    """P(a random attacked set outranks a random clean one), ties counted as half.

    Computed by rank rather than by calling out to sklearn, because the bootstrap needs
    it thousands of times over the same arrays and the rank form vectorizes across
    replicates. `higher_is_attack=False` flips the sign, which is the whole-context
    check's direction: it flags the *low* tail of P(supported).
    """
    p = np.asarray(poisoned, dtype=float)
    c = np.asarray(clean, dtype=float)
    if not higher_is_attack:
        p, c = -p, -c
    if p.size == 0 or c.size == 0:
        return float("nan")

    combined = np.concatenate([p, c])
    order = combined.argsort(kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1, dtype=float)

    # Average the ranks within each tie block, which is what makes a tie count as half a
    # win rather than a whole one. Conflict scores tie often — a saturated passage pins
    # them at the same value — so this is not a formality.
    values = combined[order]
    boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()

    rank_sum = ranks[: p.size].sum()
    return float((rank_sum - p.size * (p.size + 1) / 2.0) / (p.size * c.size))


def _order_statistic(ordered: np.ndarray, q: float) -> float:
    """numpy's linear-interpolation quantile, minus the arithmetic on infinities.

    `inf - inf` is nan, so interpolating between an infinite neighbour and anything else
    would report "undefined" where the honest bound is "unbounded". Interpolating across
    an infinite neighbour has one answer, which is that infinity. Identical to
    `np.quantile(..., method="linear")` whenever both neighbours are finite.
    """
    pos = q * (ordered.size - 1)
    lo_i, hi_i = int(np.floor(pos)), int(np.ceil(pos))
    lo, hi = ordered[lo_i], ordered[hi_i]
    if lo_i == hi_i or lo == hi:
        return float(lo)
    if np.isinf(hi):
        return float(hi)
    if np.isinf(lo):
        return float(lo)
    return float(lo + (hi - lo) * (pos - lo_i))


def _percentile_interval(samples: np.ndarray, level: float) -> Interval:
    """Percentile bounds over *every* replicate, infinities kept in the ordering.

    Dropping the infinite replicates instead would report a percentile of the
    distribution *conditional on the ratio being finite* — a tighter interval than the
    one the name promises, and tighter in the direction that flatters the reference. An
    infinite replicate is a real outcome, not a glitch: the denominator detected nothing
    that time, and the ratio really is unbounded. Kept in the sort, it pushes the bound
    to inf and says so.

    NaN is the different case. An undefined replicate has no place in an ordering at
    all, so it voids the interval rather than being quietly skipped.
    """
    s = np.asarray(samples, dtype=float)
    if s.size == 0 or np.isnan(s).any():
        return Interval(float("nan"), float("nan"), level)
    tail = (1.0 - level) / 2.0
    ordered = np.sort(s)
    return Interval(_order_statistic(ordered, tail), _order_statistic(ordered, 1.0 - tail), level)


def evaluate(
    detectors: Sequence[DetectorScores],
    target_fpr: float = 0.10,
    n_boot: int = 5000,
    seed: int = 0,
    level: float = 0.95,
    reference: str | None = None,
    auroc_boot: int = 1000,
) -> Comparison:
    """Score every detector at the same false-positive budget and bootstrap the gaps.

    `reference` is the detector the edges are measured against; it defaults to the first,
    which by convention here is the canary. Every replicate resamples one set of poisoned
    indices and one of clean indices and applies them to all detectors, because they are
    reading the same sets.

    `auroc_boot` caps how many of those replicates the AUROC interval uses. Ranking is
    the one quantity here that does not vectorize across replicates, and 1000 is already
    finer than the third decimal place anyone reads off it.
    """
    if not detectors:
        raise ValueError("need at least one detector")

    names = [d.name for d in detectors]
    if len(set(names)) != len(names):
        raise ValueError(f"detector names must be unique, got {names}")

    n_poisoned = {d.poisoned.size for d in detectors}
    n_clean = {d.clean.size for d in detectors}
    if len(n_poisoned) != 1 or len(n_clean) != 1:
        raise ValueError(
            "detectors must be scored on the same sets: got poisoned sizes "
            f"{sorted(n_poisoned)} and clean sizes {sorted(n_clean)}"
        )

    reference = reference or names[0]
    if reference not in names:
        raise ValueError(f"reference {reference!r} is not among {names}")

    n_p, n_c = n_poisoned.pop(), n_clean.pop()
    rng = np.random.default_rng(seed)
    idx_p = rng.integers(0, n_p, size=(n_boot, n_p))
    idx_c = rng.integers(0, n_c, size=(n_boot, n_c))

    # The clean sets each replicate did *not* draw — on average a bit over a third of
    # them. The refit threshold is scored here rather than on the resample it was cut
    # from, for the reason spelled out on `achieved_fpr_ci`.
    held_out = np.ones((n_boot, n_c), dtype=bool)
    np.put_along_axis(held_out, idx_c, False, axis=1)
    n_held_out = held_out.sum(axis=1)
    scorable = n_held_out > 0

    detections: dict[str, Detection] = {}
    boot_rates: dict[str, np.ndarray] = {}
    boot_auroc: dict[str, np.ndarray] = {}

    for d in detectors:
        threshold = threshold_at_fpr(d.clean, target_fpr, d.higher_is_attack)

        # Refit per replicate: the threshold is estimated from the same clean sets whose
        # false-positive rate it is scored on, and holding it fixed would report an
        # interval for a threshold the experiment did not have.
        clean_b, poisoned_b = d.clean[idx_c], d.poisoned[idx_p]
        q = 1.0 - target_fpr if d.higher_is_attack else target_fpr
        thresholds_b = np.quantile(clean_b, q, axis=1, keepdims=True)
        rates_b = flags(poisoned_b, thresholds_b, d.higher_is_attack).mean(axis=1)
        fixed_b = flags(poisoned_b, threshold, d.higher_is_attack).mean(axis=1)

        # Every clean set scored against every replicate's threshold, then masked down to
        # the ones that replicate held out.
        flagged_clean = flags(d.clean[None, :], thresholds_b, d.higher_is_attack)
        fpr_b = (flagged_clean & held_out)[scorable].sum(axis=1) / n_held_out[scorable]

        auroc_b = np.array(
            [
                auroc(poisoned_b[i], clean_b[i], d.higher_is_attack)
                for i in range(min(n_boot, auroc_boot))
            ]
        )

        boot_rates[d.name] = rates_b
        boot_auroc[d.name] = auroc_b
        detections[d.name] = Detection(
            name=d.name,
            threshold=threshold,
            detection_rate=float(flags(d.poisoned, threshold, d.higher_is_attack).mean()),
            detection_ci=_percentile_interval(rates_b, level),
            detection_ci_given_threshold=_percentile_interval(fixed_b, level),
            target_fpr=target_fpr,
            achieved_fpr=float(flags(d.clean, threshold, d.higher_is_attack).mean()),
            achieved_fpr_ci=_percentile_interval(fpr_b, level),
            auroc=auroc(d.poisoned, d.clean, d.higher_is_attack),
            auroc_ci=_percentile_interval(auroc_b, level),
            n_poisoned=n_p,
            n_clean=n_c,
        )

    ref_rate = detections[reference].detection_rate
    ref_boot = boot_rates[reference]
    edges: dict[str, Edge] = {}
    for name in names:
        if name == reference:
            continue
        other, other_boot = detections[name].detection_rate, boot_rates[name]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_b = np.divide(ref_boot, other_boot)
        edges[name] = Edge(
            name=name,
            # A zero denominator makes the ratio unbounded, but only if the numerator is
            # not also zero: two detectors that both found nothing are tied, not
            # infinitely far apart.
            ratio=ref_rate / other if other else (float("inf") if ref_rate else float("nan")),
            ratio_ci=_percentile_interval(ratio_b, level),
            ratio_unbounded=float(np.mean(other_boot == 0.0)),
            difference=ref_rate - other,
            difference_ci=_percentile_interval(ref_boot - other_boot, level),
            auroc_difference=detections[reference].auroc - detections[name].auroc,
            auroc_difference_ci=_percentile_interval(
                boot_auroc[reference] - boot_auroc[name], level
            ),
        )

    return Comparison(
        target_fpr=target_fpr,
        n_boot=n_boot,
        reference=reference,
        detections=detections,
        edges=edges,
    )


def proportion_ci(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval, for plain rates like localization.

    Wilson rather than the textbook normal approximation because these rates sit near
    1.0 (localization runs ~0.96), where the normal interval runs past 1 and reports
    coverage it cannot have. No bootstrap: for a single proportion the closed form is
    exact enough and does not need a seed.
    """
    if n <= 0:
        raise ValueError("cannot form an interval over zero trials")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")

    # 95% -> 1.96, without pulling scipy in for one number.
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}.get(level)
    if z is None:
        raise ValueError(f"unsupported level {level}; use 0.90, 0.95, or 0.99")

    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(float(max(0.0, centre - half)), float(min(1.0, centre + half)), level)
