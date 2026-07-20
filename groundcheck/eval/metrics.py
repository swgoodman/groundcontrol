"""Quality metrics, computed in exactly one place.

Training-time eval and harness-time eval both call this module, so the numbers in a
training log and the numbers on the leaderboard can never diverge.

Convention throughout: `y_true` and `y_pred` are boolean arrays where True means
*supported*, and `p_supported` is the calibrated P(supported). The class of interest
is the rare, costly one, **not-supported** (hallucinated), so the headline
precision/recall/F1 are reported on that class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


@dataclass(slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(slots=True)
class Metrics:
    n: int
    balanced_acc: float
    precision_notsup: float
    recall_notsup: float
    f1_notsup: float
    pr_auc_notsup: float
    ece: float
    reliability: list[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reliability"] = [asdict(b) for b in self.reliability]
        return d


def expected_calibration_error(
    y_true: np.ndarray, p_supported: np.ndarray, n_bins: int = 10
) -> tuple[float, list[ReliabilityBin]]:
    """Standard ECE over the predicted class's confidence.

    Confidence is max(p, 1-p) and a prediction is correct when the thresholded
    decision matches the gold label. Empty bins contribute nothing.
    """
    y_true = np.asarray(y_true, dtype=bool)
    p = np.asarray(p_supported, dtype=float)
    if y_true.shape != p.shape:
        raise ValueError("y_true and p_supported must have the same shape")
    if y_true.size == 0:
        return 0.0, []

    confidence = np.maximum(p, 1.0 - p)
    correct = (p >= 0.5) == y_true

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[ReliabilityBin] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Include the left edge, and the right edge only in the final bin, so every
        # sample lands in exactly one bin.
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(ReliabilityBin(float(lo), float(hi), 0, 0.0, 0.0))
            continue
        acc = float(correct[in_bin].mean())
        conf = float(confidence[in_bin].mean())
        ece += (count / y_true.size) * abs(acc - conf)
        bins.append(ReliabilityBin(float(lo), float(hi), count, conf, acc))

    return float(ece), bins


def compute(y_true: np.ndarray, p_supported: np.ndarray, n_bins: int = 10) -> Metrics:
    """All quality metrics for one (scorer, dataset) run."""
    y_true = np.asarray(y_true, dtype=bool)
    p = np.asarray(p_supported, dtype=float)
    if y_true.shape != p.shape:
        raise ValueError("y_true and p_supported must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics over zero examples")

    y_pred = p >= 0.5

    # Flip to the not-supported class: it is the positive class for P/R/F1 and PR-AUC.
    notsup_true = ~y_true
    notsup_pred = ~y_pred
    p_notsup = 1.0 - p

    precision, recall, f1, _ = precision_recall_fscore_support(
        notsup_true, notsup_pred, average="binary", zero_division=0
    )

    # average_precision_score needs both classes present to be meaningful.
    pr_auc = (
        float(average_precision_score(notsup_true, p_notsup))
        if 0 < notsup_true.sum() < notsup_true.size
        else float("nan")
    )

    bacc = (
        float(balanced_accuracy_score(y_true, y_pred))
        if 0 < y_true.sum() < y_true.size
        else float("nan")
    )

    ece, reliability = expected_calibration_error(y_true, p, n_bins=n_bins)

    return Metrics(
        n=int(y_true.size),
        balanced_acc=bacc,
        precision_notsup=float(precision),
        recall_notsup=float(recall),
        f1_notsup=float(f1),
        pr_auc_notsup=pr_auc,
        ece=ece,
        reliability=reliability,
    )
