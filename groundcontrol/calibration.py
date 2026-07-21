"""Temperature scaling.

A trained classifier's softmax scores are not probabilities. Cross-entropy rewards
confident correctness, so networks end up systematically overconfident: the zero-shot
baseline puts most of its predictions in the top confidence bin and is right less than
half the time there.

Temperature scaling divides the logits by a single scalar T fitted on validation data.
T > 1 softens, T < 1 sharpens. One parameter, fitted after training with the network
frozen, so it cannot change any decision: dividing every logit by the same positive
number leaves the argmax untouched. Accuracy is identical before and after, and only
the confidence moves. That is what makes it safe to ship as a post-hoc step.

The fitted T travels with the model, because a scorer that reports calibrated
probabilities in the notebook and raw softmax in production is worse than one that
never claimed calibration.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017).
"""

from __future__ import annotations

import warnings

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature)
    rows = np.arange(len(labels))
    return float(-np.log(np.clip(probs[rows, labels], 1e-12, None)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    lower: float = 0.01,
    upper: float = 100.0,
    tolerance: float = 1e-4,
    warn_on_bound: bool = True,
) -> float:
    """Find the T minimizing validation NLL, by golden-section search.

    NLL as a function of T is smooth and unimodal, so a derivative-free line search is
    enough and avoids depending on an optimizer. Searching a bounded interval also
    keeps a degenerate validation set from returning an absurd temperature.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {logits.shape}")
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have the same length")
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on zero examples")

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lower, upper
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc = negative_log_likelihood(logits, labels, c)
    fd = negative_log_likelihood(logits, labels, d)

    while abs(b - a) > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = negative_log_likelihood(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = negative_log_likelihood(logits, labels, d)

    fitted = float((a + b) / 2.0)

    # Temperature scales logits, so nearness to a bound is a ratio, not a distance.
    if warn_on_bound and (fitted / lower < 1.05 or upper / fitted < 1.05):
        warnings.warn(
            f"fitted temperature {fitted:.3f} sits on the edge of [{lower}, {upper}]: "
            f"this is a clamp, not an optimum. Widen the range, or read it as "
            f"evidence of miscalibration beyond what the range can express.",
            RuntimeWarning,
            stacklevel=2,
        )

    return fitted


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits, temperature)


def collapse_to_binary_logits(logits: np.ndarray, supported_index: int = 0) -> np.ndarray:
    """Marginalize 3-way logits to `[not_supported, supported]` via logsumexp.

    Do this *before* scaling, never after. Temperature preserves an argmax, but the
    reported decision thresholds P(supported) at 0.5, and in a three-class softmax that
    probability can cross 0.5 as temperature changes — so scaling the 3-vector silently
    moves decisions while claiming to be post-hoc.

    logsumexp over the two unsupported classes is the exact marginal, so no probability
    mass is discarded, unlike taking the larger of the two.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"expected (batch, 3) logits, got {logits.shape}")

    unsupported = np.delete(logits, supported_index, axis=1)
    shift = unsupported.max(axis=1)
    not_supported = shift + np.log(np.exp(unsupported - shift[:, None]).sum(axis=1))
    return np.stack([not_supported, logits[:, supported_index]], axis=1)
