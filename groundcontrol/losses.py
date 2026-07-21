"""Training objective for a 3-way head over mixed-precision labels.

The head predicts supported / contradicted / neutral, because the contradiction signal
is what the injection canary reads. But roughly half the training pool cannot supply
that distinction: HaluEval and AggreFact mark a claim unsupported without recording
whether it conflicts with the passage or merely invents detail (`label3_source="coarse"`).

Training those rows as "neutral" would teach the model that any hallucination is
baseless rather than contradictory, corrupting the exact signal the head exists for.
Dropping them discards most of the data. So they supervise the collapsed decision only:

    fine rows   -> cross-entropy over 3 classes
    coarse rows -> binary cross-entropy on P(supported) vs P(not supported),
                   where P(not supported) = P(contradicted) + P(neutral)

A coarse row therefore constrains how much probability mass sits on "supported" while
staying silent about how the remainder splits. Same head, same batch, two supervision
strengths.

Class weighting addresses the other asymmetry: not-supported is rare in RAGTruth
(35%) and it is the costly error, since a missed hallucination reaches the user while a
false alarm only costs a review.
"""

from __future__ import annotations

SUPPORTED, CONTRADICTED, NEUTRAL = 0, 1, 2
LABEL_ORDER = ("supported", "contradicted", "neutral")


def masked_three_way_loss(logits, labels, coarse_mask, class_weights=None, eps: float = 1e-7):
    """Cross-entropy on fine rows, collapsed binary cross-entropy on coarse rows.

    logits: (batch, 3). labels: (batch,) indices into LABEL_ORDER; coarse rows must
    already carry a not-supported label. coarse_mask: (batch,) bool, True where the
    3-way distinction is unreliable.

    Returns the mean over the batch, so the coarse fraction changes what is supervised
    but not the loss scale.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"expected (batch, 3) logits, got {tuple(logits.shape)}")
    if not (len(logits) == len(labels) == len(coarse_mask)):
        raise ValueError("logits, labels, and coarse_mask must have the same length")

    coarse_mask = coarse_mask.bool()
    log_probs = F.log_softmax(logits, dim=-1)

    per_example = torch.zeros(len(logits), dtype=log_probs.dtype, device=logits.device)

    fine = ~coarse_mask
    if fine.any():
        per_example[fine] = F.nll_loss(
            log_probs[fine],
            labels[fine],
            weight=class_weights.to(log_probs.dtype) if class_weights is not None else None,
            reduction="none",
        )

    if coarse_mask.any():
        probs = log_probs[coarse_mask].exp()
        p_supported = probs[:, SUPPORTED].clamp(eps, 1.0 - eps)
        is_supported = (labels[coarse_mask] == SUPPORTED).to(log_probs.dtype)

        binary = -(
            is_supported * p_supported.log() + (1.0 - is_supported) * (1.0 - p_supported).log()
        )
        if class_weights is not None:
            # Collapse the 3-way weights the same way the probabilities collapse: the
            # not-supported weight is the mean of its two constituent classes.
            weights = class_weights.to(log_probs.dtype)
            not_supported_weight = weights[[CONTRADICTED, NEUTRAL]].mean()
            binary = binary * torch.where(
                is_supported.bool(), weights[SUPPORTED], not_supported_weight
            )
        per_example[coarse_mask] = binary

    return per_example.mean()


def class_weights_from_labels(labels, num_classes: int = 3):
    """Inverse-frequency weights, normalized to mean 1 so the loss scale is unchanged.

    Absent classes get weight 1 rather than infinity.
    """
    import torch

    counts = torch.bincount(labels, minlength=num_classes).to(torch.float)
    present = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights / weights.mean()
