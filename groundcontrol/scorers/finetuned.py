"""The fine-tuned groundedness scorer.

Same cross-encoder machinery as the zero-shot baseline, but its head was trained with
the internal label names rather than NLI's, so the vocabulary mapping differs. Kept as
a separate class because the distinction is real: one is a general entailment model
being borrowed for this task, the other was trained on it.

`training_corpora` matters here. A fine-tuned scorer is not in the known-checkpoint
table, and the contamination check reads exactly this to decide whether a result is
leaderboard-comparable.
"""

from __future__ import annotations

from groundcontrol.scorers.nli_zeroshot import NLIZeroShot


class Finetuned(NLIZeroShot):
    LABEL_MAP = {
        "supported": "supported",
        "contradicted": "contradicted",
        "neutral": "neutral",
    }
    SUPPORTED_KEY = "supported"

    def __init__(self, model_name: str, name: str = "groundcontrol", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.name = name
