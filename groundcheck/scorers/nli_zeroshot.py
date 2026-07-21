"""Zero-shot NLI scorer: baseline A.

Treats groundedness as entailment. The evidence is the premise, the claim under test is
the hypothesis, and an off-the-shelf NLI model decides between entailment, neutral, and
contradiction. Nothing is trained here; this is the floor every later scorer has to beat.

Two details that are easy to get wrong and expensive to get wrong silently:

**Label order is read from the checkpoint, never assumed.** NLI models disagree on it.
`MoritzLaurer/DeBERTa-v3-base-mnli` is entailment-first, `cross-encoder/nli-deberta-v3-base`
is contradiction-first. A hardcoded index would invert every prediction for one of them
and still produce plausible-looking numbers.

**Truncation drops evidence, never the claim.** Contexts routinely exceed the encoder's
512-token window. Truncating the hypothesis would mean scoring a claim the model never
finished reading, so `only_first` trims the premise and leaves the claim intact.

The default checkpoint is MNLI-only rather than the `-fever-anli` variant, which is
trained on FEVER. FEVER is one of the evaluation datasets here, and a "zero-shot"
baseline that has seen the test set is not one. The training corpora are recorded in
`training_corpora` so a report can state what a given checkpoint has already seen.
"""

from __future__ import annotations

from groundcheck.data.base import Example, Label3
from groundcheck.device import resolve_compute_device
from groundcheck.scorers.base import EfficiencyProfile, Verdict

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli"

KNOWN_TRAINING_CORPORA: dict[str, tuple[str, ...]] = {
    "MoritzLaurer/DeBERTa-v3-base-mnli": ("mnli",),
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": ("mnli", "fever", "anli"),
    "cross-encoder/nli-deberta-v3-base": ("snli", "mnli"),
}

_NLI_TO_LABEL3: dict[str, Label3] = {
    "entailment": "supported",
    "neutral": "neutral",
    "contradiction": "contradicted",
}


class NLIZeroShot:
    # The checkpoint's label vocabulary, mapped onto the internal scheme. Overridden by
    # scorers whose head was trained with our own label names.
    LABEL_MAP = _NLI_TO_LABEL3
    SUPPORTED_KEY = "entailment"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        threshold: float = 0.5,
        temperature: float = 1.0,
        training_corpora: tuple[str, ...] | None = None,
    ):
        self.model_name = model_name
        # A fine-tuned checkpoint is not in the known-corpora table, and its own
        # training mix is exactly what the contamination check needs to see.
        self._training_corpora = training_corpora
        self.temperature = temperature
        suffix = "" if temperature == 1.0 else f"+T{temperature:.2f}"
        self.name = f"nli-zeroshot:{model_name.split('/')[-1]}{suffix}"
        self.device = resolve_compute_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._label_index: dict[str, int] = {}

    @property
    def training_corpora(self) -> tuple[str, ...]:
        if self._training_corpora is not None:
            return self._training_corpora
        return KNOWN_TRAINING_CORPORA.get(self.model_name, ())

    def _load(self) -> None:
        """Deferred so constructing a scorer stays cheap; weights load on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        model.to(self.device)
        self._model = model
        self._torch = torch

        self._label_index = {
            str(label).lower(): int(idx) for idx, label in model.config.id2label.items()
        }
        missing = set(self.LABEL_MAP) - set(self._label_index)
        if missing:
            raise ValueError(
                f"{self.model_name} does not expose the labels {sorted(missing)}; "
                f"found {sorted(self._label_index)}. This scorer needs a 3-way NLI head."
            )

    def _verdict_from_probs(self, probs) -> Verdict:
        p_entail = float(probs[self._label_index[self.SUPPORTED_KEY]])
        winner = max(self.LABEL_MAP, key=lambda k: float(probs[self._label_index[k]]))
        return Verdict(
            # The binary head thresholds P(entailment) rather than taking the argmax,
            # because that is the quantity the leaderboard scores and calibrates. The
            # two can disagree: entailment can win a three-way argmax while still
            # sitting below the threshold. label3 keeps the argmax view.
            supported=p_entail >= self.threshold,
            score=p_entail,
            label3=self.LABEL_MAP[winner],
        )

    def logits(self, items: list[Example]):
        """Raw logits, which temperature scaling needs and probabilities cannot recover."""
        import numpy as np

        if not items:
            return np.zeros((0, 3))
        self._load()
        torch = self._torch

        batches = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            encoded = self._tokenizer(
                [e.context for e in chunk],
                [e.claim for e in chunk],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                batches.append(self._model(**encoded).logits.float().cpu().numpy())

        return np.concatenate(batches, axis=0)

    def score_batch(self, items: list[Example]) -> list[Verdict]:
        if not items:
            return []
        from groundcheck.calibration import collapse_to_binary_logits, softmax

        self._load()
        raw = self.logits(items)

        # Unscaled 3-way probabilities decide label3; the reported P(supported) comes
        # from the marginalized binary problem, which is the one temperature is fitted
        # on and the one a threshold is applied to.
        three_way = softmax(raw)
        entail = self._label_index[self.SUPPORTED_KEY]
        binary = collapse_to_binary_logits(raw, supported_index=entail)
        p_supported = softmax(binary, self.temperature)[:, 1]

        verdicts = []
        for probs, p in zip(three_way, p_supported, strict=True):
            verdict = self._verdict_from_probs(probs)
            verdicts.append(
                Verdict(
                    supported=bool(p >= self.threshold),
                    score=float(p),
                    label3=verdict.label3,
                    p_label3={
                        internal: float(probs[self._label_index[native]])
                        for native, internal in self.LABEL_MAP.items()
                    },
                )
            )
        return verdicts

    def score(self, context: str, claim: str) -> Verdict:
        example = Example(context=context, claim=claim, label="supported")
        return self.score_batch([example])[0]

    def efficiency(self) -> EfficiencyProfile:
        self._load()
        params = sum(p.numel() for p in self._model.parameters())
        bytes_ = sum(p.numel() * p.element_size() for p in self._model.parameters())
        return EfficiencyProfile(
            hosted=False,
            params_m=round(params / 1e6, 1),
            size_mb=round(bytes_ / 1e6, 1),
            cost_per_1k_usd=0.0,
        )
