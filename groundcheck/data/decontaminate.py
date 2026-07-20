"""Keep training data out of the evaluation sets.

The trap this exists to close: LLM-AggreFact aggregates eleven corpora, and RAGTruth is
one of them. Fine-tuning on RAGTruth's train split and then reporting an AggreFact
number produces a result that looks leaderboard-comparable and is not, because the
published entries it would sit beside are zero-shot.

The check is content-based rather than id-based. Ids do not survive aggregation: the
same passage and claim appear in AggreFact under an entirely different identifier than
in RAGTruth, so comparing ids would report zero overlap while the overlap is real.

Matching is exact after normalization (case, whitespace, and surrounding punctuation).
That catches redistribution, which is the actual failure mode here, since aggregators
copy text verbatim. It does not catch paraphrase or truncation. Near-duplicate detection
(MinHash/LSH) is the upgrade when curated data enters the mix, where restatement is
generated rather than copied.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from groundcheck.data.base import Example

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    return _WHITESPACE.sub(" ", text)


def fingerprint(example: Example) -> str:
    """A content hash over the pair, since a passage may be reused with other claims."""
    joined = f"{normalize(example.context)}\x00{normalize(example.claim)}"
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(slots=True)
class DecontaminationReport:
    n_before: int
    n_removed: int
    by_eval_source: dict[str, int] = field(default_factory=dict)

    @property
    def n_after(self) -> int:
        return self.n_before - self.n_removed

    @property
    def removed_rate(self) -> float:
        return self.n_removed / self.n_before if self.n_before else 0.0

    def summary(self) -> str:
        if not self.n_removed:
            return f"No overlap found across {self.n_before} training examples."
        parts = ", ".join(f"{src}: {n}" for src, n in sorted(self.by_eval_source.items()))
        return (
            f"Removed {self.n_removed} of {self.n_before} training examples "
            f"({self.removed_rate:.1%}) found in evaluation data [{parts}]."
        )


def build_index(eval_examples: list[Example]) -> dict[str, str]:
    """Fingerprint -> the upstream corpus it came from, for attributing each removal."""
    index: dict[str, str] = {}
    for example in eval_examples:
        source = example.meta.get("source_dataset") or example.meta.get("dataset") or "unknown"
        index[fingerprint(example)] = str(source)
    return index


def decontaminate(
    train_examples: list[Example],
    eval_examples: list[Example],
) -> tuple[list[Example], DecontaminationReport]:
    """Drop training examples whose (context, claim) pair appears in the evaluation data."""
    index = build_index(eval_examples)

    kept: list[Example] = []
    removed: Counter[str] = Counter()
    for example in train_examples:
        source = index.get(fingerprint(example))
        if source is None:
            kept.append(example)
        else:
            removed[source] += 1

    return kept, DecontaminationReport(
        n_before=len(train_examples),
        n_removed=sum(removed.values()),
        by_eval_source=dict(removed),
    )


def overlap_report(
    train_examples: list[Example], eval_examples: list[Example]
) -> DecontaminationReport:
    """Measure overlap without removing anything, for reporting on an existing run."""
    return decontaminate(train_examples, eval_examples)[1]
