"""FEVER adapter.

Source: `copenlu/fever_gold_evidence`, which attaches gold evidence sentences to each
claim. Plain `fever/fever` ships evidence as Wikipedia pointers rather than text, and a
groundedness example needs the passage itself.

Labels are native 3-way and map directly onto the internal scheme:

    SUPPORTS         -> supported
    REFUTES          -> contradicted
    NOT ENOUGH INFO  -> neutral

FEVER's role here is the entailment warm-start. It is Wikipedia fact-checking rather
than RAG output, so it teaches the entailment relation cheaply at scale (228k train)
while RAGTruth supplies the in-domain signal.

Note: a small number of NOT ENOUGH INFO rows carry no evidence text (62 of 6,410 in
validation). Those are dropped rather than given an empty context, since "no evidence"
and "evidence that fails to support" are different things and only the latter is a
groundedness judgment.
"""

from __future__ import annotations

from datasets import load_dataset

from groundcheck.data.base import Example, Label3
from groundcheck.data.splits import subsample

REPO = "copenlu/fever_gold_evidence"

_LABEL_MAP: dict[str, Label3] = {
    "SUPPORTS": "supported",
    "REFUTES": "contradicted",
    "NOT ENOUGH INFO": "neutral",
}


def _join_evidence(evidence) -> tuple[str, list[str]]:
    """Flatten [[page, sent_id, sentence], ...] into one passage plus its page titles.

    Sentences are deduplicated while preserving order: FEVER frequently cites the same
    sentence from multiple evidence sets, and repeating it in the context would inflate
    the passage without adding information.
    """
    sentences: list[str] = []
    pages: list[str] = []
    seen: set[str] = set()

    for item in evidence or []:
        if not item or len(item) < 3:
            continue
        page, sentence = item[0], (item[2] or "").strip()
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)
        if page and page not in pages:
            pages.append(page)

    return " ".join(sentences), pages


def to_example(row: dict, split: str) -> Example | None:
    """Convert one native row. Returns None when the row carries no usable evidence."""
    label = _LABEL_MAP.get((row.get("label") or "").strip())
    if label is None:
        return None

    context, pages = _join_evidence(row.get("evidence"))
    claim = (row.get("claim") or "").strip()
    if not context or not claim:
        return None

    return Example(
        context=context,
        claim=claim,
        label=label,
        meta={
            "dataset": "fever",
            "domain": "wiki",
            "id": f"fever-{row.get('id')}",
            "task": "fact-check",
            "granularity": "claim",
            "label3_source": "native",
            "split": split,
            "pages": pages,
            "verifiable": row.get("verifiable"),
        },
    )


class Fever:
    name = "fever"
    domain = "wiki"

    def __init__(self, repo: str = REPO):
        self.repo = repo

    def load(self, split: str, limit: int | None = None) -> list[Example]:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"fever has splits train/validation/test, got {split!r}")

        raw = subsample(load_dataset(self.repo, split=split), limit)
        out = [to_example(row, split) for row in raw]
        return [e for e in out if e is not None]
