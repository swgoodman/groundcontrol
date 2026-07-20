"""RAGTruth adapter.

Source: `wandb/RAGTruth-processed` (15,090 train / 2,700 test), response-level
hallucination annotations over three RAG tasks: QA, summarization, and data-to-text.

RAGTruth is the only source here that annotates *which kind* of hallucination occurred,
so its 3-way label is native rather than inferred:

    evident_conflict  -> contradicted   (the response conflicts with the source)
    baseless_info     -> neutral        (the response invents detail the source lacks)
    neither           -> supported

When a response is annotated with both, contradiction wins: a response that both
conflicts and invents is a conflict, and the stronger signal is the one worth training.
"""

from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset

from groundcheck.data.base import Example, Label3
from groundcheck.registry import register_dataset

REPO = "wandb/RAGTruth-processed"

# RAGTruth's own task names -> the domain recorded on every example.
_TASK_DOMAIN = {
    "QA": "qa",
    "Summary": "summarization",
    "Data2txt": "data2text",
}


def _label_from_counts(counts: dict[str, int] | None) -> Label3:
    counts = counts or {}
    if counts.get("evident_conflict", 0):
        return "contradicted"
    if counts.get("baseless_info", 0):
        return "neutral"
    return "supported"


def _parse_spans(raw: Any) -> list[dict]:
    """Span annotations ship as a JSON string; keep them for the demo's highlighting."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def to_example(row: dict, split: str) -> Example | None:
    """Convert one native row. Returns None for rows that cannot be judged."""
    context = (row.get("context") or "").strip()
    claim = (row.get("output") or "").strip()
    if not context or not claim:
        return None

    task = row.get("task_type") or ""
    return Example(
        context=context,
        claim=claim,
        label=_label_from_counts(row.get("hallucination_labels_processed")),
        meta={
            "dataset": "ragtruth",
            "domain": _TASK_DOMAIN.get(task, "unknown"),
            "id": f"ragtruth-{row.get('id')}",
            "task": task,
            "granularity": "answer",
            "label3_source": "native",
            "split": split,
            # The instruction the response was generated from. Not part of the
            # groundedness judgment, but the query a context-precision scorer needs.
            "query": row.get("query"),
            "generator_model": row.get("model"),
            "quality": row.get("quality"),
            "spans": _parse_spans(row.get("hallucination_labels")),
        },
    )


@register_dataset("ragtruth")
class RAGTruth:
    name = "ragtruth"
    domain = "rag"

    def __init__(self, repo: str = REPO):
        self.repo = repo

    def load(self, split: str, limit: int | None = None) -> list[Example]:
        if split not in {"train", "test"}:
            raise ValueError(f"ragtruth has splits train/test, got {split!r}")

        raw = load_dataset(self.repo, split=split)
        if limit is not None:
            raw = raw.select(range(min(limit, len(raw))))

        out = [to_example(row, split) for row in raw]
        return [e for e in out if e is not None]
