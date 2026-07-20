"""LLM-AggreFact adapter.

Source: `lytang/LLM-AggreFact`, the aggregation of eleven factual-consistency datasets
that the public leaderboard ranks on. This is the primary evaluation surface: one
schema, many domains, and numbers directly comparable to published entries.

Access is gated (auto-approved). Accept the terms on the dataset page and provide a
read token via `HF_TOKEN` before loading.

Three things about this source shape how it is used.

**Evaluation only.** It ships `dev` and `test` and no train split, by design. Nothing
here should ever be trained on.

**Labels are binary.** The source records supported or not, without distinguishing a
contradiction from an unsupported claim, so unsupported rows are mapped to "neutral"
and tagged `label3_source="coarse"`, the same treatment HaluEval gets.

**It contains RAGTruth, among others.** Training on RAGTruth and then reporting an
AggreFact number is not leaderboard-comparable. Every row carries the upstream corpus
in `meta["source_dataset"]` and the upstream `contamination_identifier`, which is what
`groundcheck.data.decontaminate` uses to make the overlap measurable rather than
assumed.

Licensing note: CC-BY-ND-4.0. Evaluating against it is fine; publishing a modified or
re-split derivative is not. Curated datasets released from this project must not be
derived from it.
"""

from __future__ import annotations

from datasets import load_dataset

from groundcheck.data.base import Example
from groundcheck.data.splits import subsample

REPO = "lytang/LLM-AggreFact"

# The source encodes support as 1 and non-support as 0. Asserted by
# `test_aggrefact_live.py`, which checks that an NLI baseline scores meaningfully above
# chance: an inverted mapping would put it symmetrically below.
SUPPORTED_LABEL = 1


def to_example(row: dict, split: str) -> Example | None:
    context = (row.get("doc") or "").strip()
    claim = (row.get("claim") or "").strip()
    if not context or not claim:
        return None

    label = row.get("label")
    if label is None:
        return None
    supported = int(label) == SUPPORTED_LABEL

    source = row.get("dataset") or "unknown"
    return Example(
        context=context,
        claim=claim,
        label="supported" if supported else "neutral",
        meta={
            "dataset": "aggrefact",
            "domain": source.lower(),
            "id": f"aggrefact-{split}-{row.get('contamination_identifier') or source}",
            "task": "fact-consistency",
            "granularity": "claim",
            "label3_source": "native" if supported else "coarse",
            "split": split,
            "source_dataset": source,
            "contamination_identifier": row.get("contamination_identifier"),
        },
    )


class AggreFact:
    name = "aggrefact"
    domain = "mixed"

    def __init__(self, repo: str = REPO, source_datasets: tuple[str, ...] | None = None):
        self.repo = repo
        # Restrict to specific upstream corpora, e.g. to exclude the ones a scorer was
        # trained on, or to report a per-corpus breakdown.
        self.source_datasets = source_datasets

    def load(self, split: str, limit: int | None = None) -> list[Example]:
        if split not in {"dev", "test"}:
            raise ValueError(f"aggrefact is evaluation-only and has splits dev/test, got {split!r}")

        raw = load_dataset(self.repo, split=split)
        if self.source_datasets:
            raw = raw.filter(lambda row: row["dataset"] in self.source_datasets)

        out = [to_example(row, split) for row in subsample(raw, limit)]
        return [e for e in out if e is not None]
