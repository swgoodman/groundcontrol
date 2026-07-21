"""Live schema checks against the HuggingFace Hub.

Deselected by default; run with `uv run pytest -m network`.

The hermetic tests in `test_adapters.py` pin the *mapping* against synthetic rows. These
pin the *assumption* that the real sources still have the field names and value
vocabularies those fixtures imitate. When an upstream dataset is re-uploaded with a
renamed column, this is what catches it, and the failure is loud rather than a silently
empty split.
"""

from __future__ import annotations

import pytest

from groundcontrol.data.base import LABEL3_SOURCES, LABELS_3
from groundcontrol.data.fever import Fever
from groundcontrol.data.halueval import CONFIGS, HaluEval
from groundcontrol.data.ragtruth import RAGTruth

pytestmark = pytest.mark.network


def _assert_well_formed(examples, dataset_name):
    assert examples, f"{dataset_name} returned no examples"
    for ex in examples:
        assert ex.context.strip() and ex.claim.strip()
        assert ex.label in LABELS_3
        assert ex.meta["dataset"] == dataset_name
        assert ex.meta["label3_source"] in LABEL3_SOURCES
        assert ex.meta["id"]


def test_ragtruth_live_schema():
    examples = RAGTruth().load("test", limit=200)
    _assert_well_formed(examples, "ragtruth")
    # All three RAG tasks should survive the mapping; "unknown" means a new task name.
    domains = {e.meta["domain"] for e in examples}
    assert domains <= {"qa", "summarization", "data2text"}
    # The test split is majority-supported, which is the imbalance the metrics module
    # is built around. If this ever flips, the class-of-interest framing needs revisiting.
    supported = sum(e.supported for e in examples)
    assert 0.4 < supported / len(examples) < 0.9


def test_fever_live_schema():
    examples = Fever().load("validation", limit=300)
    _assert_well_formed(examples, "fever")
    assert all(e.meta["label3_source"] == "native" for e in examples)
    # All three native labels should appear in any reasonable slice.
    assert {e.label for e in examples} == set(LABELS_3)


def test_halueval_live_schema():
    for config in CONFIGS:
        examples = HaluEval(configs=(config,)).load("all", limit=50)
        _assert_well_formed(examples, "halueval")
        assert all(e.meta["config"] == config for e in examples)
    # Pairs make the source close to balanced, unlike RAGTruth.
    both = HaluEval().load("all", limit=200)
    supported = sum(e.supported for e in both)
    assert 0.4 < supported / len(both) < 0.6


def test_halueval_pairs_never_straddle_a_split():
    # A hallucinated answer in test whose correct twin is in train would leak.
    examples = HaluEval(configs=("qa",)).load("all", limit=400)
    by_pair: dict[str, set[str]] = {}
    for ex in examples:
        by_pair.setdefault(ex.meta["pair_id"], set()).add(ex.meta["split"])
    assert all(len(splits) == 1 for splits in by_pair.values())
