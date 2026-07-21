from __future__ import annotations

import pytest

from groundcontrol.data import aggrefact
from groundcontrol.registry import get_dataset


def _row(label: int = 1, **over):
    row = {
        "dataset": "RAGTruth",
        "doc": "Acme reported Q2 revenue of $4.2M.",
        "claim": "Acme reported Q2 revenue of $4.2M.",
        "label": label,
        "contamination_identifier": "abc123",
    }
    row.update(over)
    return row


def test_binary_labels_map_onto_the_three_way_scheme():
    assert aggrefact.to_example(_row(label=1), "test").label == "supported"
    # AggreFact does not separate contradiction from unsupported, so the unsupported
    # side is neutral and marked coarse rather than claiming a contradiction.
    unsupported = aggrefact.to_example(_row(label=0), "test")
    assert unsupported.label == "neutral"
    assert unsupported.meta["label3_source"] == "coarse"
    assert unsupported.supported is False


def test_upstream_corpus_is_recorded_for_contamination_checks():
    ex = aggrefact.to_example(_row(), "test")
    assert ex.meta["source_dataset"] == "RAGTruth"
    assert ex.meta["contamination_identifier"] == "abc123"


def test_rows_without_text_or_label_are_dropped():
    assert aggrefact.to_example(_row(doc=""), "test") is None
    assert aggrefact.to_example(_row(claim="   "), "test") is None
    assert aggrefact.to_example(_row(label=None), "test") is None


def test_training_splits_are_refused():
    # AggreFact ships dev and test only. Asking for a train split is a mistake worth
    # failing loudly on, since training on it would corrupt the benchmark.
    with pytest.raises(ValueError, match="evaluation-only"):
        aggrefact.AggreFact().load("train")


def test_registered_and_named():
    assert get_dataset("aggrefact").name == "aggrefact"
