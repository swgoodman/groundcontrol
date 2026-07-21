"""Hermetic adapter tests.

Row conversion is a pure function in every adapter, so the mapping that matters can be
tested against synthetic rows shaped like the real ones. Fixtures mirror the field names
and value vocabularies observed on the Hub; `test_adapters_live.py` checks that those
shapes still hold.
"""

from __future__ import annotations

import pytest

from groundcontrol.data import fever, halueval, ragtruth
from groundcontrol.registry import available_datasets, get_dataset

# --- RAGTruth ---------------------------------------------------------------------


def _rt_row(evident: int = 0, baseless: int = 0, **over):
    row = {
        "id": "1183",
        "query": "Summarize the following news within 86 words:",
        "context": "Acme Corp reported Q2 revenue of $4.2M. It did not disclose net income.",
        "output": "Acme posted Q2 revenue of $4.2M and net income of $0.5M.",
        "task_type": "Summary",
        "quality": "good",
        "model": "gpt-3.5-turbo",
        "hallucination_labels": "[]",
        "hallucination_labels_processed": {
            "evident_conflict": evident,
            "baseless_info": baseless,
        },
    }
    row.update(over)
    return row


@pytest.mark.parametrize(
    ("evident", "baseless", "expected"),
    [
        (0, 0, "supported"),
        (0, 1, "neutral"),
        (1, 0, "contradicted"),
        # Annotated as both: contradiction is the stronger signal and wins.
        (1, 1, "contradicted"),
    ],
)
def test_ragtruth_label_mapping(evident, baseless, expected):
    ex = ragtruth.to_example(_rt_row(evident, baseless), "test")
    assert ex.label == expected
    assert ex.meta["label3_source"] == "native"


def test_ragtruth_maps_task_to_domain():
    for task, domain in [("QA", "qa"), ("Summary", "summarization"), ("Data2txt", "data2text")]:
        ex = ragtruth.to_example(_rt_row(task_type=task), "test")
        assert ex.meta["domain"] == domain
    assert ragtruth.to_example(_rt_row(task_type="Novel"), "test").meta["domain"] == "unknown"


def test_ragtruth_keeps_query_and_spans_for_downstream_use():
    spans = [{"text": "net income of $0.5M", "type": "Evident Baseless Info"}]
    import json

    ex = ragtruth.to_example(_rt_row(baseless=1, hallucination_labels=json.dumps(spans)), "test")
    assert ex.meta["spans"] == spans
    assert ex.meta["query"].startswith("Summarize")


def test_ragtruth_tolerates_unparseable_spans():
    assert ragtruth.to_example(_rt_row(hallucination_labels="not json"), "test").meta["spans"] == []
    assert ragtruth.to_example(_rt_row(hallucination_labels=None), "test").meta["spans"] == []


def test_ragtruth_drops_rows_without_text():
    assert ragtruth.to_example(_rt_row(context="   "), "test") is None
    assert ragtruth.to_example(_rt_row(output=""), "test") is None


# --- FEVER ------------------------------------------------------------------------


def _fv_row(label="SUPPORTS", evidence=None, **over):
    row = {
        "id": "62037",
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "label": label,
        "evidence": evidence
        if evidence is not None
        else [["Nikolaj_Coster-Waldau", "0", "He played Detective John Amsterdam on Fox."]],
        "verifiable": "VERIFIABLE",
    }
    row.update(over)
    return row


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("SUPPORTS", "supported"),
        ("REFUTES", "contradicted"),
        ("NOT ENOUGH INFO", "neutral"),
    ],
)
def test_fever_label_mapping(native, expected):
    ex = fever.to_example(_fv_row(native), "validation")
    assert ex.label == expected
    assert ex.meta["label3_source"] == "native"


def test_fever_joins_evidence_and_dedupes_repeated_sentences():
    # FEVER cites the same sentence from multiple evidence sets; repeating it would
    # pad the context without adding information.
    evidence = [
        ["Page_A", "0", "First sentence."],
        ["Page_A", "0", "First sentence."],
        ["Page_B", "3", "Second sentence."],
    ]
    ex = fever.to_example(_fv_row(evidence=evidence), "validation")
    assert ex.context == "First sentence. Second sentence."
    assert ex.meta["pages"] == ["Page_A", "Page_B"]


def test_fever_drops_evidence_free_rows():
    # 62 of 6,410 validation NEI rows have no evidence text. "No evidence" is not a
    # groundedness judgment, so these are dropped rather than given an empty context.
    assert fever.to_example(_fv_row("NOT ENOUGH INFO", evidence=[]), "validation") is None
    assert fever.to_example(_fv_row(evidence=[["Page", "0", "  "]]), "validation") is None
    assert fever.to_example(_fv_row(evidence=[["Page"]]), "validation") is None


def test_fever_ignores_unknown_labels():
    assert fever.to_example(_fv_row("DISPUTED"), "validation") is None


# --- HaluEval ---------------------------------------------------------------------


def _he_qa_row(**over):
    row = {
        "knowledge": "Arthur's Magazine was an American literary periodical, first published 1844.",
        "question": "Which magazine was started first?",
        "right_answer": "Arthur's Magazine",
        "hallucinated_answer": "First for Women was started first.",
    }
    row.update(over)
    return row


def test_halueval_row_yields_a_supported_and_unsupported_pair():
    out = halueval.to_examples(_he_qa_row(), "qa", 0, split="train")
    assert [e.label for e in out] == ["supported", "neutral"]
    assert [e.supported for e in out] == [True, False]


def test_halueval_marks_only_the_hallucinated_half_as_coarse():
    # HaluEval says an answer is hallucinated but not whether it contradicts the
    # passage or merely invents detail, so the 3-way label there is not trustworthy.
    pos, neg = halueval.to_examples(_he_qa_row(), "qa", 0, split="train")
    assert pos.meta["label3_source"] == "native"
    assert neg.meta["label3_source"] == "coarse"


def test_halueval_pairs_share_a_split_and_a_pair_id():
    out = halueval.to_examples(_he_qa_row(), "qa", 7)
    assert len({e.meta["split"] for e in out}) == 1
    assert len({e.meta["pair_id"] for e in out}) == 1


def test_halueval_config_field_mapping():
    summ = halueval.to_examples(
        {
            "document": "A document.",
            "right_summary": "Correct.",
            "hallucinated_summary": "Invented.",
        },
        "summarization",
        0,
        split="train",
    )
    assert summ[0].context == "A document." and summ[0].claim == "Correct."

    dial = halueval.to_examples(
        {
            "knowledge": "Iron Man stars Robert Downey Jr.",
            "dialogue_history": "[Human]: Do you like Iron Man",
            "right_response": "Sure do.",
            "hallucinated_response": "RDJ starred with Tom Hanks.",
        },
        "dialogue",
        0,
        split="train",
    )
    assert dial[1].claim == "RDJ starred with Tom Hanks."
    assert dial[0].meta["query"].startswith("[Human]")


def test_halueval_skips_rows_without_a_passage():
    assert halueval.to_examples(_he_qa_row(knowledge=""), "qa", 0) == []


def test_halueval_skips_a_missing_half_without_dropping_the_other():
    out = halueval.to_examples(_he_qa_row(hallucinated_answer=""), "qa", 0, split="train")
    assert [e.label for e in out] == ["supported"]


def test_qa_is_excluded_by_default_but_still_available():
    # Correct QA answers are bare spans, so an entailment scorer reads them as
    # unsupported regardless of truth. Opt in explicitly for curation experiments.
    assert "qa" not in halueval.DEFAULT_CONFIGS
    assert halueval.HaluEval().configs == ("summarization", "dialogue")
    assert halueval.HaluEval(configs=("qa",)).configs == ("qa",)


def test_halueval_rejects_the_general_config():
    # `general` has no source passage, so it cannot support a groundedness judgment.
    assert "general" not in halueval.CONFIGS
    with pytest.raises(ValueError, match="no source passage"):
        halueval.HaluEval(configs=("general",))


# --- registry ---------------------------------------------------------------------


def test_adapters_are_registered_by_name():
    for name in ("ragtruth", "halueval", "fever"):
        assert name in available_datasets()
        assert get_dataset(name).name == name
