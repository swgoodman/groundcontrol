from __future__ import annotations

from groundcheck.data.base import Example
from groundcheck.data.decontaminate import (
    decontaminate,
    fingerprint,
    normalize,
    overlap_report,
)


def _ex(context: str, claim: str, **meta) -> Example:
    return Example(context=context, claim=claim, label="supported", meta=meta)


def test_normalization_ignores_case_and_whitespace():
    assert normalize("  The   Cat\nsat ") == normalize("the cat sat")


def test_fingerprint_matches_redistributed_text_across_ids():
    # The point of the whole module: AggreFact carries RAGTruth rows under different
    # identifiers, so id comparison would report no overlap while the overlap is real.
    a = _ex("Acme earned $4.2M.", "Acme earned $4.2M.", dataset="ragtruth", id="rt-1")
    b = _ex("acme earned $4.2m.", "Acme  earned $4.2M.", dataset="aggrefact", id="af-9")
    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_separates_context_from_claim():
    # Without a separator, ("ab", "c") and ("a", "bc") would collide.
    assert fingerprint(_ex("ab", "c")) != fingerprint(_ex("a", "bc"))


def test_same_passage_with_a_different_claim_is_not_a_match():
    passage = "Acme earned $4.2M."
    assert fingerprint(_ex(passage, "Acme earned $4.2M.")) != fingerprint(
        _ex(passage, "Acme earned $9.9M.")
    )


def test_decontaminate_removes_only_the_overlapping_examples():
    train = [
        _ex("shared passage", "shared claim", dataset="ragtruth"),
        _ex("unique passage", "unique claim", dataset="ragtruth"),
    ]
    evaluation = [_ex("shared passage", "shared claim", dataset="aggrefact")]

    kept, report = decontaminate(train, evaluation)

    assert [e.claim for e in kept] == ["unique claim"]
    assert (report.n_before, report.n_removed, report.n_after) == (2, 1, 1)


def test_report_attributes_removals_to_the_upstream_corpus():
    # Which corpus caused the removal is the actionable part: it names the benchmark
    # whose comparability was at risk.
    train = [_ex("p", "c", dataset="ragtruth")]
    evaluation = [_ex("p", "c", dataset="aggrefact", source_dataset="RAGTruth")]

    _, report = decontaminate(train, evaluation)

    assert report.by_eval_source == {"RAGTruth": 1}
    assert "RAGTruth: 1" in report.summary()


def test_clean_training_data_is_left_untouched():
    train = [_ex("a", "b"), _ex("c", "d")]
    kept, report = decontaminate(train, [_ex("x", "y")])

    assert len(kept) == 2
    assert report.n_removed == 0
    assert report.removed_rate == 0.0
    assert "No overlap" in report.summary()


def test_overlap_report_measures_without_removing():
    train = [_ex("p", "c")]
    report = overlap_report(train, [_ex("p", "c")])

    assert report.n_removed == 1
    assert len(train) == 1


def test_empty_inputs_are_safe():
    kept, report = decontaminate([], [_ex("a", "b")])
    assert kept == [] and report.removed_rate == 0.0
