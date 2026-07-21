from collections import Counter

import pytest

from groundcontrol.data.splits import assign_split, subsample


def test_assignment_is_stable_across_calls():
    # The whole point of hashing rather than shuffling: the same key must land in the
    # same split on every machine, run, and library version.
    assert assign_split("halueval-qa-42") == assign_split("halueval-qa-42")


def test_known_keys_are_pinned():
    # Regression guard. If the hash or bucketing changes, previously-trained splits
    # would silently reshuffle and leak across train/test.
    assert assign_split("halueval-qa-0") == "train"
    assert assign_split("halueval-qa-1") == "train"
    assert assign_split("halueval-qa-13") == "test"


def test_distribution_is_close_to_the_requested_ratios():
    counts = Counter(assign_split(f"row-{i}") for i in range(10_000))
    assert 0.77 < counts["train"] / 10_000 < 0.83
    assert 0.07 < counts["validation"] / 10_000 < 0.13
    assert 0.07 < counts["test"] / 10_000 < 0.13


def test_custom_ratios():
    counts = Counter(
        assign_split(f"row-{i}", {"train": 0.5, "validation": 0.25, "test": 0.25})
        for i in range(4_000)
    )
    assert 0.47 < counts["train"] / 4_000 < 0.53


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_split("x", {"train": 0.5, "test": 0.4})


class _FakeSplit:
    """Stands in for a datasets.Dataset ordered by corpus, as the real ones are."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        ordered = sorted(self.rows, key=lambda r: hash((seed, r["i"])) % 997)
        return _FakeSplit(ordered)

    def select(self, indices):
        return _FakeSplit([self.rows[i] for i in indices])


def test_subsample_spans_the_whole_split_not_just_the_head():
    # The bug this closes: published sets are ordered by corpus, so head-slicing
    # returns one corpus. AggreFact's first 300 test rows are 95% supported.
    rows = [{"i": i, "corpus": "a" if i < 500 else "b"} for i in range(1000)]
    sampled = subsample(_FakeSplit(rows), 200)

    assert len({r["corpus"] for r in sampled.rows}) == 2


def test_subsample_is_reproducible():
    rows = [{"i": i} for i in range(100)]
    first = [r["i"] for r in subsample(_FakeSplit(rows), 10).rows]
    second = [r["i"] for r in subsample(_FakeSplit(rows), 10).rows]
    assert first == second


def test_subsample_is_a_noop_without_a_limit_or_when_limit_exceeds_size():
    rows = [{"i": i} for i in range(10)]
    split = _FakeSplit(rows)
    assert subsample(split, None) is split
    assert subsample(split, 50) is split
