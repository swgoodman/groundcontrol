from collections import Counter

import pytest

from groundcheck.data.splits import assign_split


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
