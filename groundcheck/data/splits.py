"""Deterministic splitting for sources that ship without one.

Some datasets (HaluEval) publish a single undivided pool. Assigning a split by
hashing a stable per-row key keeps the partition identical across machines, runs, and
library versions, which an RNG-shuffled split does not. It also means a row's split is
a pure function of its id, so decontamination can be checked without materializing the
splits first.
"""

from __future__ import annotations

import hashlib

SPLITS: tuple[str, ...] = ("train", "validation", "test")

DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}

_BUCKETS = 10_000


def assign_split(key: str, ratios: dict[str, float] | None = None) -> str:
    """Map a stable key to a split. Same key always yields the same split."""
    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % _BUCKETS

    edge = 0.0
    for split in SPLITS:
        if split not in ratios:
            continue
        edge += ratios[split]
        if bucket < edge * _BUCKETS:
            return split
    return SPLITS[-1]
