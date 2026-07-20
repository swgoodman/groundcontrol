"""Unified example schema.

Every source dataset normalizes onto `Example` so nothing downstream special-cases a
dataset. Source-specific detail goes in `meta`, never into new top-level fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Label3 = Literal["supported", "contradicted", "neutral"]

LABELS_3: tuple[Label3, ...] = ("supported", "contradicted", "neutral")

# Polarity, stated once so it is never re-derived:
#   3-way internal label -> binary reported label
#   supported                -> True   (grounded)
#   contradicted | neutral   -> False  (NOT supported; the rare, costly, class of interest)
# Headline metrics report on the not-supported class.
NOT_SUPPORTED: frozenset[str] = frozenset({"contradicted", "neutral"})


def is_supported(label: Label3) -> bool:
    """Collapse the 3-way internal label to the binary reported label."""
    if label not in LABELS_3:
        raise ValueError(f"unknown label {label!r}; expected one of {LABELS_3}")
    return label == "supported"


@dataclass(slots=True)
class Example:
    context: str
    """Evidence / source passage(s), joined."""

    claim: str
    """The answer under test, or a decomposed sub-claim."""

    label: Label3
    """Gold 3-way label. Collapse with `is_supported` for reporting."""

    meta: dict[str, Any] = field(default_factory=dict)
    """{dataset, domain, id, task, granularity, ...} plus source-specific fields."""

    def __post_init__(self) -> None:
        if self.label not in LABELS_3:
            raise ValueError(f"unknown label {self.label!r}; expected one of {LABELS_3}")
        if not self.context:
            raise ValueError("context must be non-empty")
        if not self.claim:
            raise ValueError("claim must be non-empty")

    @property
    def supported(self) -> bool:
        return is_supported(self.label)


@runtime_checkable
class Dataset(Protocol):
    """A source of `Example`s, addressable by split."""

    name: str
    domain: str

    def load(self, split: str) -> list[Example]: ...
