"""Name -> factory lookup for datasets and scorers.

Config files address components by string name, so adding a scorer or dataset is one
new file plus one `@register_*` decorator. No refactor, no central import list to edit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_SCORERS: dict[str, Callable[..., Any]] = {}
_DATASETS: dict[str, Callable[..., Any]] = {}


def _make_register(table: dict[str, Callable[..., Any]], kind: str):
    def register(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in table:
                raise ValueError(f"{kind} {name!r} is already registered")
            table[name] = factory
            return factory

        return decorator

    return register


def _make_get(table: dict[str, Callable[..., Any]], kind: str):
    def get(name: str, **kwargs: Any) -> Any:
        if name not in table:
            available = ", ".join(sorted(table)) or "(none registered)"
            raise KeyError(f"unknown {kind} {name!r}; available: {available}")
        return table[name](**kwargs)

    return get


register_scorer = _make_register(_SCORERS, "scorer")
register_dataset = _make_register(_DATASETS, "dataset")
get_scorer = _make_get(_SCORERS, "scorer")
get_dataset = _make_get(_DATASETS, "dataset")


def available_scorers() -> list[str]:
    return sorted(_SCORERS)


def available_datasets() -> list[str]:
    return sorted(_DATASETS)
