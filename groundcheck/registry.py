"""Lazy name -> component lookup for datasets and scorers.

Entries are declared as `"module:attr"` strings and imported only when something is
actually constructed. This matters because components carry heavy, *optional*
dependencies: the dataset adapters need `datasets`, the scorers need `torch` and
`transformers`. A registry that populated itself through import side effects would make
listing the available names require importing every backend, so a bare install could
not even import the metrics module.

Registering by string keeps the cost proportional: `available_scorers()` is free,
`get_scorer("nli-zeroshot")` pays for torch, and nothing else does.

`register_*` also accepts an object directly, which is what tests and third-party code
use when the class is already in hand.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


class _Registry:
    def __init__(self, kind: str):
        self._kind = kind
        self._entries: dict[str, str | Callable[..., Any]] = {}

    def register(self, name: str, target: str | Callable[..., Any]) -> None:
        if name in self._entries:
            raise ValueError(f"{self._kind} {name!r} is already registered")
        self._entries[name] = target

    def get(self, name: str, **kwargs: Any) -> Any:
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "(none registered)"
            raise KeyError(f"unknown {self._kind} {name!r}; available: {available}")

        target = self._entries[name]
        if isinstance(target, str):
            module_path, _, attr = target.partition(":")
            target = getattr(importlib.import_module(module_path), attr)
            self._entries[name] = target  # resolved once, cached
        return target(**kwargs)

    def available(self) -> list[str]:
        return sorted(self._entries)


_datasets = _Registry("dataset")
_scorers = _Registry("scorer")

# Declared, not imported. Each value is resolved on first `get_*`.
_datasets.register("ragtruth", "groundcheck.data.ragtruth:RAGTruth")
_datasets.register("fever", "groundcheck.data.fever:Fever")
_datasets.register("halueval", "groundcheck.data.halueval:HaluEval")


def register_dataset(name: str, target: str | Callable[..., Any]) -> None:
    """Register a dataset, by import path or by object."""
    _datasets.register(name, target)


def register_scorer(name: str, target: str | Callable[..., Any]) -> None:
    """Register a scorer, by import path or by object."""
    _scorers.register(name, target)


def get_dataset(name: str, **kwargs: Any) -> Any:
    return _datasets.get(name, **kwargs)


def get_scorer(name: str, **kwargs: Any) -> Any:
    return _scorers.get(name, **kwargs)


def available_datasets() -> list[str]:
    return _datasets.available()


def available_scorers() -> list[str]:
    return _scorers.available()
