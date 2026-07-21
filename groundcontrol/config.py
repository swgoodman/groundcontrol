"""Declarative experiment specs.

A run is scorers x datasets in a YAML file, so a report can name the config that
produced it and anyone can reproduce it without reading the script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ComponentSpec:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSpec(ComponentSpec):
    split: str = "test"
    limit: int | None = None


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    scorers: list[ComponentSpec]
    datasets: list[DatasetSpec]
    benchmark_device: str = "cpu"
    benchmark_device_label: str | None = None
    benchmark_n: int = 50
    output_dir: str = "reports"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> ExperimentConfig:
        missing = {"name", "scorers", "datasets"} - set(raw)
        if missing:
            raise ValueError(f"config is missing required keys: {sorted(missing)}")

        scorers = [ComponentSpec(name=s["name"], args=s.get("args", {})) for s in raw["scorers"]]
        datasets = [
            DatasetSpec(
                name=d["name"],
                args=d.get("args", {}),
                split=d.get("split", "test"),
                limit=d.get("limit"),
            )
            for d in raw["datasets"]
        ]
        return cls(
            name=raw["name"],
            scorers=scorers,
            datasets=datasets,
            benchmark_device=raw.get("benchmark_device", "cpu"),
            benchmark_device_label=raw.get("benchmark_device_label"),
            benchmark_n=raw.get("benchmark_n", 50),
            output_dir=raw.get("output_dir", "reports"),
        )
