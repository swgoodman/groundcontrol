"""Run an experiment config and write a leaderboard.

uv run python scripts/run_eval.py configs/phase0_smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from groundcheck.config import ExperimentConfig
from groundcheck.device import BenchmarkDevice
from groundcheck.eval import report
from groundcheck.eval.runner import RunResult, run
from groundcheck.registry import get_dataset, get_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--no-benchmark", action="store_true", help="skip latency measurement")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    benchmark_device = None
    if not args.no_benchmark:
        if not cfg.benchmark_device_label:
            raise SystemExit(
                "config must set benchmark_device_label: an unlabeled latency number "
                "is not comparable across machines. Use --no-benchmark to skip timing."
            )
        benchmark_device = BenchmarkDevice(
            device=cfg.benchmark_device, label=cfg.benchmark_device_label
        )

    results: list[RunResult] = []
    for scorer_spec in cfg.scorers:
        scorer = get_scorer(scorer_spec.name, **scorer_spec.args)
        for dataset_spec in cfg.datasets:
            dataset = get_dataset(dataset_spec.name, **dataset_spec.args)
            examples = dataset.load(dataset_spec.split, limit=dataset_spec.limit)
            print(f"{scorer.name} x {dataset_spec.name}/{dataset_spec.split}: {len(examples)}")
            results.append(
                run(
                    scorer,
                    examples,
                    dataset_name=dataset_spec.name,
                    split=dataset_spec.split,
                    benchmark_device=benchmark_device,
                    benchmark_n=cfg.benchmark_n,
                )
            )

    paths = report.write(results, Path(cfg.output_dir), stem=cfg.name)
    print(f"\n{report.to_markdown(results, title=cfg.name)}")
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
