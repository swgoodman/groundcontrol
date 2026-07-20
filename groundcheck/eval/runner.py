"""Run a scorer over a dataset and produce one comparable result.

Everything a leaderboard row needs comes from here, and quality metrics come from
`eval.metrics` rather than being recomputed locally, so a training log and a report can
never disagree about what balanced accuracy means.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from groundcheck.data.base import Example
from groundcheck.device import BenchmarkDevice, describe_runtime
from groundcheck.eval import metrics as metrics_mod
from groundcheck.eval.efficiency import Efficiency
from groundcheck.eval.efficiency import measure as measure_efficiency
from groundcheck.eval.metrics import Metrics


@dataclass(slots=True)
class RunResult:
    scorer: str
    dataset: str
    split: str
    metrics: Metrics
    efficiency: Efficiency | None = None
    runtime: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        d["efficiency"] = self.efficiency.to_dict() if self.efficiency else None
        return d

    def to_row(self) -> dict:
        """Flatten to the single leaderboard row shape: quality axis + cost axis."""
        m, e = self.metrics, self.efficiency
        row = {
            "scorer": self.scorer,
            "dataset": self.dataset,
            "split": self.split,
            "n": m.n,
            "balanced_acc": m.balanced_acc,
            "f1_notsup": m.f1_notsup,
            "pr_auc_notsup": m.pr_auc_notsup,
            "ece": m.ece,
        }
        if e:
            row.update(
                {
                    "params_m": e.params_m,
                    "size_mb": e.size_mb,
                    "device": e.device_label,
                    "latency_ms_p50": e.latency_ms_p50,
                    "throughput_qps": e.throughput_qps,
                    "cost_per_1k_usd": e.cost_per_1k_usd,
                    "hosted": e.hosted,
                }
            )
        return row


def run(
    scorer,
    examples: list[Example],
    dataset_name: str,
    split: str,
    benchmark_device: BenchmarkDevice | None = None,
    benchmark_n: int = 50,
    n_bins: int = 10,
) -> RunResult:
    """Score every example, then compute quality and (optionally) cost.

    Efficiency is measured on a separate small slice rather than by timing the scoring
    pass. The scoring pass runs at whatever batch size is fastest, which is not the
    request-path latency anyone deploys against, and it includes lazy model loading.
    """
    if not examples:
        raise ValueError("cannot run over zero examples")

    verdicts = scorer.score_batch(examples)
    if len(verdicts) != len(examples):
        raise ValueError(
            f"{getattr(scorer, 'name', scorer)} returned {len(verdicts)} verdicts "
            f"for {len(examples)} examples"
        )

    y_true = np.array([e.supported for e in examples], dtype=bool)
    p_supported = np.array([v.score for v in verdicts], dtype=float)
    quality = metrics_mod.compute(y_true, p_supported, n_bins=n_bins)

    efficiency = None
    if benchmark_device is not None:
        efficiency = measure_efficiency(
            scorer, examples[:benchmark_n], benchmark_device=benchmark_device
        )

    coarse = sum(1 for e in examples if e.meta.get("label3_source") == "coarse")
    upstream_corpora_inside_this_eval_set = sorted(
        {
            str(e.meta.get("source_dataset") or e.meta.get("dataset") or dataset_name)
            for e in examples
        }
    )

    return RunResult(
        scorer=getattr(scorer, "name", type(scorer).__name__),
        dataset=dataset_name,
        split=split,
        metrics=quality,
        efficiency=efficiency,
        runtime=describe_runtime(),
        notes={
            "supported_rate": round(float(y_true.mean()), 4),
            "coarse_label3_rate": round(coarse / len(examples), 4),
            "corpora_the_scorer_trained_on": list(getattr(scorer, "training_corpora", ())),
            "upstream_corpora_inside_this_eval_set": upstream_corpora_inside_this_eval_set,
        },
    )
