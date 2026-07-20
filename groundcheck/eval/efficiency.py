"""The cost axis.

Accuracy alone cannot decide a deployment. A scorer that wins by two points of balanced
accuracy and costs fifty times more per call loses in the request path, which is where
this is meant to run. So every leaderboard row carries footprint and latency next to
quality.

Latency is only meaningful attached to the machine that produced it, so `measure`
requires an explicit `BenchmarkDevice`. See `groundcheck/device.py` for why that is not
inferred.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

from groundcheck.data.base import Example
from groundcheck.device import BenchmarkDevice
from groundcheck.scorers.base import EfficiencyProfile


@dataclass(slots=True)
class Efficiency:
    device: str
    device_label: str
    batch_size: int
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_qps: float
    hosted: bool = False
    params_m: float | None = None
    size_mb: float | None = None
    cost_per_1k_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def measure(
    scorer,
    items: list[Example],
    benchmark_device: BenchmarkDevice,
    batch_size: int = 1,
    warmup: int = 2,
    repeats: int = 1,
) -> Efficiency:
    """Time a scorer over `items`, batch by batch.

    Warmup batches are discarded: the first calls pay for lazy weight loading, kernel
    compilation, and cache population, none of which a steady-state service pays.

    `batch_size=1` is the default because the headline latency number should describe
    the request path, where one answer is scored at a time. Throughput at larger batch
    sizes is a separate question, measured by passing one.
    """
    if not items:
        raise ValueError("cannot measure efficiency over zero examples")

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    for batch in batches[:warmup]:
        scorer.score_batch(batch)

    timings_ms: list[float] = []
    scored = 0
    wall_start = time.perf_counter()
    for _ in range(repeats):
        for batch in batches:
            start = time.perf_counter()
            scorer.score_batch(batch)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
            scored += len(batch)
    wall = time.perf_counter() - wall_start

    profile: EfficiencyProfile = (
        scorer.efficiency() if hasattr(scorer, "efficiency") else EfficiencyProfile()
    )

    return Efficiency(
        device=benchmark_device.device,
        device_label=benchmark_device.label,
        batch_size=batch_size,
        latency_ms_p50=round(statistics.median(timings_ms), 2),
        latency_ms_p95=round(_percentile(timings_ms, 0.95), 2),
        throughput_qps=round(scored / wall, 2) if wall > 0 else float("inf"),
        hosted=profile.hosted,
        params_m=profile.params_m,
        size_mb=profile.size_mb,
        cost_per_1k_usd=profile.cost_per_1k_usd,
    )
