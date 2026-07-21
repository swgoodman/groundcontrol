"""Device selection.

Two devices, deliberately distinct:

* The **compute device** is wherever training or bulk scoring happens to run. It is a
  convenience: cuda > mps > cpu, whatever is fastest here. Quality metrics must not
  depend on it, so every report records which one was used.

* The **benchmark device** is what latency and throughput are measured on, and it is
  never inferred. The efficiency thesis ("small enough to run on commodity CPU") is
  only credible if the number comes from a machine the reader would actually deploy
  on, so the caller states it explicitly and the leaderboard row carries the label.

Measuring latency on an Apple M4 Pro and reporting it as "CPU latency" would flatter
the result against the x86 vCPU the target audience runs. Hence the asymmetry.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


def resolve_compute_device(prefer: str | None = None) -> str:
    """Best available device for training / bulk inference."""
    if prefer:
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(slots=True, frozen=True)
class BenchmarkDevice:
    """An explicitly-declared machine that a latency number was measured on."""

    device: str
    """Torch device string: "cpu", "cuda", "mps"."""

    label: str
    """Human-readable chip / instance, e.g. "Apple M4 Pro" or "AWS c7i.xlarge"."""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError(
                "benchmark device requires an explicit label; an unlabeled latency "
                "number is not comparable across machines"
            )

    @classmethod
    def local_cpu(cls) -> BenchmarkDevice:
        """This machine's CPU, auto-labeled. Convenient for dev, not for publication."""
        return cls(device="cpu", label=f"{platform.processor() or platform.machine()} (local)")


def describe_runtime() -> dict[str, str]:
    """Provenance recorded into every report."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compute_device": resolve_compute_device(),
    }
