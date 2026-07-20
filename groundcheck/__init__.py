"""groundcheck: a calibrated groundedness scorer and pluggable eval harness."""

from groundcheck.data.base import Dataset, Example, is_supported
from groundcheck.scorers.base import EfficiencyProfile, Scorer, Verdict

__version__ = "0.0.1"

__all__ = [
    "Dataset",
    "EfficiencyProfile",
    "Example",
    "Scorer",
    "Verdict",
    "is_supported",
]
