"""groundcontrol: a calibrated groundedness scorer and pluggable eval harness."""

from groundcontrol.data.base import Dataset, Example, is_supported
from groundcontrol.scorers.base import EfficiencyProfile, Scorer, Verdict

__version__ = "0.0.1"

__all__ = [
    "Dataset",
    "EfficiencyProfile",
    "Example",
    "Scorer",
    "Verdict",
    "is_supported",
]
