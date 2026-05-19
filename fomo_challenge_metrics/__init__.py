"""challenges_metrics — callable metric functions for the FOMO26 Challenge."""

from ._config import MetricsConfig
from .classification import compute_auroc, compute_macro_f1, compute_ovr_auroc
from .fairness import compute_fairness_gap, compute_per_group_metric
from .regression import compute_absolute_error, compute_correlation
from .segmentation import (
    compute_dice_coefficient,
    compute_multiclass_dsc,
    compute_multiclass_nsd,
    compute_normalized_surface_distance,
)

__all__ = [
    "MetricsConfig",
    "compute_auroc",
    "compute_ovr_auroc",
    "compute_macro_f1",
    "compute_absolute_error",
    "compute_correlation",
    "compute_dice_coefficient",
    "compute_normalized_surface_distance",
    "compute_multiclass_dsc",
    "compute_multiclass_nsd",
    "compute_per_group_metric",
    "compute_fairness_gap",
]
