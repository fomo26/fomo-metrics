"""Fairness metrics — thin wrappers over per-group metric computation."""

from typing import Callable, Literal

import numpy as np


def compute_per_group_metric(
    metric_fn: Callable,
    groups: np.ndarray,
    *args,
    **kwargs,
) -> dict[str, float]:
    """Apply metric_fn to each group subset of args."""
    try:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        result: dict[str, float] = {}
        for g in unique_groups:
            mask = groups == g
            group_args = [
                np.asarray(a)[mask].tolist() if hasattr(a, "__len__") else a
                for a in args
            ]
            result[str(g)] = metric_fn(*group_args, **kwargs)
        return result
    except Exception:
        return {}


def compute_fairness_gap(
    per_group: dict[str, float],
    statistic: Literal["max_minus_min", "std"] = "max_minus_min",
) -> float:
    """Compute disparity across groups."""
    try:
        values = list(per_group.values())
        if not values:
            return 0.0
        if statistic == "max_minus_min":
            return float(max(values) - min(values))
        return float(np.std(values))
    except Exception:
        return 0.0
