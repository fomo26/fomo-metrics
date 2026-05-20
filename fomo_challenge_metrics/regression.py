"""Regression metrics."""

import numpy as np
from scipy.stats import pearsonr
from .utils import _is_invalid


def compute_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """Compute mean absolute error. None predictions are penalized with 100.0."""
    y_pred_clean = [100.0 if _is_invalid(pred) else float(pred) for pred in y_pred]
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred_clean))))


def compute_correlation(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Pearson correlation. None predictions are penalized with -100.0."""
    y_pred_clean = [-100.0 if _is_invalid(pred) else float(pred) for pred in y_pred]
    correlation, _ = pearsonr(y_true, y_pred_clean)
    return float(correlation)
