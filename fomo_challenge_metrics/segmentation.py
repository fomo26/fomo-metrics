"""Segmentation metrics using google-deepmind/surface-distance."""

import numpy as np
import surface_distance as surfdist

from ._config import MetricsConfig


def compute_dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between prediction and ground truth masks."""
    try:
        pred_binary = pred_mask.astype(bool)
        gt_binary = gt_mask.astype(bool)

        pred_sum = np.sum(pred_binary)
        gt_sum = np.sum(gt_binary)

        if pred_sum == 0 and gt_sum == 0:
            return 1.0

        if pred_sum == 0 or gt_sum == 0:
            return MetricsConfig.DSC_WORST

        intersection = np.sum(pred_binary & gt_binary)
        return float(2.0 * intersection / (pred_sum + gt_sum))
    except Exception:
        return MetricsConfig.DSC_WORST


def compute_normalized_surface_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> float:
    """Compute Normalized Surface Distance (surface Dice at tolerance)."""
    try:
        pred_binary = pred_mask.astype(bool)
        gt_binary = gt_mask.astype(bool)

        if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
            return 0.0

        if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
            return MetricsConfig.NSD_WORST

        surface_distances = surfdist.compute_surface_distances(gt_binary, pred_binary, spacing_mm)
        nsd_score = surfdist.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)

        if np.isnan(nsd_score) or np.isinf(nsd_score):
            return MetricsConfig.NSD_WORST

        return float(nsd_score)
    except Exception:
        return MetricsConfig.NSD_WORST


def compute_multiclass_dsc(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    labels: list[int] | None = None,
) -> dict[int, float]:
    """Compute Dice coefficient per class. Label 0 always excluded."""
    _labels = labels
    try:
        if _labels is None:
            _labels = [int(l) for l in np.unique(gt_mask) if l != 0]
        result = {}
        for label in _labels:
            result[label] = compute_dice_coefficient(pred_mask == label, gt_mask == label)
        return result
    except Exception:
        if _labels is None:
            return {}
        return {int(l): MetricsConfig.DSC_WORST for l in _labels}


def compute_multiclass_nsd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    tolerance_mm: float = 1.0,
    labels: list[int] | None = None,
) -> dict[int, float]:
    """Compute NSD per class. Label 0 always excluded."""
    _labels = labels
    try:
        if _labels is None:
            _labels = [int(l) for l in np.unique(gt_mask) if l != 0]
        result = {}
        for label in _labels:
            result[label] = compute_normalized_surface_distance(
                pred_mask == label, gt_mask == label, spacing_mm, tolerance_mm
            )
        return result
    except Exception:
        if _labels is None:
            return {}
        return {int(l): MetricsConfig.NSD_WORST for l in _labels}
