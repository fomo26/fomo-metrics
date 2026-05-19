"""Tests for segmentation metrics."""

import numpy as np
import pytest

from fomo_challenge_metrics import (
    MetricsConfig,
    compute_dice_coefficient,
    compute_multiclass_dsc,
    compute_multiclass_nsd,
    compute_normalized_surface_distance,
)

SPACING = (1.0, 1.0, 1.5)
SHAPE = (32, 32, 16)


@pytest.fixture
def perfect_binary():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[8:24, 8:24, 4:12] = True
    return mask, mask.copy()


@pytest.fixture
def disjoint_binary():
    pred = np.zeros(SHAPE, dtype=bool)
    gt = np.zeros(SHAPE, dtype=bool)
    pred[0:8, 0:8, 0:4] = True
    gt[24:32, 24:32, 12:16] = True
    return pred, gt


@pytest.fixture
def multiclass_masks():
    gt = np.zeros(SHAPE, dtype=np.int32)
    gt[4:16, 4:16, 2:8] = 1
    gt[16:28, 16:28, 8:14] = 2
    pred = gt.copy()
    return pred, gt


class TestComputeDiceCoefficient:
    def test_perfect_overlap(self, perfect_binary):
        pred, gt = perfect_binary
        assert compute_dice_coefficient(pred, gt) == pytest.approx(1.0)

    def test_no_overlap(self, disjoint_binary):
        pred, gt = disjoint_binary
        assert compute_dice_coefficient(pred, gt) == MetricsConfig.DSC_WORST

    def test_both_empty(self):
        empty = np.zeros(SHAPE, dtype=bool)
        assert compute_dice_coefficient(empty, empty) == pytest.approx(1.0)

    def test_pred_empty(self, perfect_binary):
        _, gt = perfect_binary
        empty = np.zeros(SHAPE, dtype=bool)
        assert compute_dice_coefficient(empty, gt) == MetricsConfig.DSC_WORST

    def test_gt_empty(self, perfect_binary):
        pred, _ = perfect_binary
        empty = np.zeros(SHAPE, dtype=bool)
        assert compute_dice_coefficient(pred, empty) == MetricsConfig.DSC_WORST

    def test_partial_overlap(self):
        pred = np.zeros(SHAPE, dtype=bool)
        gt = np.zeros(SHAPE, dtype=bool)
        pred[8:24, 8:24, 4:12] = True
        gt[8:24, 8:16, 4:12] = True
        result = compute_dice_coefficient(pred, gt)
        assert 0.0 < result < 1.0


class TestComputeNormalizedSurfaceDistance:
    def test_perfect_overlap(self, perfect_binary):
        pred, gt = perfect_binary
        result = compute_normalized_surface_distance(pred, gt, SPACING)
        assert result == pytest.approx(1.0)

    def test_no_overlap(self, disjoint_binary):
        pred, gt = disjoint_binary
        assert compute_normalized_surface_distance(pred, gt, SPACING) == MetricsConfig.NSD_WORST

    def test_both_empty(self):
        empty = np.zeros(SHAPE, dtype=bool)
        assert compute_normalized_surface_distance(empty, empty, SPACING) == pytest.approx(0.0)

    def test_pred_empty(self, perfect_binary):
        _, gt = perfect_binary
        empty = np.zeros(SHAPE, dtype=bool)
        assert compute_normalized_surface_distance(empty, gt, SPACING) == MetricsConfig.NSD_WORST

    def test_gt_empty(self, perfect_binary):
        pred, _ = perfect_binary
        empty = np.zeros(SHAPE, dtype=bool)
        assert compute_normalized_surface_distance(pred, empty, SPACING) == MetricsConfig.NSD_WORST


class TestComputeMulticlassDsc:
    def test_perfect(self, multiclass_masks):
        pred, gt = multiclass_masks
        result = compute_multiclass_dsc(pred, gt)
        assert set(result.keys()) == {1, 2}
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)

    def test_excludes_background(self, multiclass_masks):
        pred, gt = multiclass_masks
        result = compute_multiclass_dsc(pred, gt)
        assert 0 not in result

    def test_explicit_labels(self, multiclass_masks):
        pred, gt = multiclass_masks
        result = compute_multiclass_dsc(pred, gt, labels=[1])
        assert set(result.keys()) == {1}

    def test_all_wrong(self):
        gt = np.zeros(SHAPE, dtype=np.int32)
        gt[4:16, 4:16, 2:8] = 1
        pred = np.zeros(SHAPE, dtype=np.int32)
        pred[16:28, 16:28, 8:14] = 1
        result = compute_multiclass_dsc(pred, gt)
        assert result[1] == MetricsConfig.DSC_WORST

    def test_empty_gt_returns_empty(self):
        empty = np.zeros(SHAPE, dtype=np.int32)
        result = compute_multiclass_dsc(empty, empty)
        assert result == {}

    def test_returns_worst_on_known_labels_failure(self):
        result = compute_multiclass_dsc(None, None, labels=[1, 2])  # type: ignore[arg-type]
        assert result == {1: MetricsConfig.DSC_WORST, 2: MetricsConfig.DSC_WORST}


class TestComputeMulticlassNsd:
    def test_perfect(self, multiclass_masks):
        pred, gt = multiclass_masks
        result = compute_multiclass_nsd(pred, gt, SPACING)
        assert set(result.keys()) == {1, 2}
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)

    def test_excludes_background(self, multiclass_masks):
        pred, gt = multiclass_masks
        result = compute_multiclass_nsd(pred, gt, SPACING)
        assert 0 not in result

    def test_all_wrong(self):
        gt = np.zeros(SHAPE, dtype=np.int32)
        gt[4:16, 4:16, 2:8] = 1
        pred = np.zeros(SHAPE, dtype=np.int32)
        pred[16:28, 16:28, 8:14] = 1
        result = compute_multiclass_nsd(pred, gt, SPACING)
        assert result[1] == MetricsConfig.NSD_WORST

    def test_empty_gt_returns_empty(self):
        empty = np.zeros(SHAPE, dtype=np.int32)
        result = compute_multiclass_nsd(empty, empty, SPACING)
        assert result == {}

    def test_returns_worst_on_known_labels_failure(self):
        result = compute_multiclass_nsd(None, None, SPACING, labels=[1, 2])  # type: ignore[arg-type]
        assert result == {1: MetricsConfig.NSD_WORST, 2: MetricsConfig.NSD_WORST}
