"""Tests for classification metrics."""

import pytest

from fomo_challenge_metrics import MetricsConfig, compute_auroc, compute_macro_f1, compute_ovr_auroc


class TestComputeAuroc:
    def test_perfect(self):
        assert compute_auroc([0, 1], [0.0, 1.0]) == pytest.approx(1.0)

    def test_all_wrong(self):
        assert compute_auroc([0, 1], [1.0, 0.0]) == pytest.approx(0.0)

    def test_random(self):
        result = compute_auroc([0, 0, 1, 1], [0.4, 0.6, 0.4, 0.6])
        assert 0.0 <= result <= 1.0

    def test_empty(self):
        assert compute_auroc([], []) == MetricsConfig.AUROC_WORST

    def test_mismatched_lengths(self):
        assert compute_auroc([0, 1], [0.5]) == MetricsConfig.AUROC_WORST

    def test_single_class_labels(self):
        assert compute_auroc([1, 1, 1], [0.5, 0.6, 0.7]) == MetricsConfig.AUROC_WORST

    def test_single_sample(self):
        assert compute_auroc([0], [0.5]) == MetricsConfig.AUROC_WORST


class TestComputeOvrAuroc:
    def test_perfect_binary(self):
        scores = [[1.0, 0.0], [0.0, 1.0]]
        assert compute_ovr_auroc([0, 1], scores) == pytest.approx(1.0)

    def test_perfect_multiclass(self):
        scores = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert compute_ovr_auroc([0, 1, 2], scores) == pytest.approx(1.0)

    def test_empty(self):
        assert compute_ovr_auroc([], []) == MetricsConfig.AUROC_WORST

    def test_mismatched_lengths(self):
        assert compute_ovr_auroc([0, 1], [[0.5, 0.5]]) == MetricsConfig.AUROC_WORST

    def test_single_class_labels(self):
        scores = [[1.0, 0.0], [0.9, 0.1]]
        assert compute_ovr_auroc([0, 0], scores) == MetricsConfig.AUROC_WORST

    def test_all_wrong(self):
        scores = [[0.0, 1.0], [1.0, 0.0]]
        assert compute_ovr_auroc([0, 1], scores) == pytest.approx(0.0)


class TestComputeMacroF1:
    def test_perfect(self):
        assert compute_macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_all_wrong(self):
        result = compute_macro_f1([0, 1], [1, 0])
        assert result == pytest.approx(0.0)

    def test_empty(self):
        assert compute_macro_f1([], []) == MetricsConfig.F1_WORST

    def test_mismatched_lengths(self):
        assert compute_macro_f1([0, 1], [0]) == MetricsConfig.F1_WORST

    def test_single_sample_correct(self):
        assert compute_macro_f1([1], [1]) == pytest.approx(1.0)

    def test_binary_perfect(self):
        assert compute_macro_f1([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
