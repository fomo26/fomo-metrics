"""Tests for regression metrics."""

import math

import pytest

from fomo_challenge_metrics import MetricsConfig, compute_absolute_error, compute_correlation


class TestComputeAbsoluteError:
    def test_perfect(self):
        assert compute_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_known_mae(self):
        assert compute_absolute_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)

    def test_empty(self):
        assert compute_absolute_error([], []) == MetricsConfig.MAE_WORST

    def test_mismatched_lengths(self):
        assert compute_absolute_error([1.0, 2.0], [1.0]) == MetricsConfig.MAE_WORST

    def test_mae_worst_is_inf(self):
        assert math.isinf(MetricsConfig.MAE_WORST)

    def test_single_sample(self):
        assert compute_absolute_error([5.0], [3.0]) == pytest.approx(2.0)


class TestComputeCorrelation:
    def test_perfect_positive(self):
        assert compute_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert compute_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_no_correlation(self):
        result = compute_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0])
        assert result == MetricsConfig.CORR_WORST

    def test_empty(self):
        assert compute_correlation([], []) == MetricsConfig.CORR_WORST

    def test_single_sample(self):
        assert compute_correlation([1.0], [1.0]) == MetricsConfig.CORR_WORST

    def test_mismatched_lengths(self):
        assert compute_correlation([1.0, 2.0], [1.0]) == MetricsConfig.CORR_WORST

    def test_two_samples(self):
        result = compute_correlation([1.0, 2.0], [1.0, 2.0])
        assert result == pytest.approx(1.0)
