"""Tests for fairness metrics."""

import numpy as np
import pytest

from fomo_challenge_metrics import compute_fairness_gap, compute_per_group_metric
from fomo_challenge_metrics.classification import compute_auroc


class TestComputePerGroupMetric:
    def test_two_groups_known_auroc(self):
        y_true = [0, 1, 0, 1]
        y_scores = [0.1, 0.9, 0.2, 0.8]
        groups = np.array(["A", "A", "B", "B"])
        result = compute_per_group_metric(compute_auroc, groups, y_true, y_scores)
        assert set(result.keys()) == {"A", "B"}
        assert result["A"] == pytest.approx(1.0)
        assert result["B"] == pytest.approx(1.0)

    def test_single_group(self):
        y_true = [0, 1]
        y_scores = [0.1, 0.9]
        groups = np.array(["A", "A"])
        result = compute_per_group_metric(compute_auroc, groups, y_true, y_scores)
        assert "A" in result
        assert result["A"] == pytest.approx(1.0)

    def test_group_with_single_class_returns_worst(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        groups = np.array(["A", "A", "B", "B"])
        result = compute_per_group_metric(compute_auroc, groups, y_true, y_scores)
        assert "A" in result
        assert "B" in result

    def test_invalid_inputs_return_empty(self):
        result = compute_per_group_metric(compute_auroc, None, [0, 1], [0.1, 0.9])  # type: ignore[arg-type]
        assert result == {}

    def test_numeric_group_keys(self):
        y_true = [0, 1, 0, 1]
        y_scores = [0.1, 0.9, 0.2, 0.8]
        groups = np.array([0, 0, 1, 1])
        result = compute_per_group_metric(compute_auroc, groups, y_true, y_scores)
        assert set(result.keys()) == {"0", "1"}


class TestComputeFairnessGap:
    def test_max_minus_min(self):
        per_group = {"A": 0.9, "B": 0.7, "C": 0.8}
        assert compute_fairness_gap(per_group) == pytest.approx(0.2)

    def test_std_statistic(self):
        per_group = {"A": 1.0, "B": 1.0}
        assert compute_fairness_gap(per_group, statistic="std") == pytest.approx(0.0)

    def test_single_group_max_minus_min(self):
        assert compute_fairness_gap({"A": 0.85}) == pytest.approx(0.0)

    def test_empty_dict(self):
        assert compute_fairness_gap({}) == pytest.approx(0.0)

    def test_perfect_fairness(self):
        per_group = {"A": 0.8, "B": 0.8}
        assert compute_fairness_gap(per_group) == pytest.approx(0.0)

    def test_std_with_spread(self):
        per_group = {"A": 0.6, "B": 1.0}
        gap = compute_fairness_gap(per_group, statistic="std")
        assert gap > 0.0
