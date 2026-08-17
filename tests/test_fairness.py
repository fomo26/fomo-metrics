"""Tests for fairness metrics."""
import math
import pytest
from fomo_challenge_metrics import (
    compute_max_disparity,
    compute_fairness_score,
    compute_ovr_f1,
    compute_ovr_auroc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_3class():
    """Perfect OvR scores for 6 samples, 3 classes."""
    y_true  = [0, 0, 1, 1, 2, 2]
    y_scores = [
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.05, 0.05, 0.9],
    ]
    return y_true, y_scores


@pytest.fixture
def balanced_groups(perfect_3class):
    """Groups balanced across two groups: alternating A / B."""
    y_true, y_scores = perfect_3class
    groups = ["A", "B", "A", "B", "A", "B"]
    return y_true, y_scores, groups


# ---------------------------------------------------------------------------
# compute_max_disparity
# ---------------------------------------------------------------------------

class TestComputeMaxDisparity:
    def test_zero_disparity_when_groups_equal(self, balanced_groups):
        y_true, y_scores, groups = balanced_groups
        d = compute_max_disparity(y_true, y_scores, groups, compute_ovr_f1)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_disparity_when_groups_differ(self):
        y_true   = [0, 1, 0, 1]
        y_scores = [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ]
        groups = ["A", "A", "B", "B"]
        d = compute_max_disparity(y_true, y_scores, groups, compute_ovr_f1)
        assert d > 0.0

    def test_single_group_returns_zero(self):
        y_true   = [0, 1, 2]
        y_scores = [[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]
        groups   = ["A", "A", "A"]
        d = compute_max_disparity(y_true, y_scores, groups, compute_ovr_f1)
        assert d == 0.0

    def test_all_none_groups_returns_nan(self):
        y_true   = [0, 1]
        y_scores = [[0.9, 0.1], [0.1, 0.9]]
        groups   = [None, None]
        d = compute_max_disparity(y_true, y_scores, groups, compute_ovr_f1)
        assert math.isnan(d)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_max_disparity([0, 1], [[0.9, 0.1]], ["A", "A"], compute_ovr_f1)

    def test_returns_float(self, balanced_groups):
        y_true, y_scores, groups = balanced_groups
        d = compute_max_disparity(y_true, y_scores, groups, compute_ovr_f1)
        assert isinstance(d, float)

    def test_disparity_in_unit_interval_for_bounded_metrics(self):
        y_true   = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_scores_A = [
            [0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9],
            [0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9],
        ]
        y_scores_B = [
            [0.05,0.9,0.05],[0.05,0.05,0.9],[0.9,0.05,0.05],
            [0.05,0.9,0.05],[0.05,0.05,0.9],[0.9,0.05,0.05],
        ]
        y_scores = y_scores_A + y_scores_B
        groups   = ["A"]*6 + ["B"]*6
        d = compute_max_disparity(y_true, y_scores, groups, compute_ovr_auroc)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# compute_fairness_score
# ---------------------------------------------------------------------------

class TestComputeFairnessScore:
    def test_perfect_equity_score_is_one(self, balanced_groups):
        y_true, y_scores, groups = balanced_groups
        result = compute_fairness_score(
            y_true, y_scores, {"grp": groups}, compute_ovr_f1
        )
        assert result["score"] == pytest.approx(1.0, abs=1e-6)

    def test_score_between_zero_and_one(self):
        y_true   = [0, 1, 0, 1]
        y_scores = [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ]
        groups = ["A","A","B","B"]
        result = compute_fairness_score(
            y_true, y_scores, {"grp": groups}, compute_ovr_f1
        )
        assert 0.0 <= result["score"] <= 1.0

    def test_variables_used_populated(self, balanced_groups):
        y_true, y_scores, groups = balanced_groups
        result = compute_fairness_score(
            y_true, y_scores, {"var_a": groups, "var_b": groups}, compute_ovr_f1
        )
        assert "var_a" in result["variables_used"]
        assert "var_b" in result["variables_used"]

    def test_mismatched_group_length_raises(self, balanced_groups):
        y_true, y_scores, _ = balanced_groups
        with pytest.raises(ValueError):
            compute_fairness_score(y_true, y_scores, {"bad": ["A"]}, compute_ovr_f1)


def test_disparity_is_measured_when_a_group_misses_a_class():
    """Regression for the silent path: group B contains no class 2, which used
    to raise inside compute_ovr_auroc, get swallowed, and leave a single valid
    group. compute_max_disparity then returned 0.0 by contract and the variable
    contributed a perfect 1.0 to the fairness score without ever having been
    evaluated."""
    y_true = [0, 1, 2] * 10 + [0, 1] * 6
    groups = ["A"] * 30 + ["B"] * 12
    scores = [
        [0.7, 0.2, 0.1] if label == 0 else [0.2, 0.7, 0.1] if label == 1
        else [0.1, 0.2, 0.7]
        for label in y_true
    ]

    result = compute_fairness_score(
        y_true, scores, {"site": groups}, compute_ovr_auroc
    )
    assert result["variables_used"] == ["site"]
    assert not math.isnan(result["disparities"]["site"])


def test_fairness_score_is_defined_for_a_binary_task():
    """Every group raised, so the score came back NaN with no variables used."""
    y_true = [0, 1] * 20
    groups = ["A"] * 20 + ["B"] * 20
    scores = [[0.8, 0.2] if label == 0 else [0.2, 0.8] for label in y_true]

    result = compute_fairness_score(
        y_true, scores, {"site": groups}, compute_ovr_auroc
    )
    assert not math.isnan(result["score"])
    assert result["variables_used"] == ["site"]
