"""Tests for classification metrics."""
import pytest
from fomo_challenge_metrics import compute_auroc, compute_ovr_auroc, compute_macro_f1, compute_ovr_f1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_perfect():
    """Scores that rank all positives above all negatives."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    return y_true, y_scores


@pytest.fixture
def binary_worst():
    """Scores that rank all positives below all negatives."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    return y_true, y_scores


@pytest.fixture
def multiclass_perfect():
    """OvR scores where argmax always matches the true label."""
    y_true = [0, 1, 2]
    y_scores = [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]
    return y_true, y_scores


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------

class TestComputeAUROC:
    def test_perfect_score(self, binary_perfect):
        y_true, y_scores = binary_perfect
        assert compute_auroc(y_true, y_scores) == pytest.approx(1.0)

    def test_worst_score(self, binary_worst):
        y_true, y_scores = binary_worst
        assert compute_auroc(y_true, y_scores) == pytest.approx(0.0)

    def test_random_score(self):
        y_true = [0, 1, 0, 1]
        y_scores = [0.4, 0.6, 0.4, 0.6]
        result = compute_auroc(y_true, y_scores)
        assert 0.0 <= result <= 1.0

    def test_none_on_positive_replaced_with_worst(self):
        # None on a positive label -> score 0.0 (worst for positives), AUROC degrades
        y_true   = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.9, None]
        result_with_none = compute_auroc(y_true, y_scores)
        result_clean     = compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
        assert result_with_none < result_clean

    def test_none_on_negative_replaced_with_worst(self):
        # None on a negative label -> score 1.0 (worst for negatives), AUROC degrades
        y_true   = [0, 0, 1, 1]
        y_scores = [0.1, None, 0.8, 0.9]
        result_with_none = compute_auroc(y_true, y_scores)
        result_clean     = compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert result_with_none < result_clean

    def test_returns_float(self, binary_perfect):
        y_true, y_scores = binary_perfect
        assert isinstance(compute_auroc(y_true, y_scores), float)


# ---------------------------------------------------------------------------
# compute_ovr_auroc
# ---------------------------------------------------------------------------

class TestComputeOVRAUROC:
    def test_perfect_score(self, multiclass_perfect):
        y_true, y_scores = multiclass_perfect
        assert compute_ovr_auroc(y_true, y_scores) == pytest.approx(1.0)

    def test_result_in_unit_interval(self):
        y_true   = [0, 1, 2, 0, 1, 2]
        y_scores = [
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.4, 0.4, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ]
        result = compute_ovr_auroc(y_true, y_scores)
        assert 0.0 <= result <= 1.0

    def test_none_row_replaced_with_worst(self):
        y_true = [0, 1, 2]
        y_scores_clean = [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
        y_scores_dirty = [
            [0.9, 0.05, 0.05],
            None,               # row for label=1 is missing
            [0.05, 0.05, 0.9],
        ]
        result_clean = compute_ovr_auroc(y_true, y_scores_clean)
        result_dirty = compute_ovr_auroc(y_true, y_scores_dirty)
        assert result_dirty < result_clean

    def test_returns_float(self, multiclass_perfect):
        y_true, y_scores = multiclass_perfect
        assert isinstance(compute_ovr_auroc(y_true, y_scores), float)


# ---------------------------------------------------------------------------
# compute_macro_f1
# ---------------------------------------------------------------------------

class TestComputeMacroF1:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        assert compute_macro_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        assert compute_macro_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_none_on_class0_replaced_with_1(self):
        # label=0, None -> pred=1 (wrong), so F1 degrades
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, None, 1]
        result_with_none = compute_macro_f1(y_true, y_pred)
        result_clean     = compute_macro_f1([0, 1, 0, 1], [0, 1, 0, 1])
        assert result_with_none < result_clean

    def test_none_on_nonzero_class_replaced_with_0(self):
        # label=1, None -> pred=0 (wrong)
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, None]
        result_with_none = compute_macro_f1(y_true, y_pred)
        result_clean     = compute_macro_f1([0, 1, 0, 1], [0, 1, 0, 1])
        assert result_with_none < result_clean

    def test_returns_float(self):
        y_true = [0, 1]
        y_pred = [0, 1]
        assert isinstance(compute_macro_f1(y_true, y_pred), float)

    def test_zero_division_returns_zero(self):
        # All predictions wrong -> some classes have no TP/FP; zero_division=0
        y_true = [0, 0]
        y_pred = [1, 1]
        result = compute_macro_f1(y_true, y_pred)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_ovr_f1
# ---------------------------------------------------------------------------

class TestComputeOVRF1:
    def test_perfect_predictions(self, multiclass_perfect):
        y_true, y_scores = multiclass_perfect
        assert compute_ovr_f1(y_true, y_scores) == pytest.approx(1.0)

    def test_result_in_unit_interval(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_scores = [
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ]
        result = compute_ovr_f1(y_true, y_scores)
        assert 0.0 <= result <= 1.0

    def test_argmax_used_for_prediction(self):
        y_true   = [0, 1, 2]
        y_scores = [
            [0.9, 0.05, 0.05],
            [0.05, 0.8, 0.15],  # argmax=1, correct
            [0.05, 0.05, 0.9],
        ]
        assert compute_ovr_f1(y_true, y_scores) == pytest.approx(1.0)

    def test_none_row_replaced_with_worst(self):
        y_true = [0, 1, 2]
        y_scores_clean = [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
        y_scores_dirty = [
            [0.9, 0.05, 0.05],
            None,
            [0.05, 0.05, 0.9],
        ]
        result_clean = compute_ovr_f1(y_true, y_scores_clean)
        result_dirty = compute_ovr_f1(y_true, y_scores_dirty)
        assert result_dirty < result_clean

    def test_worst_prediction_is_always_wrong(self):
        # Valid row needed to infer n_classes=3, but argmax=1 != label=0 -> also wrong
        # None rows: label=1 -> pred=2, label=2 -> pred=0 -> both wrong
        y_true   = [0, 1, 2]
        y_scores = [[0.05, 0.9, 0.05], None, None]  # argmax=1, label=0 -> wrong
        result = compute_ovr_f1(y_true, y_scores)
        assert result == pytest.approx(0.0)

    def test_returns_float(self, multiclass_perfect):
        y_true, y_scores = multiclass_perfect
        assert isinstance(compute_ovr_f1(y_true, y_scores), float)