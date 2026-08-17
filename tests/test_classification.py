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

# ---------------------------------------------------------------------------
# compute_ovr_auroc on class subsets and binary tasks
#
# Both of these raised before. compute_max_disparity catches every exception
# and drops the group, so the failure was invisible: a variable whose disparity
# was never measured could still contribute a perfect 1.0 to the fairness score.
# ---------------------------------------------------------------------------

def test_ovr_auroc_scores_a_group_missing_a_class():
    """A demographic group that happens to contain only 2 of 3 classes is
    common in small bins, and is not a reason to skip it."""
    y_true = [0, 1] * 6
    y_scores = [[0.7, 0.2, 0.1] if label == 0 else [0.2, 0.7, 0.1] for label in y_true]
    assert compute_ovr_auroc(y_true, y_scores) == pytest.approx(1.0)


def test_ovr_auroc_handles_a_binary_task():
    """sklearn wants a 1-D score array when there are two classes."""
    y_true = [0, 1] * 10
    y_scores = [[0.8, 0.2] if label == 0 else [0.2, 0.8] for label in y_true]
    assert compute_ovr_auroc(y_true, y_scores) == pytest.approx(1.0)

    inverted = [[0.2, 0.8] if label == 0 else [0.8, 0.2] for label in y_true]
    assert compute_ovr_auroc(y_true, inverted) == pytest.approx(0.0)


def test_ovr_auroc_unchanged_when_every_class_is_present(multiclass_perfect):
    """The fix must not move any score that already computed."""
    y_true, y_scores = multiclass_perfect
    assert compute_ovr_auroc(y_true, y_scores) == pytest.approx(1.0)


def test_ovr_auroc_matches_sklearn_on_the_full_case():
    """Agreement with the previous implementation, on data where it worked."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y_true = [0, 1, 2] * 10
    scores = rng.random((30, 3))
    scores /= scores.sum(axis=1, keepdims=True)
    expected = roc_auc_score(y_true, scores, multi_class="ovr", average="macro")
    assert compute_ovr_auroc(y_true, scores.tolist()) == pytest.approx(expected)


def test_ovr_auroc_is_nan_when_a_group_has_one_class():
    """Genuinely undefined, so NaN is right and compute_max_disparity should
    still drop it."""
    import math

    assert math.isnan(compute_ovr_auroc([1] * 8, [[0.2, 0.7, 0.1]] * 8))
    assert math.isnan(compute_ovr_auroc([0] * 8, [[0.7, 0.3]] * 8))


def test_ovr_auroc_still_penalises_invalid_rows():
    """The None-row replacement is orthogonal to this fix and must survive."""
    y_true = [0, 1] * 6
    good = [[0.9, 0.1] if label == 0 else [0.1, 0.9] for label in y_true]
    assert compute_ovr_auroc(y_true, good) == pytest.approx(1.0)

    with_invalid = list(good)
    with_invalid[0] = None
    assert compute_ovr_auroc(y_true, with_invalid) < 1.0


def test_ovr_auroc_handles_an_invalid_row_in_first_position():
    """The docstring promises None rows are replaced, but the class count was
    read as len(y_scores[0]), which raises when that row is the None."""
    y_true = [0, 1] * 6
    scores = [[0.9, 0.1] if label == 0 else [0.1, 0.9] for label in y_true]
    scores[0] = None
    assert compute_ovr_auroc(y_true, scores) < 1.0
