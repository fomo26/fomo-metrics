"""Classification metrics."""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from .utils import _is_invalid


def compute_auroc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute AUROC. None scores are replaced with the worst possible score for that label."""
    y_scores_clean = []
    for label, score in zip(y_true, y_scores):
        if _is_invalid(score):
            y_scores_clean.append(0.0 if label == 1 else 1.0)
        else:
            y_scores_clean.append(float(score))
    return float(roc_auc_score(y_true, y_scores_clean))


def compute_ovr_auroc(y_true: list[int], y_scores: list[list[float]]) -> float:
    """Compute OvR macro AUROC. y_scores shape (N, C). None rows replaced with worst scores.

    The class set is taken from the width of y_scores rather than from the labels
    present in y_true, so a subset that happens to be missing a class still
    scores. `roc_auc_score(..., multi_class="ovr")` requires the columns of
    y_score to match the classes present in y_true, and wants a 1-D array when
    there are two, so calling it directly raised on two inputs that are fine:
    a per-group slice missing one class, and any binary task. Callers that
    evaluate within demographic groups hit both.

    Returns NaN when no class has both positives and negatives, which is
    genuinely undefined rather than an error.

    The class count is read from the first row that is not None, so a None in
    the first position is replaced like any other rather than raising.
    """
    # Width from the first row that has one. The docstring promises None rows
    # are replaced, but `len(y_scores[0])` raises when the first row is the None,
    # so the promise did not hold for that case. compute_ovr_f1 already reads
    # the width this way.
    n_classes = next((len(row) for row in y_scores if row is not None), 0)
    if n_classes < 2:
        return float("nan")

    y_scores_clean = []
    for label, row in zip(y_true, y_scores):
        if row is None or any(_is_invalid(s) for s in row):
            y_scores_clean.append(
                [0.0 if c == label else 1.0 / (n_classes - 1) for c in range(n_classes)]
            )
        else:
            y_scores_clean.append([float(s) for s in row])

    labels = list(range(n_classes))
    y = np.asarray(y_true)
    scores = np.asarray(y_scores_clean, dtype=float)

    if n_classes == 2:
        # Binary: sklearn takes the positive class column, not both.
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, scores[:, 1], labels=labels))

    indicator = label_binarize(y, classes=labels)
    # A class needs both positives and negatives for its one-vs-rest curve to
    # exist. Averaging over the ones that do is what macro OvR already means.
    usable = [c for c in range(n_classes) if 0 < indicator[:, c].sum() < len(y)]
    if not usable:
        return float("nan")
    return float(
        np.mean([roc_auc_score(indicator[:, c], scores[:, c]) for c in usable])
    )


def compute_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute macro F1. None predictions are replaced with the worst possible prediction."""
    y_pred_clean = []
    for label, pred in zip(y_true, y_pred):
        if _is_invalid(pred):
            y_pred_clean.append(0 if label != 0 else 1)
        else:
            y_pred_clean.append(int(pred))
    return float(f1_score(y_true, y_pred_clean, average="macro", zero_division=0))


def compute_ovr_f1(y_true: list[int], y_pred: list[list[float]]) -> float:
    """Compute OvR macro F1. y_pred shape (N, C). None rows replaced with worst predictions."""
    n_classes = next(len(row) for row in y_pred if row is not None)
    y_pred_clean = []
    for label, row in zip(y_true, y_pred):
        if row is None or any(_is_invalid(s) for s in row):
            y_pred_clean.append((label + 1) % n_classes)
        else:
            y_pred_clean.append(int(max(range(n_classes), key=lambda c: row[c])))
    return float(f1_score(y_true, y_pred_clean, average="macro", zero_division=0))
