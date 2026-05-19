"""Classification metrics."""

from sklearn.metrics import f1_score, roc_auc_score

from ._config import MetricsConfig


def compute_auroc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute Area Under ROC Curve."""
    if len(y_true) != len(y_scores) or len(y_true) == 0:
        return MetricsConfig.AUROC_WORST
    try:
        if len(set(y_true)) < 2:
            return MetricsConfig.AUROC_WORST
        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        return MetricsConfig.AUROC_WORST


def compute_ovr_auroc(y_true: list[int], y_scores: list[list[float]]) -> float:
    """OvR macro AUROC. y_scores shape (N, C)."""
    if len(y_true) == 0 or len(y_scores) == 0:
        return MetricsConfig.AUROC_WORST
    if len(y_true) != len(y_scores):
        return MetricsConfig.AUROC_WORST
    try:
        if len(set(y_true)) < 2:
            return MetricsConfig.AUROC_WORST
        return float(roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro"))
    except Exception:
        return MetricsConfig.AUROC_WORST


def compute_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute macro-averaged F1 score."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return MetricsConfig.F1_WORST
    try:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return MetricsConfig.F1_WORST
