# challenges-metrics

Metric functions for the [FOMO26 Challenge](https://fomo26.github.io).

## Installation

```bash
uv pip install challenges-metrics
```

For development:

```bash
uv pip install -e ".[dev]"
```

## Metrics

### Classification

```python
from fomo_challenge_metrics import compute_auroc, compute_ovr_auroc, compute_macro_f1

# Binary AUROC
compute_auroc(y_true=[0, 1, 1], y_scores=[0.1, 0.8, 0.9])

# One-vs-Rest macro AUROC (y_scores shape: N × C)
compute_ovr_auroc(y_true=[0, 1, 2], y_scores=[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]])

# Macro F1
compute_macro_f1(y_true=[0, 1, 2], y_pred=[0, 1, 2])
```

### Regression

```python
from fomo_challenge_metrics import compute_absolute_error, compute_correlation

compute_absolute_error(y_true=[1.0, 2.0], y_pred=[1.1, 1.9])
compute_correlation(y_true=[1.0, 2.0, 3.0], y_pred=[1.0, 2.1, 2.9])
```

### Segmentation

```python
import numpy as np
from fomo_challenge_metrics import (
    compute_dice_coefficient,
    compute_normalized_surface_distance,
    compute_multiclass_dsc,
    compute_multiclass_nsd,
)

pred = np.zeros((64, 64, 32), dtype=bool)
gt   = np.zeros((64, 64, 32), dtype=bool)

# Binary
compute_dice_coefficient(pred, gt)
compute_normalized_surface_distance(pred, gt, spacing_mm=(1.0, 1.0, 1.5))

# Multiclass — label 0 (background) always excluded
# Labels inferred from gt if not provided
compute_multiclass_dsc(pred_mask, gt_mask)              # -> {1: 0.95, 2: 0.87, ...}
compute_multiclass_nsd(pred_mask, gt_mask, spacing_mm=(1.0, 1.0, 1.5))
```

### Fairness

```python
from fomo_challenge_metrics import compute_max_disparity, compute_fairness_score

y_true   = [0, 1, 2, 0, 1, 2]
y_scores = [
    [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9],
    [0.8, 0.10, 0.10], [0.10, 0.8, 0.10], [0.10, 0.10, 0.8],
]

# Group labels must be pre-binned integers (or any hashable) by the caller.
# None entries are excluded from disparity calculations.
groups_a = []   # e.g. a variable A
groups_b = [2]   # e.g. a variable B

# Maximum disparity for a single variable
d = compute_max_disparity(y_true, y_scores, groups_a, compute_ovr_f1)

# Fairness score aggregated across multiple variables
result = compute_fairness_score(
    y_true, y_scores,
    groups_by_variable={"var_a": groups_a, "var_b": groups_b},
    metric_fn=compute_ovr_f1,
)
# result["score"]                      → scalar in [0, 1]; 1.0 = perfect equity
# result["disparities"]                → {"var_a": ..., "var_b": ...}
# result["variables_used"]             → variables with a defined disparity
# result["per_variable_contribution"]  → 1 - disparity per variable
```

**Fairness score** (per metric M):

    FairnessScore(M) = (1 / |V'|) * Σ_{v in V'} (1 - D_v(M))

where `D_v(M)` is the maximum disparity across groups in variable `v`,
and `V'` is the subset of variables for which `D_v` is defined.
Score of **1.0** = perfect equity; **0.0** = maximum disparity on every variable.


