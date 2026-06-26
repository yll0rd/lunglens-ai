"""
Threshold selection utility with specificity constraint.

Adapted from dual_stage_xray_pipeline_v2.ipynb
"""

import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple, Dict


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float = 0.70
) -> Dict[str, float]:
    """
    Compute optimal threshold with specificity constraint.

    Selects threshold that:
    1. Achieves at least `min_specificity` (default 70%)
    2. Among valid thresholds, maximizes sensitivity (TPR)
    3. Falls back to Youden's Index if min_specificity unreachable

    This is more medically sound than pure Youden because it prioritizes
    avoiding false positives (maintaining specificity) while still catching
    as many true positives as possible.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities [0, 1]
        min_specificity: Minimum required specificity (default 0.70 = 70%)

    Returns:
        Dictionary with metrics:
        - threshold: Selected decision threshold
        - sensitivity: TPR at selected threshold
        - specificity: TNR at selected threshold
        - auroc: Area under ROC curve
        - auprc: Area under precision-recall curve
    """
    from sklearn.metrics import auc

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "auprc": None, "threshold": 0.5}

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Fix potential inf in first threshold
    if thresholds.size > 0:
        thresholds[0] = min(thresholds[0], 1.0)

    # Calculate specificity (1 - FPR)
    specificity = 1 - fpr

    # Find thresholds meeting minimum specificity requirement
    valid = specificity >= min_specificity

    if valid.any():
        # Among valid thresholds, pick one with highest sensitivity
        max_valid_tpr = np.max(tpr[valid])
        best_indices = np.where((tpr == max_valid_tpr) & valid)[0]
        best_index = best_indices[np.argmax(specificity[best_indices])]
        threshold = float(thresholds[best_index])
    else:
        # Fallback: use Youden's Index if min_specificity unreachable
        print(
            f"Warning: Model never achieved {min_specificity*100:.0f}% specificity. "
            "Falling back to Youden's Index."
        )
        youden_index = tpr - fpr
        best_index = int(np.argmax(youden_index))
        threshold = float(thresholds[best_index])

    # Compute metrics at selected threshold
    preds = (y_prob >= threshold).astype(int)
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tp = np.sum((preds == 1) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_actual = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # AUROC
    auroc = auc(fpr, tpr) if len(fpr) > 0 else None

    # AUPRC
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision) if len(recall) > 0 else None

    return {
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity_actual),
        "ppv": float(ppv),
        "npv": float(npv),
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
    }


if __name__ == "__main__":
    import json

    # Example usage
    print("Threshold selection utility for Streamlit app")
    print("\nUsage:")
    print("  from threshold_utils import compute_optimal_threshold")
    print("  metrics = compute_optimal_threshold(y_true, y_prob, min_specificity=0.70)")
    print("  print(f'Optimal threshold: {metrics[\"threshold\"]}')")
