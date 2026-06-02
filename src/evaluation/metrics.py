"""
Classification metrics suitable for imbalanced clinical outcomes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

# Threshold policy / provenance labels (for JSON reports).
THRESHOLD_POLICY_YOUDEN_J = "youden_j"
THRESHOLD_SOURCE_TRAIN_CV_OOF = "train_cv_oof_predictions"
THRESHOLD_SOURCE_TRAIN_INTERNAL_VAL = "train_internal_validation_split"
# Fallback when no saved train-derived threshold is available (debug / evaluation tools).
THRESHOLD_SOURCE_FIXED_DEFAULT = "fixed_default_0_5"


def youden_j_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC threshold that maximizes Youden's J (TPR - FPR).

    Uses training or out-of-fold predictions only — never holdout test labels.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def select_threshold_youden_j(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, str]:
    """Return (threshold, threshold_policy)."""
    return youden_j_threshold(y_true, y_score), THRESHOLD_POLICY_YOUDEN_J


def classification_metrics_dict(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute threshold-free and optional threshold-based metrics.

    y_true: binary labels (0/1).
    y_score: predicted probability of positive class.
    threshold: if provided, adds sensitivity, specificity, balanced accuracy, MCC at that threshold.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    out: Dict[str, Any] = {}

    if len(np.unique(y_true)) >= 2 and not np.isnan(y_score).any():
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
        try:
            out["brier_score"] = float(brier_score_loss(y_true, y_score))
        except ValueError:
            out["brier_score"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        out["brier_score"] = float("nan")

    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        out["threshold"] = float(threshold)
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        out["sensitivity"] = float(sens)
        out["specificity"] = float(spec)
        out["tp"] = int(tp)
        out["tn"] = int(tn)
        out["fp"] = int(fp)
        out["fn"] = int(fn)

    return out


def format_metrics_log(metrics: Dict[str, Any], title: str) -> str:
    lines = [f"\n--- {title} ---"]
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"{k}: {v:.6f}" if not np.isnan(v) else f"{k}: nan")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"
