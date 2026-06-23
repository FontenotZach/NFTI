from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.paths import METRICS_CSV_PATH, PREDICTIONS_CSV_PATH, ensure_dirs

METRICS_COLUMNS = [
    "timestamp",
    "model_type",
    "metric_to_predict",
    "split_name",
    "threshold_type",
    "threshold",
    "n",
    "n_positive",
    "n_negative",
    "prevalence",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision_ppv",
    "npv",
    "f1",
    "auroc",
    "auprc",
    "brier",
    "model_path",
    "log_filename",
]

PREDICTIONS_COLUMNS = [
    "record_id",
    "y_true",
    "y_pred_prob",
    "y_pred",
    "timestamp",
    "model_type",
    "metric_to_predict",
    "split_name",
]


def record_id_from_trauma_record(record, idx: int) -> str:
    for key in ("inc_key", "INC_KEY", "IncKey"):
        value = record.data.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return str(value)
    return f"row_{idx}"


def clean_binary_eval_inputs(
    y_true,
    y_prob,
    record_ids: Optional[Sequence] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List], np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)

    if record_ids is not None and len(record_ids) != len(y_true_arr):
        raise ValueError(
            f"record_ids length ({len(record_ids)}) must match y_true length ({len(y_true_arr)})."
        )

    valid_y = ~np.isnan(y_true_arr)
    valid_prob = np.isfinite(y_prob_arr)
    mask = valid_y & valid_prob

    filtered_ids = None
    if record_ids is not None:
        filtered_ids = [record_ids[i] for i in range(len(mask)) if mask[i]]

    return (
        y_true_arr[mask].astype(int),
        y_prob_arr[mask],
        filtered_ids,
        mask,
    )


def select_threshold_youden(y_true, y_prob) -> float:
    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    if len(np.unique(y_clean)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_clean, p_clean)
    if len(thresholds) == 0:
        return 0.5
    optimal_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[optimal_idx])


def select_threshold_f1(y_true, y_prob) -> float:
    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    if len(np.unique(y_clean)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_clean, p_clean)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = [f1_score(y_clean, p_clean >= t, zero_division=0) for t in thresholds]
    optimal_idx = int(np.argmax(f1_scores))
    return float(thresholds[optimal_idx])


def _safe_ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float("nan")
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        brier = float("nan")
    return auroc, auprc, brier


def _confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = int(len(y_true))
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    prevalence = n_positive / n if n else float("nan")

    accuracy = (tp + tn) / n if n else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision_ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    if not np.isnan(precision_ppv) and not np.isnan(sensitivity) and (precision_ppv + sensitivity):
        f1 = 2 * precision_ppv * sensitivity / (precision_ppv + sensitivity)
    else:
        f1 = float("nan")

    return {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": prevalence,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision_ppv,
        "npv": npv,
        "f1": f1,
    }


def evaluate_binary_classifier(
    y_true,
    y_prob,
    *,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    auroc, auprc, brier = _safe_ranking_metrics(y_clean, p_clean)

    result: Dict[str, float] = {
        "n": float(len(y_clean)),
        "n_positive": float((y_clean == 1).sum()) if len(y_clean) else float("nan"),
        "n_negative": float((y_clean == 0).sum()) if len(y_clean) else float("nan"),
        "prevalence": float((y_clean == 1).mean()) if len(y_clean) else float("nan"),
        "tp": float("nan"),
        "tn": float("nan"),
        "fp": float("nan"),
        "fn": float("nan"),
        "accuracy": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "precision_ppv": float("nan"),
        "npv": float("nan"),
        "f1": float("nan"),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
    }

    if threshold is not None and len(y_clean):
        confusion = _confusion_at_threshold(y_clean, p_clean, threshold)
        result.update(confusion)

    return result


def _append_csv_row(path: Path, columns: Sequence[str], row: Dict[str, Any]) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def save_metrics_row(row: Dict[str, Any], path: Optional[Union[str, Path]] = None) -> Path:
    out_path = Path(path) if path is not None else METRICS_CSV_PATH
    _append_csv_row(out_path, METRICS_COLUMNS, row)
    return out_path


def save_row_level_predictions(
    rows: Iterable[Dict[str, Any]],
    path: Optional[Union[str, Path]] = None,
) -> Path:
    out_path = Path(path) if path is not None else PREDICTIONS_CSV_PATH
    for row in rows:
        _append_csv_row(out_path, PREDICTIONS_COLUMNS, row)
    return out_path


def format_metrics_for_log(row: Dict[str, Any]) -> str:
    def _fmt(key: str, pct: bool = False) -> str:
        val = row.get(key, float("nan"))
        if val == "" or val is None:
            return "N/A"
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            return str(val)
        if np.isnan(val_f):
            return "N/A"
        if pct:
            return f"{val_f * 100:.2f}%"
        return f"{val_f:.4f}"

    return (
        f"\n--- {row.get('metric_to_predict')} {row.get('model_type')} "
        f"{row.get('split_name')} ({row.get('threshold_type')}) ---\n"
        f"Threshold: {_fmt('threshold')}\n"
        f"n={row.get('n', 'N/A')}, n_positive={row.get('n_positive', 'N/A')}, "
        f"n_negative={row.get('n_negative', 'N/A')}, prevalence={_fmt('prevalence', pct=True)}\n"
        f"Accuracy: {_fmt('accuracy', pct=True)}\n"
        f"Sensitivity: {_fmt('sensitivity', pct=True)}\n"
        f"Specificity: {_fmt('specificity', pct=True)}\n"
        f"Precision (PPV): {_fmt('precision_ppv', pct=True)}\n"
        f"NPV: {_fmt('npv', pct=True)}\n"
        f"F1: {_fmt('f1', pct=True)}\n"
        f"AUROC: {_fmt('auroc')}\n"
        f"AUPRC: {_fmt('auprc')}\n"
        f"Brier: {_fmt('brier')}\n"
        f"Confusion matrix: tp={row.get('tp')}, tn={row.get('tn')}, "
        f"fp={row.get('fp')}, fn={row.get('fn')}\n"
    )
