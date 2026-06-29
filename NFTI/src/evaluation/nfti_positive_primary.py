from __future__ import annotations

import csv
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.binary_metrics import (
    calculate_binary_classification_metrics,
    clean_binary_eval_inputs,
)
from src.data.human_readable import get_label, load_human_readable_map
from src.evaluation.data_splits import NftiPositiveSplits, build_nfti_positive_splits
from src.models.logistic_regression_model import (
    LogisticRegressionTrainingResult,
    predict_proba_positive,
    train_logistic_regression_classifier,
)
from src.models.xgboost_model import train_xgboost_classifier
from src.plotting import NAVY, RED, apply_manuscript_grid
from src.reporting.training_headers import write_training_headers_report
from src.paths import (
    LOGS_DIR,
    NFTI_POSITIVE_DISCRIMINATION_CURVES_FIGURE_PATH,
    NFTI_POSITIVE_EVALUATION_SUMMARY_PATH,
    NFTI_POSITIVE_FEATURE_LIST_PATH,
    NFTI_POSITIVE_LR_HOLDOUT_PREDICTIONS_PATH,
    NFTI_POSITIVE_LR_MODEL_PATH,
    NFTI_POSITIVE_LR_VALIDATION_PREDICTIONS_PATH,
    NFTI_POSITIVE_MODEL_COMPARISON_PATH,
    NFTI_POSITIVE_MODEL_COMPARISON_SUMMARY_PATH,
    FIGURE4_CALIBRATION_AND_DECILES_PATH,
    NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH,
    NFTI_POSITIVE_XGB_CALIBRATION_BINS_PATH,
    NFTI_POSITIVE_XGB_CALIBRATION_CURVE_PATH,
    NFTI_POSITIVE_XGB_HOLDOUT_80_SENSITIVITY_METRICS_PATH,
    NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH,
    NFTI_POSITIVE_XGB_HOLDOUT_THRESHOLD_SWEEP_PATH,
    NFTI_POSITIVE_XGB_MODEL_PATH,
    NFTI_POSITIVE_XGB_RISK_DECILES_FIGURE_PATH,
    NFTI_POSITIVE_XGB_RISK_DECILES_PATH,
    NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH,
    NFTI_POSITIVE_XGB_THRESHOLD_0_5_METRICS_PATH,
    NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_HEATMAP_PATH,
    NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_LINES_PATH,
    NFTI_POSITIVE_XGB_VALIDATION_PREDICTIONS_PATH,
    NFTI_POSITIVE_XGB_VALIDATION_THRESHOLD_SWEEP_PATH,
    NFTI_POSITIVE_XGB_VS_LR_DISCRIMINATION_CURVES_FIGURE_PATH,
    NFTI_POSITIVE_XGB_VS_LR_FIGURE_PATH,
    PRIMARY_CRITERION,
    ensure_dirs,
)

VALIDATION_THRESHOLD_GRID = np.arange(0.01, 1.00, 0.01)
HOLDOUT_THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)
TARGET_VALIDATION_SENSITIVITY = 0.80


def _save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    columns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_single_row_csv(path: Path, row: Dict[str, Any]) -> Path:
    return _save_csv(path, [row])


def _save_predictions_csv(
    path: Path,
    record_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Path:
    y_clean, p_clean, ids_clean, _ = clean_binary_eval_inputs(y_true, y_prob, record_ids)
    rows = [
        {"record_id": rid, "y_true": int(yt), "y_pred_prob": float(yp)}
        for rid, yt, yp in zip(ids_clean, y_clean, p_clean)
    ]
    return _save_csv(path, rows, fieldnames=["record_id", "y_true", "y_pred_prob"])


def _metrics_without_ranking(y_true, y_prob, threshold: float) -> Dict[str, float]:
    row = calculate_binary_classification_metrics(y_true, y_prob, threshold)
    for key in ("AUROC", "AUPRC", "Brier"):
        row.pop(key, None)
    return row


def _ranking_metrics(y_true, y_prob) -> Dict[str, float]:
    return calculate_binary_classification_metrics(y_true, y_prob, threshold=0.5)


def _select_validation_sensitivity_threshold(
    y_true,
    y_prob,
) -> Tuple[float, List[Dict[str, float]], bool]:
    sweep_rows: List[Dict[str, float]] = []
    for threshold in VALIDATION_THRESHOLD_GRID:
        row = _metrics_without_ranking(y_true, y_prob, float(threshold))
        row["threshold"] = float(threshold)
        sweep_rows.append(row)

    qualifying = [
        float(row["threshold"])
        for row in sweep_rows
        if np.isfinite(row["sensitivity"]) and row["sensitivity"] >= TARGET_VALIDATION_SENSITIVITY
    ]
    warning_issued = False
    if qualifying:
        selected = max(qualifying)
    else:
        warning_issued = True
        sensitivities = [
            (float(row["threshold"]), row["sensitivity"])
            for row in sweep_rows
            if np.isfinite(row["sensitivity"])
        ]
        selected = max(sensitivities, key=lambda item: item[1])[0] if sensitivities else 0.5
        warnings.warn(
            f"No validation threshold reached sensitivity >= {TARGET_VALIDATION_SENSITIVITY:.2f}; "
            f"selected threshold {selected:.2f} with maximum validation sensitivity.",
            UserWarning,
            stacklevel=2,
        )

    return selected, sweep_rows, warning_issued


def _calibration_bins(y_true, y_prob, *, n_bins: int = 10) -> List[Dict[str, Any]]:
    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    if len(y_clean) == 0:
        return []

    df = pd.DataFrame({"y_true": y_clean, "y_prob": p_clean})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    rows: List[Dict[str, Any]] = []
    for bin_number, (bin_label, group) in enumerate(df.groupby("bin", observed=False), start=1):
        positives = int((group["y_true"] == 1).sum())
        negatives = int((group["y_true"] == 0).sum())
        rows.append(
            {
                "bin": bin_number,
                "n": int(len(group)),
                "mean_predicted_probability": float(group["y_prob"].mean()),
                "observed_nfti_rate": float(group["y_true"].mean()),
                "nfti_positive_count": positives,
                "nfti_negative_count": negatives,
                "bin_lower": float(bin_label.left),
                "bin_upper": float(bin_label.right),
            }
        )
    return rows


def _risk_deciles(y_true, y_prob) -> List[Dict[str, Any]]:
    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    if len(y_clean) == 0:
        return []

    df = pd.DataFrame({"y_true": y_clean, "y_prob": p_clean})
    df["decile"] = pd.qcut(df["y_prob"], q=10, labels=False, duplicates="drop") + 1
    rows: List[Dict[str, Any]] = []
    for decile, group in df.groupby("decile", observed=False):
        positives = int((group["y_true"] == 1).sum())
        negatives = int((group["y_true"] == 0).sum())
        rows.append(
            {
                "decile": int(decile),
                "n": int(len(group)),
                "mean_predicted_probability": float(group["y_prob"].mean()),
                "median_predicted_probability": float(group["y_prob"].median()),
                "min_predicted_probability": float(group["y_prob"].min()),
                "max_predicted_probability": float(group["y_prob"].max()),
                "observed_nfti_rate": float(group["y_true"].mean()),
                "nfti_positive_count": positives,
                "nfti_negative_count": negatives,
            }
        )
    return sorted(rows, key=lambda row: row["decile"])


def _plot_xgb_vs_lr(metrics_xgb: Dict[str, float], metrics_lr: Dict[str, float], path: Path) -> None:
    labels = ["AUROC", "AUPRC", "Brier"]
    xgb_vals = [metrics_xgb["AUROC"], metrics_xgb["AUPRC"], metrics_xgb["Brier"]]
    lr_vals = [metrics_lr["AUROC"], metrics_lr["AUPRC"], metrics_lr["Brier"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    apply_manuscript_grid(ax)
    ax.bar(x - width / 2, xgb_vals, width, label="XGBoost", color=NAVY)
    ax.bar(x + width / 2, lr_vals, width, label="Logistic Regression", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("nfti_positive Holdout Core Metrics: XGBoost vs LR")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_threshold_sweep_lines(sweep_rows: List[Dict[str, float]], path: Path) -> None:
    if not sweep_rows:
        return
    df = pd.DataFrame(sweep_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_manuscript_grid(ax)
    for metric in ("sensitivity", "specificity", "precision", "NPV", "F1"):
        if metric in df.columns:
            ax.plot(df["threshold"], df[metric], label=metric.upper() if metric != "precision" else "PPV")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("XGBoost Holdout Threshold Sweep")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_threshold_sweep_heatmap(sweep_rows: List[Dict[str, float]], path: Path) -> None:
    if not sweep_rows:
        return
    df = pd.DataFrame(sweep_rows)
    metrics = ["sensitivity", "specificity", "precision", "NPV", "F1", "accuracy"]
    heatmap_df = df.set_index("threshold")[metrics].T
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.upper() if m != "precision" else "PPV" for m in metrics])
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels([f"{t:.2f}" for t in heatmap_df.columns], rotation=90)
    ax.set_xlabel("Threshold")
    ax.set_title("XGBoost Holdout Threshold Sweep Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_calibration_curve(calibration_rows: List[Dict[str, Any]], path: Path) -> None:
    if not calibration_rows:
        return
    df = pd.DataFrame(calibration_rows)
    fig, ax = plt.subplots(figsize=(7, 6))
    apply_manuscript_grid(ax)
    ax.plot(df["mean_predicted_probability"], df["observed_nfti_rate"], marker="o", color=NAVY, label="XGBoost")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed NFTI rate")
    ax.set_title("XGBoost Holdout Calibration Curve")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _augment_decile_summary(decile_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build the decile summary DataFrame used for plotting, augmenting it with
    observed-rate 95% binomial confidence intervals derived from the decile
    counts. Predicted-risk CIs are only added when an uncertainty estimate is
    genuinely available in the summary; otherwise they are left absent rather
    than fabricated.
    """
    df = pd.DataFrame(decile_rows).sort_values("decile").reset_index(drop=True)
    df["decile"] = df["decile"].astype(int)

    df["observed_nfti_rate"] = df["observed_nfti_rate"].astype(float)
    df["mean_predicted_risk"] = df["mean_predicted_probability"].astype(float)

    # Observed-rate 95% CI via the binomial standard error: SE = sqrt(p(1-p)/n).
    p = df["observed_nfti_rate"].to_numpy(dtype=float)
    n = df["n"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(np.where(n > 0, p * (1.0 - p) / n, np.nan))
    df["observed_ci_lower"] = np.clip(p - 1.96 * se, 0.0, 1.0)
    df["observed_ci_upper"] = np.clip(p + 1.96 * se, 0.0, 1.0)

    return df


def _plot_risk_deciles(decile_rows: List[Dict[str, Any]], path: Path) -> None:
    if not decile_rows:
        return

    df = _augment_decile_summary(decile_rows)
    deciles = df["decile"].to_numpy()
    observed_rate = df["observed_nfti_rate"].to_numpy(dtype=float)
    mean_predicted_risk = df["mean_predicted_risk"].to_numpy(dtype=float)

    # Asymmetric error-bar offsets for the observed-rate bars (lower, upper).
    observed_yerr = np.vstack(
        [
            observed_rate - df["observed_ci_lower"].to_numpy(dtype=float),
            df["observed_ci_upper"].to_numpy(dtype=float) - observed_rate,
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    apply_manuscript_grid(ax)

    ax.bar(
        deciles,
        observed_rate,
        color=NAVY,
        alpha=0.85,
        label="Observed NFTI rate",
        yerr=observed_yerr,
        ecolor=NAVY,
        capsize=3,
        error_kw={"elinewidth": 1.0, "alpha": 0.7},
    )

    # Per-record predicted probabilities are not retained in the decile
    # summary, so no valid predicted-risk uncertainty estimate exists;
    # plot the mean predicted risk as red points without error bars.
    ax.scatter(
        deciles,
        mean_predicted_risk,
        color=RED,
        s=35,
        label="Mean predicted risk",
        zorder=3,
    )

    ax.set_xlabel("Predicted-risk decile (1 = lowest risk)")
    ax.set_ylabel("NFTI-positive rate")
    ax.set_title("Observed NFTI Rate by Predicted Risk Decile")
    ax.set_xticks(deciles)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_calibration_panel(ax, calibration_rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(calibration_rows)
    apply_manuscript_grid(ax)
    ax.plot(
        df["mean_predicted_probability"],
        df["observed_nfti_rate"],
        marker="o",
        color=NAVY,
        label="XGBoost",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed NFTI rate")
    ax.set_title("A. Calibration Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=True)


def _plot_risk_deciles_panel(ax, decile_rows: List[Dict[str, Any]]) -> None:
    df = _augment_decile_summary(decile_rows)
    deciles = df["decile"].to_numpy()
    observed_rate = df["observed_nfti_rate"].to_numpy(dtype=float)
    mean_predicted_risk = df["mean_predicted_risk"].to_numpy(dtype=float)
    observed_yerr = np.vstack(
        [
            observed_rate - df["observed_ci_lower"].to_numpy(dtype=float),
            df["observed_ci_upper"].to_numpy(dtype=float) - observed_rate,
        ]
    )

    apply_manuscript_grid(ax)
    ax.bar(
        deciles,
        observed_rate,
        color=NAVY,
        alpha=0.85,
        label="Observed NFTI rate",
        yerr=observed_yerr,
        ecolor=NAVY,
        capsize=3,
        error_kw={"elinewidth": 1.0, "alpha": 0.7},
    )
    ax.scatter(
        deciles,
        mean_predicted_risk,
        color=RED,
        s=35,
        label="Mean predicted risk",
        zorder=3,
    )
    ax.set_xlabel("Predicted-risk decile (1 = lowest risk)")
    ax.set_ylabel("NFTI-positive rate")
    ax.set_title("B. Observed NFTI Rate by Predicted Risk Decile")
    ax.set_xticks(deciles)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=True)


def _plot_calibration_and_deciles_combined(
    calibration_rows: List[Dict[str, Any]],
    decile_rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    if not calibration_rows or not decile_rows:
        return

    fig, (ax_cal, ax_dec) = plt.subplots(1, 2, figsize=(14, 5.5))
    _plot_calibration_panel(ax_cal, calibration_rows)
    _plot_risk_deciles_panel(ax_dec, decile_rows)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _discrimination_curve_data(y_true, y_prob) -> Optional[Dict[str, Any]]:
    """
    Compute ROC and precision-recall curve coordinates plus AUROC/AUPRC for a
    single model from already-generated labels and probabilities. Returns None
    when discrimination is undefined (no rows or a single observed class).
    """
    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    if len(y_clean) == 0 or len(np.unique(y_clean)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_clean, p_clean)
    precision, recall, _ = precision_recall_curve(y_clean, p_clean)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "auroc": float(roc_auc_score(y_clean, p_clean)),
        "auprc": float(average_precision_score(y_clean, p_clean)),
        "prevalence": float((y_clean == 1).mean()),
        "n": int(len(y_clean)),
    }


def _plot_discrimination_curves(
    y_true,
    xgb_pred_prob,
    output_path: Path,
    *,
    lr_pred_prob=None,
    model_name: str = "XGBoost",
) -> Optional[Path]:
    """
    Manuscript-quality two-panel holdout discrimination figure: ROC (Panel A)
    and precision-recall (Panel B), comparing XGBoost and, when available,
    logistic regression. This function consumes already-generated holdout
    labels and probabilities; it never fits or recomputes models. A small CSV
    summary (model, auroc, auprc, prevalence, n) is written next to the figure.
    """
    xgb_data = _discrimination_curve_data(y_true, xgb_pred_prob)
    if xgb_data is None:
        return None

    models: List[Tuple[str, str, Dict[str, Any]]] = [(model_name, NAVY, xgb_data)]
    if lr_pred_prob is not None:
        lr_data = _discrimination_curve_data(y_true, lr_pred_prob)
        if lr_data is not None:
            models.append(("Logistic Regression", RED, lr_data))

    prevalence = xgb_data["prevalence"]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))
    apply_manuscript_grid(ax_roc)
    apply_manuscript_grid(ax_pr)

    # Panel A: ROC curve.
    for label, color, data in models:
        ax_roc.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            lw=2.0,
            label=f"{label} (AUROC = {data['auroc']:.3f})",
        )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="No discrimination")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("1 - Specificity")
    ax_roc.set_ylabel("Sensitivity")
    ax_roc.set_title("A. ROC Curve")
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.set_box_aspect(1)

    # Panel B: Precision-recall curve.
    for label, color, data in models:
        ax_pr.plot(
            data["recall"],
            data["precision"],
            color=color,
            lw=2.0,
            label=f"{label} (AUPRC = {data['auprc']:.3f})",
        )
    ax_pr.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        lw=1.5,
        label=f"Prevalence = {prevalence:.3f}",
    )
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Sensitivity")
    ax_pr.set_ylabel("Positive predictive value")
    ax_pr.set_title("B. Precision-Recall Curve")
    ax_pr.legend(loc="upper right", frameon=True)
    ax_pr.set_box_aspect(1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    summary_rows = [
        {
            "model": label,
            "auroc": data["auroc"],
            "auprc": data["auprc"],
            "prevalence": data["prevalence"],
            "n": data["n"],
        }
        for label, _, data in models
    ]
    _save_csv(
        output_path.with_suffix(".csv"),
        summary_rows,
        fieldnames=["model", "auroc", "auprc", "prevalence", "n"],
    )
    return output_path


def _comparison_row(model_name: str, y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    metrics = calculate_binary_classification_metrics(y_true, y_prob, threshold)
    return {"model": model_name, **metrics}


def _discrimination_winner(xgb_metrics: Dict[str, float], lr_metrics: Dict[str, float]) -> str:
    xgb_score = (xgb_metrics.get("AUROC", float("nan")), xgb_metrics.get("AUPRC", float("nan")))
    lr_score = (lr_metrics.get("AUROC", float("nan")), lr_metrics.get("AUPRC", float("nan")))
    if xgb_score > lr_score:
        return "XGBoost"
    if lr_score > xgb_score:
        return "Logistic Regression"
    return "Tie"


def run_nfti_positive_primary_evaluation(
    trauma_dataset,
    *,
    class_weight_balanced: bool = True,
    grid_search: bool = True,
    use_smote: bool = False,
    validation_fraction: float = 0.15,
    random_state: int = 42,
    log_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train XGBoost and logistic regression baselines for nfti_positive and generate
    the full primary-model evaluation package.
    """
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = log_filename or f"log_nfti_positive_primary_{timestamp}.txt"
    log_path = LOGS_DIR / log_name

    splits = build_nfti_positive_splits(
        trauma_dataset,
        criterion=PRIMARY_CRITERION,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    hr_map = load_human_readable_map()
    feature_rows = [
        {"index": i, "feature_name": name, "human_readable": get_label(name, hr_map)}
        for i, name in enumerate(splits.feature_names)
    ]
    _save_csv(
        NFTI_POSITIVE_FEATURE_LIST_PATH,
        feature_rows,
        fieldnames=["index", "feature_name", "human_readable"],
    )

    # Definitive, human-meaningful audit of the source headers used in training
    # (distinct from the expanded post-one-hot feature matrix columns above).
    write_training_headers_report(trauma_dataset)

    xgb_model, xgb_hyperparameters = train_xgboost_classifier(
        splits.X_train,
        splits.y_train,
        grid_search=grid_search,
        use_smote=use_smote,
        random_state=random_state,
    )
    xgb_model.save_model(str(NFTI_POSITIVE_XGB_MODEL_PATH))

    lr_result: LogisticRegressionTrainingResult = train_logistic_regression_classifier(
        splits.X_train,
        splits.y_train,
        class_weight_balanced=class_weight_balanced,
        random_state=random_state,
        model_path=str(NFTI_POSITIVE_LR_MODEL_PATH),
    )

    xgb_val_prob = xgb_model.predict_proba(splits.X_validation)[:, 1]
    xgb_holdout_prob = xgb_model.predict_proba(splits.X_holdout)[:, 1]
    lr_val_prob = predict_proba_positive(lr_result.pipeline, splits.X_validation)
    lr_holdout_prob = predict_proba_positive(lr_result.pipeline, splits.X_holdout)

    _save_predictions_csv(
        NFTI_POSITIVE_XGB_VALIDATION_PREDICTIONS_PATH,
        splits.validation_record_ids,
        splits.y_validation,
        xgb_val_prob,
    )
    _save_predictions_csv(
        NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH,
        splits.holdout_record_ids,
        splits.y_holdout,
        xgb_holdout_prob,
    )
    _save_predictions_csv(
        NFTI_POSITIVE_LR_VALIDATION_PREDICTIONS_PATH,
        splits.validation_record_ids,
        splits.y_validation,
        lr_val_prob,
    )
    _save_predictions_csv(
        NFTI_POSITIVE_LR_HOLDOUT_PREDICTIONS_PATH,
        splits.holdout_record_ids,
        splits.y_holdout,
        lr_holdout_prob,
    )

    xgb_holdout_ranking = _ranking_metrics(splits.y_holdout, xgb_holdout_prob)
    lr_holdout_ranking = _ranking_metrics(splits.y_holdout, lr_holdout_prob)

    comparison_rows = [
        _comparison_row("xgboost", splits.y_holdout, xgb_holdout_prob, threshold=0.5),
        _comparison_row("logistic_regression", splits.y_holdout, lr_holdout_prob, threshold=0.5),
    ]
    _save_csv(NFTI_POSITIVE_MODEL_COMPARISON_PATH, comparison_rows)

    winner = _discrimination_winner(xgb_holdout_ranking, lr_holdout_ranking)
    comparison_summary = (
        "nfti_positive Model Comparison: XGBoost vs Logistic Regression (holdout)\n"
        f"XGBoost AUROC={xgb_holdout_ranking['AUROC']:.4f}, "
        f"AUPRC={xgb_holdout_ranking['AUPRC']:.4f}, Brier={xgb_holdout_ranking['Brier']:.4f}\n"
        f"Logistic Regression AUROC={lr_holdout_ranking['AUROC']:.4f}, "
        f"AUPRC={lr_holdout_ranking['AUPRC']:.4f}, Brier={lr_holdout_ranking['Brier']:.4f}\n"
        f"Better holdout discrimination: {winner}\n"
        "Logistic regression is included as a conventional baseline comparator only.\n"
        "XGBoost remains the primary model for threshold, calibration, risk-decile, "
        "and interpretability analyses.\n"
    )
    NFTI_POSITIVE_MODEL_COMPARISON_SUMMARY_PATH.write_text(comparison_summary, encoding="utf-8")
    _plot_xgb_vs_lr(xgb_holdout_ranking, lr_holdout_ranking, NFTI_POSITIVE_XGB_VS_LR_FIGURE_PATH)

    xgb_threshold_0_5 = calculate_binary_classification_metrics(
        splits.y_holdout, xgb_holdout_prob, threshold=0.5
    )
    _save_single_row_csv(NFTI_POSITIVE_XGB_THRESHOLD_0_5_METRICS_PATH, xgb_threshold_0_5)

    selected_threshold, validation_sweep, sensitivity_warning = _select_validation_sensitivity_threshold(
        splits.y_validation,
        xgb_val_prob,
    )
    _save_csv(NFTI_POSITIVE_XGB_VALIDATION_THRESHOLD_SWEEP_PATH, validation_sweep)

    validation_at_selected = calculate_binary_classification_metrics(
        splits.y_validation,
        xgb_val_prob,
        threshold=selected_threshold,
    )
    selected_row = {
        "selected_threshold": selected_threshold,
        "selection_source": "validation",
        "target_sensitivity": TARGET_VALIDATION_SENSITIVITY,
        "sensitivity_warning_issued": sensitivity_warning,
        **validation_at_selected,
    }
    _save_single_row_csv(NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH, selected_row)

    holdout_at_selected = calculate_binary_classification_metrics(
        splits.y_holdout,
        xgb_holdout_prob,
        threshold=selected_threshold,
    )
    holdout_selected_row = {
        "selected_threshold": selected_threshold,
        "selection_source": "validation_locked_applied_to_holdout",
        **holdout_at_selected,
    }
    _save_single_row_csv(
        NFTI_POSITIVE_XGB_HOLDOUT_80_SENSITIVITY_METRICS_PATH,
        holdout_selected_row,
    )

    holdout_sweep = [
        {**_metrics_without_ranking(splits.y_holdout, xgb_holdout_prob, float(t)), "threshold": float(t)}
        for t in HOLDOUT_THRESHOLD_GRID
    ]
    _save_csv(NFTI_POSITIVE_XGB_HOLDOUT_THRESHOLD_SWEEP_PATH, holdout_sweep)
    _plot_threshold_sweep_lines(holdout_sweep, NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_LINES_PATH)
    _plot_threshold_sweep_heatmap(holdout_sweep, NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_HEATMAP_PATH)

    calibration_rows = _calibration_bins(splits.y_holdout, xgb_holdout_prob, n_bins=10)
    _save_csv(NFTI_POSITIVE_XGB_CALIBRATION_BINS_PATH, calibration_rows)
    _plot_calibration_curve(calibration_rows, NFTI_POSITIVE_XGB_CALIBRATION_CURVE_PATH)

    decile_rows = _risk_deciles(splits.y_holdout, xgb_holdout_prob)
    _save_csv(NFTI_POSITIVE_XGB_RISK_DECILES_PATH, decile_rows)
    _plot_risk_deciles(decile_rows, NFTI_POSITIVE_XGB_RISK_DECILES_FIGURE_PATH)
    _plot_calibration_and_deciles_combined(
        calibration_rows,
        decile_rows,
        NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH,
    )
    _plot_calibration_and_deciles_combined(
        calibration_rows,
        decile_rows,
        FIGURE4_CALIBRATION_AND_DECILES_PATH,
    )

    discrimination_figure_path = (
        NFTI_POSITIVE_XGB_VS_LR_DISCRIMINATION_CURVES_FIGURE_PATH
        if lr_holdout_prob is not None
        else NFTI_POSITIVE_DISCRIMINATION_CURVES_FIGURE_PATH
    )
    _plot_discrimination_curves(
        splits.y_holdout,
        xgb_holdout_prob,
        discrimination_figure_path,
        lr_pred_prob=lr_holdout_prob,
        model_name="XGBoost",
    )

    y_holdout_clean, _, _, _ = clean_binary_eval_inputs(splits.y_holdout, xgb_holdout_prob)
    holdout_prevalence = float((y_holdout_clean == 1).mean()) if len(y_holdout_clean) else float("nan")
    decile_1_rate = decile_rows[0]["observed_nfti_rate"] if decile_rows else float("nan")
    decile_10_rate = decile_rows[-1]["observed_nfti_rate"] if decile_rows else float("nan")

    artifact_paths = {
        "feature_list": str(NFTI_POSITIVE_FEATURE_LIST_PATH),
        "xgboost_model": str(NFTI_POSITIVE_XGB_MODEL_PATH),
        "logistic_regression_model": str(NFTI_POSITIVE_LR_MODEL_PATH),
        "lr_holdout_predictions": str(NFTI_POSITIVE_LR_HOLDOUT_PREDICTIONS_PATH),
        "lr_validation_predictions": str(NFTI_POSITIVE_LR_VALIDATION_PREDICTIONS_PATH),
        "xgb_holdout_predictions": str(NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH),
        "xgb_validation_predictions": str(NFTI_POSITIVE_XGB_VALIDATION_PREDICTIONS_PATH),
        "model_comparison": str(NFTI_POSITIVE_MODEL_COMPARISON_PATH),
        "model_comparison_summary": str(NFTI_POSITIVE_MODEL_COMPARISON_SUMMARY_PATH),
        "xgb_threshold_0_5_metrics": str(NFTI_POSITIVE_XGB_THRESHOLD_0_5_METRICS_PATH),
        "xgb_validation_threshold_sweep": str(NFTI_POSITIVE_XGB_VALIDATION_THRESHOLD_SWEEP_PATH),
        "xgb_selected_80_sensitivity_threshold": str(NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH),
        "xgb_holdout_80_sensitivity_metrics": str(NFTI_POSITIVE_XGB_HOLDOUT_80_SENSITIVITY_METRICS_PATH),
        "xgb_holdout_threshold_sweep": str(NFTI_POSITIVE_XGB_HOLDOUT_THRESHOLD_SWEEP_PATH),
        "xgb_calibration_bins": str(NFTI_POSITIVE_XGB_CALIBRATION_BINS_PATH),
        "xgb_risk_deciles": str(NFTI_POSITIVE_XGB_RISK_DECILES_PATH),
        "figure_xgb_vs_lr": str(NFTI_POSITIVE_XGB_VS_LR_FIGURE_PATH),
        "figure_threshold_sweep_lines": str(NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_LINES_PATH),
        "figure_threshold_sweep_heatmap": str(NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_HEATMAP_PATH),
        "figure_calibration_curve": str(NFTI_POSITIVE_XGB_CALIBRATION_CURVE_PATH),
        "figure_risk_deciles": str(NFTI_POSITIVE_XGB_RISK_DECILES_FIGURE_PATH),
        "figure_calibration_and_deciles": str(NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH),
        "figure4_calibration_and_deciles": str(FIGURE4_CALIBRATION_AND_DECILES_PATH),
        "figure_discrimination_curves": str(discrimination_figure_path),
        "discrimination_curves_summary": str(discrimination_figure_path.with_suffix(".csv")),
    }

    evaluation_summary = (
        "nfti_positive Primary Model Evaluation Summary\n"
        f"timestamp: {timestamp}\n"
        f"cohort_size: {splits.n_cohort}\n"
        f"train_count: {len(splits.y_train)}\n"
        f"validation_count: {len(splits.y_validation)}\n"
        f"holdout_count: {len(splits.y_holdout)}\n"
        f"holdout_prevalence: {holdout_prevalence:.4f}\n"
        f"feature_count: {len(splits.feature_names)}\n"
        f"feature_list_path: {NFTI_POSITIVE_FEATURE_LIST_PATH}\n"
        f"XGBoost AUROC: {xgb_holdout_ranking['AUROC']:.4f}\n"
        f"XGBoost AUPRC: {xgb_holdout_ranking['AUPRC']:.4f}\n"
        f"XGBoost Brier: {xgb_holdout_ranking['Brier']:.4f}\n"
        f"LR AUROC: {lr_holdout_ranking['AUROC']:.4f}\n"
        f"LR AUPRC: {lr_holdout_ranking['AUPRC']:.4f}\n"
        f"LR Brier: {lr_holdout_ranking['Brier']:.4f}\n"
        f"XGBoost holdout metrics at threshold 0.5: sensitivity={xgb_threshold_0_5['sensitivity']:.4f}, "
        f"specificity={xgb_threshold_0_5['specificity']:.4f}, PPV={xgb_threshold_0_5['precision']:.4f}\n"
        f"validation_selected_80pct_sensitivity_threshold: {selected_threshold:.4f}\n"
        f"XGBoost holdout metrics at locked threshold: sensitivity={holdout_at_selected['sensitivity']:.4f}, "
        f"specificity={holdout_at_selected['specificity']:.4f}, PPV={holdout_at_selected['precision']:.4f}\n"
        f"risk_decile_observed_nfti_rate_decile_1: {decile_1_rate:.4f}\n"
        f"risk_decile_observed_nfti_rate_decile_10: {decile_10_rate:.4f}\n"
        f"discrimination_winner: {winner}\n"
        "\nOutput files:\n"
        + "\n".join(f"  {key}: {path}" for key, path in artifact_paths.items())
        + "\n"
    )
    NFTI_POSITIVE_EVALUATION_SUMMARY_PATH.write_text(evaluation_summary, encoding="utf-8")

    log_lines = [
        f"nfti_positive primary evaluation log ({timestamp})",
        f"model_type_xgboost: xgboost",
        f"model_type_lr: logistic_regression",
        f"xgboost_hyperparameters: {xgb_hyperparameters}",
        f"lr_hyperparameters: {lr_result.hyperparameters}",
        f"lr_used_imputation: {lr_result.used_imputation}",
        f"lr_imputation_strategy: {lr_result.imputation_strategy}",
        f"lr_used_scaling: {lr_result.used_scaling}",
        f"lr_class_weight_enabled: {lr_result.class_weight_enabled}",
        f"validation_threshold_selection: sweep 0.01-0.99 step 0.01 on validation only",
        f"selected_threshold: {selected_threshold:.4f}",
        f"selected_threshold_source: validation",
        f"sensitivity_warning_issued: {sensitivity_warning}",
        "artifact_paths:",
        *[f"  {key}: {path}" for key, path in artifact_paths.items()],
        "",
        comparison_summary,
        "",
        evaluation_summary,
    ]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(evaluation_summary)
    print(f"Detailed log written to {log_path}")

    return {
        "xgboost_model": xgb_model,
        "logistic_regression_pipeline": lr_result.pipeline,
        "splits": splits,
        "selected_validation_threshold": selected_threshold,
        "artifact_paths": artifact_paths,
        "log_path": str(log_path),
    }
