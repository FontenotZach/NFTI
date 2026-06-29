from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.human_readable import get_label, load_human_readable_map
from src.paths import FIGURES_DIR, REPORTS_DIR, ensure_dirs
from src.preprocessing.feature_preprocessor import build_model_design_matrix

NFTI_POSITIVE_CRITERION = "nfti_positive"

# One-hot GCS columns collapsed to a single SHAP entry each (display only).
SHAP_COLLAPSE_OHE_PREFIXES: Tuple[str, ...] = (
    "EMSGCSEYE",
    "EMSGCSMOTOR",
    "EMSGCSVERBAL",
    "EMSTOTALGCS",
)

# Standard SHAP sampling defaults for tree models (background + explained rows).
SHAP_BACKGROUND_SAMPLES = 1000
SHAP_EXPLAIN_SAMPLES = 5000
SHAP_RANDOM_STATE = 42
SHAP_MAX_DISPLAY = 15


def _sample_rows(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    indices = rng.choice(len(X), size=max_rows, replace=False)
    return X[indices]


def _normalize_shap_values(shap_values) -> np.ndarray:
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _shap_ohe_group_label(feature_name: str) -> Optional[str]:
    """Map one-hot column names like EMSGCSEYE_3.0 to EMSGCSEYE."""
    for prefix in SHAP_COLLAPSE_OHE_PREFIXES:
        if feature_name == prefix or feature_name.startswith(f"{prefix}_"):
            return prefix
    return None


def collapse_shap_ohe_groups_for_display(
    feature_names: List[str],
    shap_values: np.ndarray,
    X_explain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collapse selected one-hot groups for SHAP plots/reports only.

    Member columns (e.g. EMSGCSEYE_1.0, EMSGCSEYE_2.0, ...) are summed so each
    GCS field appears once in the output.
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix width does not match feature_names length.")
    if X_explain.shape != shap_values.shape:
        raise ValueError("X_explain shape must match shap_values shape.")

    group_indices: dict[str, List[int]] = {prefix: [] for prefix in SHAP_COLLAPSE_OHE_PREFIXES}
    for index, name in enumerate(feature_names):
        label = _shap_ohe_group_label(name)
        if label is not None:
            group_indices[label].append(index)

    seen_groups: set[str] = set()
    out_names: List[str] = []
    out_shap_cols: List[np.ndarray] = []
    out_x_cols: List[np.ndarray] = []

    for index, name in enumerate(feature_names):
        label = _shap_ohe_group_label(name)
        if label is None:
            out_names.append(name)
            out_shap_cols.append(shap_values[:, index])
            out_x_cols.append(X_explain[:, index])
            continue

        if label in seen_groups:
            continue

        seen_groups.add(label)
        indices = group_indices[label]
        out_names.append(label)
        out_shap_cols.append(shap_values[:, indices].sum(axis=1))
        out_x_cols.append(X_explain[:, indices].sum(axis=1))

    return (
        np.column_stack(out_shap_cols),
        np.column_stack(out_x_cols),
        out_names,
    )


def compute_nfti_positive_xgboost_shap(
    trauma_dataset,
    model,
    *,
    background_samples: int = SHAP_BACKGROUND_SAMPLES,
    explain_samples: int = SHAP_EXPLAIN_SAMPLES,
    random_state: int = SHAP_RANDOM_STATE,
    max_display: int = SHAP_MAX_DISPLAY,
) -> Tuple[Path, Path]:
    """
    Compute SHAP values for the nfti_positive XGBoost model using TreeExplainer.

    Uses training records only (for_testing=False) to match model fitting data.
    Returns paths to the summary plot PNG and mean-|SHAP| CSV report.
    """
    ensure_dirs()

    X_train, _y_train, feature_names = build_model_design_matrix(
        trauma_dataset,
        NFTI_POSITIVE_CRITERION,
        testing=False,
    )
    if X_train.shape[0] == 0:
        raise ValueError("No training records available for SHAP analysis.")
    if X_train.shape[1] == 0:
        raise ValueError("No model features available for SHAP analysis.")

    rng = np.random.default_rng(random_state)
    background = _sample_rows(X_train, background_samples, rng)
    X_explain = _sample_rows(X_train, explain_samples, rng)

    print(
        f"Computing SHAP for {NFTI_POSITIVE_CRITERION} XGBoost "
        f"(background={len(background)}, explain={len(X_explain)})..."
    )

    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = _normalize_shap_values(explainer.shap_values(X_explain))

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP output width ({shap_values.shape[1]}) does not match feature names "
            f"({len(feature_names)})."
        )

    shap_display, X_display, feature_names_display = collapse_shap_ohe_groups_for_display(
        feature_names,
        shap_values,
        X_explain,
    )
    collapsed_groups = [g for g in SHAP_COLLAPSE_OHE_PREFIXES if g in feature_names_display]
    if collapsed_groups:
        print(
            "SHAP display: collapsed one-hot groups -> "
            + ", ".join(collapsed_groups)
        )

    hr_map = load_human_readable_map()
    display_labels = [get_label(name, hr_map) for name in feature_names_display]

    summary_path = FIGURES_DIR / "shap_nfti_positive_summary.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_display,
        X_display,
        feature_names=display_labels,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close()

    mean_abs_shap = pd.DataFrame(
        {
            "feature": feature_names_display,
            "human_readable": display_labels,
            "mean_abs_shap": np.abs(shap_display).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    report_path = REPORTS_DIR / "shap_nfti_positive_importance.csv"
    mean_abs_shap.to_csv(report_path, index=False)

    print(f"SHAP summary plot saved to {summary_path}")
    print(f"SHAP importance report saved to {report_path}")
    return summary_path, report_path
