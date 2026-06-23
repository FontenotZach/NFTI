from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.paths import FIGURES_DIR, REPORTS_DIR, ensure_dirs
from src.preprocessing.feature_preprocessor import build_model_design_matrix

NFTI_POSITIVE_CRITERION = "nfti_positive"

# Standard SHAP sampling defaults for tree models (background + explained rows).
SHAP_BACKGROUND_SAMPLES = 100
SHAP_EXPLAIN_SAMPLES = 100
SHAP_RANDOM_STATE = 42
SHAP_MAX_DISPLAY = 20


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

    summary_path = FIGURES_DIR / "shap_nfti_positive_summary.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close()

    mean_abs_shap = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    report_path = REPORTS_DIR / "shap_nfti_positive_importance.csv"
    mean_abs_shap.to_csv(report_path, index=False)

    print(f"SHAP summary plot saved to {summary_path}")
    print(f"SHAP importance report saved to {report_path}")
    return summary_path, report_path
