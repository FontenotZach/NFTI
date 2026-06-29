from __future__ import annotations

from pathlib import Path


# `NFTI/src/paths.py` -> app root is `NFTI/`
APP_ROOT = Path(__file__).resolve().parents[1]

# Data (inputs)
DATA_DIR = APP_ROOT / "data"
SCHEMAS_DIR = DATA_DIR / "schemas"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLES_DATA_DIR = DATA_DIR / "samples"

# Generated artifacts (outputs)
ARTIFACTS_DIR = APP_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MODELS_XGBOOST_DIR = MODELS_DIR / "xgboost"
MODELS_LR_DIR = MODELS_DIR / "logistic_regression"

PICKLES_DIR = ARTIFACTS_DIR / "pickles"
LOGS_DIR = ARTIFACTS_DIR / "logs"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
PREDICTIONS_ARTIFACTS_DIR = ARTIFACTS_DIR / "predictions"
EXPORT_DIR = ARTIFACTS_DIR / "export"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TABLES_DIR = ARTIFACTS_DIR / "tables"

# Structured metrics outputs (model performance summaries, etc.)
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Prehospital vital-sign fidelity audit (run before imputation/normalization)
FIDELITY_FIGURES_DIR = FIGURES_DIR / "fidelity"
FIDELITY_TABLES_DIR = TABLES_DIR / "fidelity"

# Missing-data audit (run before imputation/normalization). Separate from the
# model training pipeline; never overwrites model outputs.
MISSINGNESS_FIGURES_DIR = FIGURES_DIR / "missingness"
MISSINGNESS_TABLES_DIR = TABLES_DIR / "missingness"
MISSINGNESS_METRICS_DIR = METRICS_DIR / "missingness"

# Exploratory 2021 Field Triage Guideline "available-variable proxy" benchmark.
# Supplemental analysis run AFTER primary training (needs holdout predictions);
# never retrains the model or overwrites primary model outputs.
GUIDELINE_PROXY_TABLES_DIR = TABLES_DIR / "guideline_proxy"
GUIDELINE_PROXY_FIGURES_DIR = FIGURES_DIR / "guideline_proxy"
GUIDELINE_PROXY_CRITERION_MAPPING_PATH = (
    GUIDELINE_PROXY_TABLES_DIR / "guideline_criterion_mapping.csv"
)
GUIDELINE_PROXY_RULE_METRICS_PATH = (
    GUIDELINE_PROXY_TABLES_DIR / "guideline_proxy_rule_metrics.csv"
)
GUIDELINE_PROXY_TIER_TABLE_PATH = (
    GUIDELINE_PROXY_TABLES_DIR / "guideline_proxy_tier_nfti_rates.csv"
)
GUIDELINE_PROXY_VS_MODEL_METRICS_PATH = (
    GUIDELINE_PROXY_TABLES_DIR / "guideline_proxy_vs_model_threshold_metrics.csv"
)
GUIDELINE_PROXY_TIER_FIGURE_PATH = (
    GUIDELINE_PROXY_FIGURES_DIR / "guideline_proxy_nfti_rate_by_tier.png"
)
GUIDELINE_PROXY_BENCHMARK_SUMMARY_PATH = (
    REPORTS_DIR / "guideline_proxy_benchmark_summary.md"
)

# Definitive audit of the source headers used as model training inputs.
# (See src/reporting/training_headers.py.) This is the human-meaningful header
# list, distinct from the expanded post-one-hot model feature matrix columns.
TRAINING_HEADERS_REPORT_PATH = REPORTS_DIR / "training_headers.csv"
TRAINING_HEADERS_SUMMARY_PATH = REPORTS_DIR / "training_headers_summary.txt"

# Primary nfti_positive model artifacts
PRIMARY_CRITERION = "nfti_positive"
NFTI_POSITIVE_FEATURE_LIST_PATH = REPORTS_DIR / "nfti_positive_feature_list.csv"
NFTI_POSITIVE_LR_MODEL_PATH = MODELS_DIR / "nfti_positive_logistic_regression.pkl"
NFTI_POSITIVE_XGB_MODEL_PATH = MODELS_DIR / "nfti_positive_xgboost.json"
NFTI_POSITIVE_LR_HOLDOUT_PREDICTIONS_PATH = (
    PREDICTIONS_ARTIFACTS_DIR / "nfti_positive_logistic_regression_holdout_predictions.csv"
)
NFTI_POSITIVE_LR_VALIDATION_PREDICTIONS_PATH = (
    PREDICTIONS_ARTIFACTS_DIR / "nfti_positive_logistic_regression_validation_predictions.csv"
)
NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH = (
    PREDICTIONS_ARTIFACTS_DIR / "nfti_positive_xgboost_holdout_predictions.csv"
)
NFTI_POSITIVE_XGB_VALIDATION_PREDICTIONS_PATH = (
    PREDICTIONS_ARTIFACTS_DIR / "nfti_positive_xgboost_validation_predictions.csv"
)
NFTI_POSITIVE_MODEL_COMPARISON_PATH = (
    REPORTS_DIR / "nfti_positive_model_comparison_xgb_vs_lr.csv"
)
NFTI_POSITIVE_MODEL_COMPARISON_SUMMARY_PATH = (
    REPORTS_DIR / "nfti_positive_model_comparison_summary.txt"
)
NFTI_POSITIVE_XGB_THRESHOLD_0_5_METRICS_PATH = (
    REPORTS_DIR / "nfti_positive_xgboost_threshold_0_5_metrics.csv"
)
NFTI_POSITIVE_XGB_VALIDATION_THRESHOLD_SWEEP_PATH = (
    REPORTS_DIR / "nfti_positive_xgboost_validation_threshold_sweep.csv"
)
NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH = (
    REPORTS_DIR / "nfti_positive_xgboost_selected_80_sensitivity_threshold.csv"
)
NFTI_POSITIVE_XGB_HOLDOUT_80_SENSITIVITY_METRICS_PATH = (
    REPORTS_DIR / "nfti_positive_xgboost_holdout_80_sensitivity_threshold_metrics.csv"
)
NFTI_POSITIVE_XGB_HOLDOUT_THRESHOLD_SWEEP_PATH = (
    REPORTS_DIR / "nfti_positive_xgboost_holdout_threshold_sweep.csv"
)
NFTI_POSITIVE_XGB_CALIBRATION_BINS_PATH = (
    REPORTS_DIR / "nfti_positive_xgboost_calibration_bins.csv"
)
NFTI_POSITIVE_XGB_RISK_DECILES_PATH = REPORTS_DIR / "nfti_positive_xgboost_risk_deciles.csv"
NFTI_POSITIVE_EVALUATION_SUMMARY_PATH = REPORTS_DIR / "nfti_positive_evaluation_summary.txt"
NFTI_POSITIVE_XGB_VS_LR_FIGURE_PATH = (
    FIGURES_DIR / "nfti_positive_xgb_vs_lr_core_metrics.png"
)
NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_LINES_PATH = (
    FIGURES_DIR / "nfti_positive_xgboost_threshold_sweep_lines.png"
)
NFTI_POSITIVE_XGB_THRESHOLD_SWEEP_HEATMAP_PATH = (
    FIGURES_DIR / "nfti_positive_xgboost_threshold_sweep_heatmap.png"
)
NFTI_POSITIVE_XGB_CALIBRATION_CURVE_PATH = (
    FIGURES_DIR / "nfti_positive_xgboost_calibration_curve.png"
)
NFTI_POSITIVE_XGB_RISK_DECILES_FIGURE_PATH = (
    FIGURES_DIR / "nfti_positive_xgboost_risk_deciles.png"
)
NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH = (
    FIGURES_DIR / "nfti_positive_xgboost_calibration_and_deciles.png"
)
# Two-panel holdout discrimination figure (ROC + precision-recall). XGBoost-only
# vs. XGBoost-vs-LR filenames are selected at call time based on availability of
# logistic-regression holdout probabilities.
NFTI_POSITIVE_DISCRIMINATION_CURVES_FIGURE_PATH = (
    FIGURES_DIR / "nfti_positive_xgboost_discrimination_curves.png"
)
NFTI_POSITIVE_XGB_VS_LR_DISCRIMINATION_CURVES_FIGURE_PATH = (
    FIGURES_DIR / "nfti_positive_xgb_vs_lr_discrimination_curves.png"
)
ARCHIVE_DIR = ARTIFACTS_DIR / "archive"

# Structured evaluation outputs
RESULTS_DIR = APP_ROOT / "results"
METRICS_CSV_PATH = RESULTS_DIR / "metrics" / "model_metrics.csv"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "predictions" / "row_level_predictions.csv"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
FIGURE4_CALIBRATION_AND_DECILES_PATH = (
    RESULTS_FIGURES_DIR / "figure4_calibration_and_risk_deciles.png"
)


def ensure_dirs() -> None:
    """
    Create the standard directory structure if missing.
    Safe to call repeatedly.
    """
    for d in [
        SCHEMAS_DIR,
        RAW_DATA_DIR,
        SAMPLES_DATA_DIR,
        MODELS_XGBOOST_DIR,
        MODELS_LR_DIR,
        PICKLES_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        PREDICTIONS_ARTIFACTS_DIR,
        EXPORT_DIR,
        REPORTS_DIR,
        FIDELITY_FIGURES_DIR,
        FIDELITY_TABLES_DIR,
        MISSINGNESS_FIGURES_DIR,
        MISSINGNESS_TABLES_DIR,
        MISSINGNESS_METRICS_DIR,
        GUIDELINE_PROXY_TABLES_DIR,
        GUIDELINE_PROXY_FIGURES_DIR,
        ARCHIVE_DIR,
        RESULTS_DIR,
        RESULTS_FIGURES_DIR,
        METRICS_CSV_PATH.parent,
        PREDICTIONS_CSV_PATH.parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)

