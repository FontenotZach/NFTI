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
MODELS_KERAS_DIR = MODELS_DIR / "keras"
MODELS_XGBOOST_DIR = MODELS_DIR / "xgboost"
MODELS_ENSEMBLE_DIR = MODELS_DIR / "ensemble"
MODELS_RANDOM_FOREST_DIR = MODELS_DIR / "random_forest"

PICKLES_DIR = ARTIFACTS_DIR / "pickles"
LOGS_DIR = ARTIFACTS_DIR / "logs"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TUNING_DIR = ARTIFACTS_DIR / "tuning"
EXPORT_DIR = ARTIFACTS_DIR / "export"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
ARCHIVE_DIR = ARTIFACTS_DIR / "archive"

# Structured evaluation outputs
RESULTS_DIR = APP_ROOT / "results"
METRICS_CSV_PATH = RESULTS_DIR / "metrics" / "model_metrics.csv"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "predictions" / "row_level_predictions.csv"


def ensure_dirs() -> None:
    """
    Create the standard directory structure if missing.
    Safe to call repeatedly.
    """
    for d in [
        SCHEMAS_DIR,
        RAW_DATA_DIR,
        SAMPLES_DATA_DIR,
        MODELS_KERAS_DIR,
        MODELS_XGBOOST_DIR,
        MODELS_ENSEMBLE_DIR,
        MODELS_RANDOM_FOREST_DIR,
        PICKLES_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        TUNING_DIR,
        EXPORT_DIR,
        REPORTS_DIR,
        ARCHIVE_DIR,
        RESULTS_DIR,
        METRICS_CSV_PATH.parent,
        PREDICTIONS_CSV_PATH.parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)

