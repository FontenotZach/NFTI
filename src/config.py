"""
Training / evaluation configuration for NFTI models.

Decouple hyperparameters, splitting, imputation, SMOTE, and metrics from CLI/menu code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Sentence for manuscript Methods (test-set leakage prevention).
METHODS_PREPROCESSING_LEAKAGE_SENTENCE = (
    "All preprocessing steps, including imputation, missingness indicators, encoding, "
    "resampling, and model fitting, were performed within training folds only to prevent "
    "test-set leakage."
)


@dataclass
class TrainingConfig:
    """Configuration for XGBoost pipeline training and dataset splitting."""

    random_seed: int = 42
    test_size: float = 0.15
    # Stratify the train/holdout split on this label (must exist in record.y).
    split_stratify_label: str = "nfti_positive"
    # Optional column name in record.data for disjointness assert (e.g. incident key).
    record_id_column: Optional[str] = None

    # Run assign_for_testing when pickling data (single source of truth for holdout).
    assign_split_on_pickle: bool = True

    # Imputation / indicators (inside sklearn Pipeline, fit on training folds only).
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    add_missing_indicators: bool = True

    # SMOTE (imblearn pipeline; off by default).
    use_smote: bool = False
    smote_kwargs: Dict[str, Any] = field(default_factory=lambda: {"random_state": 42})

    # Inner CV for GridSearchCV (train subset only).
    cv_folds: int = 3
    primary_selection_metric: str = "average_precision"  # GridSearchCV scoring

    # Default XGBoost grid (prefixed xgb__ when inside Pipeline).
    xgb_param_grid: Dict[str, Any] = field(
        default_factory=lambda: {
            "xgb__max_depth": [3, 5],
            "xgb__learning_rate": [0.1, 0.01],
            "xgb__n_estimators": [100, 200],
            "xgb__subsample": [1.0],
            "xgb__colsample_bytree": [0.8, 1.0],
            "xgb__gamma": [0.0, 0.2],
            "xgb__reg_lambda": [1.0, 2.0],
        }
    )

    grid_search: bool = True
    grid_search_verbose: int = 1


DEFAULT_TRAINING_CONFIG = TrainingConfig()
