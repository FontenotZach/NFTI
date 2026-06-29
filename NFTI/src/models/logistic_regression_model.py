from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.paths import NFTI_POSITIVE_LR_MODEL_PATH, ensure_dirs


@dataclass
class LogisticRegressionTrainingResult:
    pipeline: Pipeline
    model_path: str
    used_imputation: bool
    imputation_strategy: str
    used_scaling: bool
    class_weight_enabled: bool
    hyperparameters: Dict[str, Any]


def _needs_imputation(X: np.ndarray) -> bool:
    return bool(np.isnan(X).any())


def train_logistic_regression_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    class_weight_balanced: bool = True,
    max_iter: int = 2000,
    random_state: int = 42,
    model_path: Optional[str] = None,
) -> LogisticRegressionTrainingResult:
    """
    Train logistic regression on the training split only.

    Imputation (median) and scaling are fit on X_train only when needed / always
    for scaling respectively.
    """
    ensure_dirs()
    out_path = model_path or str(NFTI_POSITIVE_LR_MODEL_PATH)

    use_imputation = _needs_imputation(X_train)
    imputation_strategy = "median" if use_imputation else "none"

    steps = []
    if use_imputation:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced" if class_weight_balanced else None,
                solver="lbfgs",
                random_state=random_state,
            ),
        )
    )

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    hyperparameters = {
        "max_iter": max_iter,
        "solver": "lbfgs",
        "class_weight": "balanced" if class_weight_balanced else None,
        "random_state": random_state,
        "imputation_strategy": imputation_strategy,
        "scaling": "standard",
    }

    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)

    return LogisticRegressionTrainingResult(
        pipeline=pipeline,
        model_path=out_path,
        used_imputation=use_imputation,
        imputation_strategy=imputation_strategy,
        used_scaling=True,
        class_weight_enabled=class_weight_balanced,
        hyperparameters=hyperparameters,
    )


def predict_proba_positive(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]
