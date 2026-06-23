from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.preprocessing.feature_preprocessor import filter_labeled_samples, preprocess_data_for_criterion
from src.paths import LOGS_DIR, MODELS_XGBOOST_DIR, ensure_dirs


def _label_counts(y) -> Tuple[int, int]:
    y_arr = np.asarray(y, dtype=float)
    labeled = y_arr[~np.isnan(y_arr)].astype(int)
    return int((labeled == 1).sum()), int((labeled == 0).sum())


def _can_train_binary_classifier(y, *, cv_folds: int = 3) -> Optional[str]:
    positive, negative = _label_counts(y)
    if positive == 0 or negative == 0:
        return (
            f"need both classes in training data "
            f"(positive={positive}, negative={negative})"
        )
    if min(positive, negative) < cv_folds:
        return (
            f"minority class too small for {cv_folds}-fold CV "
            f"(positive={positive}, negative={negative})"
        )
    return None


def train_xgboost_model(
    trauma_dataset,
    metric_to_predict: str,
    *,
    grid_search: bool = True,
    param_grid: Optional[Dict] = None,
    use_smote: bool = False,
    test_size: float = 0.15,
    random_state: int = 42,
    log_filename: Optional[str] = None,
):
    ensure_dirs()

    """
    Train an XGBoost model for one criterion.

    Returns:
      fitted xgboost model
    """

    X_binary, X_categorical, X_continuous, y = preprocess_data_for_criterion(
        trauma_dataset, metric_to_predict, testing=False
    )
    (X_binary, X_categorical, X_continuous), y = filter_labeled_samples(
        X_binary, X_categorical, X_continuous, y=y
    )
    X_data = np.hstack((X_binary, X_categorical, X_continuous))

    skip_reason = _can_train_binary_classifier(y)
    if skip_reason:
        print(f"Skipping {metric_to_predict}: {skip_reason}.")
        return None
        
    if param_grid is None:
        param_grid = {
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [300],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0],
            "reg_lambda": [1],
        }

    X_resampled, y_resampled = X_data, y
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_data, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_resampled,
    )

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )

    try:
        if grid_search:
            stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            grid_search_cv = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=stratified_kfold,
                n_jobs=-1,
                verbose=2,
            )
            grid_search_cv.fit(X_train, y_train)
            best_model = grid_search_cv.best_estimator_
        else:
            xgb_model.fit(X_train, y_train)
            best_model = xgb_model
    except ValueError as exc:
        print(f"Skipping {metric_to_predict}: training failed ({exc}).")
        return None

    # Internal evaluation on the train_test_split holdout.
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        roc_auc = float("nan")
        print(
            f"Warning: ROC AUC undefined for {metric_to_predict} on holdout split "
            f"(positive={int((y_test == 1).sum())}, negative={int((y_test == 0).sum())})."
        )

    if log_filename:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        roc_auc_text = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A (single class in holdout)"
        with open(LOGS_DIR / log_filename, "a") as f:
            f.write(
                "\n"
                f"--- {metric_to_predict} XGBoost Test Set Model Evaluation ---\n"
                f"Accuracy: {accuracy * 100:.2f}%\n"
                f"Precision: {precision * 100:.2f}%\n"
                f"Recall (Sensitivity): {recall * 100:.2f}%\n"
                f"F1 Score: {f1 * 100:.2f}%\n"
                f"ROC AUC: {roc_auc_text}\n"
            )

    # Save the trained XGBoost model.
    MODELS_XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_XGBOOST_DIR / f"xgboost_model_{timestamp}_{metric_to_predict}.json"
    best_model.save_model(str(model_path))

    return best_model

