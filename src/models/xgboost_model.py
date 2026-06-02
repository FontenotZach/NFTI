from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.config import DEFAULT_TRAINING_CONFIG, METHODS_PREPROCESSING_LEAKAGE_SENTENCE, TrainingConfig
from src.evaluation.metrics import (
    THRESHOLD_POLICY_YOUDEN_J,
    THRESHOLD_SOURCE_TRAIN_CV_OOF,
    THRESHOLD_SOURCE_TRAIN_INTERNAL_VAL,
    classification_metrics_dict,
    format_metrics_log,
    select_threshold_youden_j,
)
from src.paths import LOGS_DIR, MODELS_XGBOOST_DIR, REPORTS_DIR, ensure_dirs
from src.preprocessing.feature_preprocessor import (
    build_features_dataframe,
    get_feature_column_groups,
    labels_for_criterion,
)
from src.preprocessing.pipeline_factory import build_xgb_classifier_pipeline
from src.splitting import ensure_assign_for_testing


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _make_stratified_kfold(
    y_train: np.ndarray, cfg: TrainingConfig
) -> Tuple[Optional[StratifiedKFold], int]:
    """Same minority-class-aware n_splits logic as GridSearchCV / OOF."""
    counts = np.bincount(y_train, minlength=2)
    min_class = int(counts.min())
    n_splits = min(cfg.cv_folds, min_class)
    if n_splits < 2:
        return None, n_splits
    return (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed),
        n_splits,
    )


def _select_threshold_train_only_no_holdout_leakage(
    best_model,
    X_train,
    y_train,
    cfg: TrainingConfig,
    cv_splitter: Optional[StratifiedKFold],
) -> Tuple[float, str, str]:
    """
    Threshold from training data only.

    Prefer out-of-fold probabilities from ``cross_val_predict`` on the training set.
    If stratified K-fold is not possible (too few minority samples), use an internal
    stratified train/validation split on training rows only.
    """
    if cv_splitter is not None:
        oof_prob = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=cv_splitter,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        thr, policy = select_threshold_youden_j(y_train, oof_prob)
        return thr, policy, THRESHOLD_SOURCE_TRAIN_CV_OOF

    # Fallback: single stratified split inside training data (still no holdout use).
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            stratify=y_train,
            random_state=cfg.random_seed,
        )
    except ValueError:
        # e.g. too few samples for stratification — conservative default
        return 0.5, THRESHOLD_POLICY_YOUDEN_J, THRESHOLD_SOURCE_TRAIN_INTERNAL_VAL

    inner = clone(best_model).fit(X_tr, y_tr)
    val_prob = inner.predict_proba(X_val)[:, 1]
    thr, policy = select_threshold_youden_j(y_val, val_prob)
    return thr, policy, THRESHOLD_SOURCE_TRAIN_INTERNAL_VAL


def train_xgboost_model(
    trauma_dataset,
    metric_to_predict: str,
    *,
    config: Optional[TrainingConfig] = None,
    param_grid: Optional[Dict] = None,
    log_filename: Optional[str] = None,
):
    """
    Train an XGBoost model behind a leakage-safe sklearn/imblearn pipeline.

    Train/holdout rows come **only** from ``TraumaRecord.for_testing`` (see ``assign_for_testing``).
    Hyperparameter search uses training rows only. Classification threshold is chosen from
    train-only out-of-fold predictions (or train-internal validation); holdout is used only
    for final scoring with that frozen threshold (plus threshold-free metrics).
    """
    cfg = config or DEFAULT_TRAINING_CONFIG
    ensure_dirs()
    ensure_assign_for_testing(trauma_dataset, config=cfg)

    df = build_features_dataframe(trauma_dataset)
    y_full = labels_for_criterion(trauma_dataset, metric_to_predict)

    train_mask = ~df["for_testing"]
    test_mask = df["for_testing"]
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "Train or holdout split is empty. Check assign_for_testing / test_size."
        )

    feature_cols = [c for c in df.columns if c != "for_testing"]
    X_train = df.loc[train_mask, feature_cols].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_train = y_full[train_mask.to_numpy()]
    y_test = y_full[test_mask.to_numpy()]

    binary_cols, categorical_cols, continuous_cols = get_feature_column_groups(trauma_dataset)
    pipeline = build_xgb_classifier_pipeline(
        binary_cols, categorical_cols, continuous_cols, cfg
    )

    grid = param_grid if param_grid is not None else cfg.xgb_param_grid

    unique_labels = np.unique(y_train)
    if unique_labels.size < 2:
        raise ValueError(
            f"Training labels for {metric_to_predict} have only one class; cannot train classifier."
        )

    cv_splitter, _ = _make_stratified_kfold(y_train, cfg)
    use_grid = bool(cfg.grid_search) and cv_splitter is not None

    if use_grid:
        grid_search_cv = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=cfg.primary_selection_metric,
            cv=cv_splitter,
            n_jobs=-1,
            refit=True,
            verbose=cfg.grid_search_verbose,
        )
        grid_search_cv.fit(X_train, y_train)
        best_model = grid_search_cv.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    thr, thr_policy, thr_source = _select_threshold_train_only_no_holdout_leakage(
        best_model, X_train, y_train, cfg, cv_splitter
    )

    y_score_test = best_model.predict_proba(X_test)[:, 1]

    # Holdout: threshold-free (ROC-AUC, AP, Brier) plus discrete metrics at frozen train-derived threshold.
    metrics_combined = classification_metrics_dict(
        y_test, y_score_test, threshold=thr
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = (
        REPORTS_DIR
        / f"xgb_metrics_{metric_to_predict}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "criterion": metric_to_predict,
        "methods_note": METHODS_PREPROCESSING_LEAKAGE_SENTENCE,
        "threshold_selection": {
            "threshold": float(thr),
            "threshold_policy": thr_policy,
            "threshold_selected_on": thr_source,
        },
        "metrics": _json_sanitize(metrics_combined),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if log_filename:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / log_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{METHODS_PREPROCESSING_LEAKAGE_SENTENCE}\n")
            f.write(
                f"threshold={thr:.6f} policy={thr_policy} selected_on={thr_source}\n"
            )
            f.write(
                format_metrics_log(
                    metrics_combined,
                    f"{metric_to_predict} XGBoost holdout (for_testing) evaluation",
                )
            )

    MODELS_XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_path = MODELS_XGBOOST_DIR / f"xgb_pipeline_{timestamp}_{metric_to_predict}.joblib"
    joblib.dump(best_model, pipeline_path)

    xgb_step = best_model.named_steps.get("xgb")
    if isinstance(xgb_step, xgb.XGBClassifier):
        json_path = MODELS_XGBOOST_DIR / f"xgboost_model_{timestamp}_{metric_to_predict}.json"
        xgb_step.save_model(str(json_path))

    return best_model
