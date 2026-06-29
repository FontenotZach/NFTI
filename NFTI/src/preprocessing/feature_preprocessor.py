from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_continuous_columns_for_scaling(trauma_dataset) -> List[str]:
    """
    Continuous model inputs to z-score after raw feature engineering.

    Includes Timing 1 and 2 so EMS/hospital vitals and derived features are
    scaled together; formulas always read from record.base_data (pre-scale).
    """
    return [
        header.name
        for header in trauma_dataset.get_headers()
        if header.usage == "1" and header.data_type == "3" and header.timing in ["1", "2"]
    ]


def _get_feature_column_groups(trauma_dataset) -> Tuple[List[str], List[str], List[str]]:
    binary_cols: List[str] = []
    categorical_cols: List[str] = []
    continuous_cols: List[str] = []

    # Keep the same selection logic as the legacy code:
    # - only headers with usage == "1"
    # - only timing == "1"
    # - map header.data_type to binary/categorical/continuous
    for header in trauma_dataset.get_headers():
        if header.usage == "1" and header.timing in ["1"]:
            if header.data_type == "1":
                binary_cols.append(header.name)
            elif header.data_type == "2":
                categorical_cols.append(header.name)
            elif header.data_type == "3":
                continuous_cols.append(header.name)

    return binary_cols, categorical_cols, continuous_cols


def _build_feature_frame(trauma_dataset) -> pd.DataFrame:
    # Column order must match the order used in TraumaDataset.add_record, which
    # populates record.data from headers flagged with load == "1".
    headers = [
        header.name
        for header in trauma_dataset.get_headers()
        if (header.data_type and header.load == "1")
    ]

    all_rows = [list(record.data.values()) for record in trauma_dataset.get_records()]
    return pd.DataFrame(all_rows, columns=headers)


def _extract_outcome(record, criterion: str) -> float:
    if criterion not in record.y:
        return np.nan
    val = record.y[criterion]
    if val is None:
        return np.nan
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return np.nan
    return fval if np.isfinite(fval) else np.nan


def preprocess_data_for_criterion(
    trauma_dataset,
    criterion: str,
    *,
    testing: bool,
):
    """
    Shared preprocessing used by model training and holdout evaluation.

    Returns:
      X_binary: (n_samples, n_binary)
      X_categorical: (n_samples, n_onehot)
      X_continuous: (n_samples, n_continuous)
      y: (n_samples,)
    """
    if getattr(trauma_dataset, "cohort_state", None) is None:
        warnings.warn(
            "TraumaDataset has no cohort_state; prehospital EMS cohort filtering may not "
            "have been applied before preprocessing.",
            UserWarning,
            stacklevel=2,
        )

    all_records = trauma_dataset.get_records()
    mask = np.array([r.for_testing == testing for r in all_records], dtype=bool)

    if mask.sum() == 0:
        raise ValueError(f"No records found for testing={testing} (criterion={criterion}).")

    df_all = _build_feature_frame(trauma_dataset)
    df_sub = df_all.loc[mask].copy()

    # Labels from the same record subset used for X. Missing labels remain NaN.
    y = np.array(
        [_extract_outcome(r, criterion) for r, keep in zip(all_records, mask) if keep],
        dtype=float,
    )

    binary_cols, categorical_cols, continuous_cols = _get_feature_column_groups(trauma_dataset)

    # Numeric conversion for binary/continuous.
    if binary_cols:
        X_binary = (
            df_sub[binary_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
            .values
        )
    else:
        X_binary = np.zeros((len(df_sub), 0), dtype=np.int64)

    if continuous_cols:
        X_continuous = (
            df_sub[continuous_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(float)
            .values
        )
    else:
        X_continuous = np.zeros((len(df_sub), 0), dtype=float)

    # One-hot encoding for categorical columns with stable columns across calls.
    if categorical_cols:
        missing_token = "__MISSING__"

        df_cat_all = df_all[categorical_cols].copy()
        df_cat_all = df_cat_all.fillna(missing_token).astype(str)

        df_cat_sub = df_sub[categorical_cols].copy()
        df_cat_sub = df_cat_sub.fillna(missing_token).astype(str)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df_cat_all)
        X_categorical = encoder.transform(df_cat_sub).astype(int)
    else:
        X_categorical = np.zeros((len(df_sub), 0), dtype=int)

    return X_binary, X_categorical, X_continuous, y


def filter_labeled_samples(
    *arrays: np.ndarray,
    y: np.ndarray,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Drop rows with missing outcome labels before model fitting."""
    mask = ~np.isnan(np.asarray(y, dtype=float))
    if not mask.any():
        raise ValueError("No labeled samples remain after filtering missing outcomes.")
    filtered = tuple(arr[mask] for arr in arrays)
    return filtered, y[mask].astype(int)


def print_preprocessing_sanity(
    label: str,
    X_binary: np.ndarray,
    X_categorical: np.ndarray,
    X_continuous: np.ndarray,
    y: np.ndarray,
) -> None:
    y_arr = np.asarray(y, dtype=float)
    labeled = ~np.isnan(y_arr)
    y_labeled = y_arr[labeled].astype(int) if labeled.any() else np.array([], dtype=int)
    n_pos = int((y_labeled == 1).sum()) if len(y_labeled) else 0
    n_neg = int((y_labeled == 0).sum()) if len(y_labeled) else 0
    missing_y = int((~labeled).sum())
    print(
        f"[preprocess] {label}: "
        f"X_binary={X_binary.shape}, X_categorical={X_categorical.shape}, "
        f"X_continuous={X_continuous.shape}, y={y_arr.shape}, "
        f"positives={n_pos}, negatives={n_neg}, missing_y={missing_y}"
    )


def get_model_feature_names(trauma_dataset) -> List[str]:
    """Column names in the same order as hstack(binary, categorical, continuous)."""
    binary_cols, categorical_cols, continuous_cols = _get_feature_column_groups(trauma_dataset)
    names: List[str] = list(binary_cols)

    if categorical_cols:
        df_all = _build_feature_frame(trauma_dataset)
        missing_token = "__MISSING__"
        df_cat_all = df_all[categorical_cols].fillna(missing_token).astype(str)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df_cat_all)
        names.extend(encoder.get_feature_names_out(categorical_cols))

    names.extend(continuous_cols)
    return names


def build_model_design_matrix(trauma_dataset, criterion: str, *, testing: bool):
    """Build the X matrix and feature names used by XGBoost training."""
    X_binary, X_categorical, X_continuous, y = preprocess_data_for_criterion(
        trauma_dataset, criterion, testing=testing
    )
    X = np.hstack((X_binary, X_categorical, X_continuous))
    feature_names = get_model_feature_names(trauma_dataset)
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature matrix width ({X.shape[1]}) does not match feature name count "
            f"({len(feature_names)})."
        )
    return X, y, feature_names

