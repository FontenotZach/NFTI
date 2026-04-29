from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def _get_feature_column_groups(trauma_dataset) -> Tuple[List[str], List[str], List[str]]:
    binary_cols: List[str] = []
    categorical_cols: List[str] = []
    continuous_cols: List[str] = []

    for header in trauma_dataset.get_headers():
        if header.usage == "1" and header.timing in ["1"]:
            if header.data_type == "1":
                binary_cols.append(header.name)
            elif header.data_type == "2":
                categorical_cols.append(header.name)
            elif header.data_type == "3":
                continuous_cols.append(header.name)

    return binary_cols, categorical_cols, continuous_cols


def get_feature_column_groups(trauma_dataset):
    """Public alias for column grouping used by pipelines and legacy paths."""
    return _get_feature_column_groups(trauma_dataset)


def _build_raw_record_feature_frame(trauma_dataset) -> pd.DataFrame:
    """Feature columns only (all usage==1 with data_type); row order matches get_records()."""
    headers = [
        header.name
        for header in trauma_dataset.get_headers()
        if (header.data_type and header.usage == "1")
    ]
    all_rows = [list(record.data.values()) for record in trauma_dataset.get_records()]
    return pd.DataFrame(all_rows, columns=headers)


def build_features_dataframe(trauma_dataset) -> pd.DataFrame:
    """
    Build a feature matrix as a DataFrame aligned with record order.

    - Numeric columns: NaNs preserved where missing in source data.
    - Categorical columns: strings or NaN.
    - Includes ``for_testing`` for train/holdout masking (single source of truth).
    """
    binary_cols, categorical_cols, continuous_cols = _get_feature_column_groups(trauma_dataset)
    rows = []
    for record in trauma_dataset.get_records():
        row: dict = {}
        for c in binary_cols:
            v = record.data.get(c, np.nan)
            row[c] = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        for c in continuous_cols:
            v = record.data.get(c, np.nan)
            row[c] = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        for c in categorical_cols:
            v = record.data.get(c, np.nan)
            row[c] = np.nan if pd.isna(v) else str(v)
        row["for_testing"] = bool(record.for_testing)
        rows.append(row)
    feat_cols = binary_cols + categorical_cols + continuous_cols
    df = pd.DataFrame(rows)
    return df[feat_cols + ["for_testing"]]


def labels_for_criterion(trauma_dataset, criterion: str) -> np.ndarray:
    """Binary labels aligned with get_records() order."""
    y = np.array([r.y.get(criterion, np.nan) for r in trauma_dataset.get_records()])
    return np.nan_to_num(y, nan=0.0).astype(int)


def preprocess_data_for_criterion(
    trauma_dataset,
    criterion: str,
    *,
    testing: bool,
):
    """
    Legacy path: numpy blocks for non-pipeline code.

    One-hot encoder is **fitted only on training rows** (not for_testing), then applied
    to the requested subset — avoids leakage from held-out rows into category vocabulary.

    Returns:
      X_binary, X_categorical, X_continuous, y
    """
    df = build_features_dataframe(trauma_dataset)
    mask = df["for_testing"].eq(testing)
    if mask.sum() == 0:
        raise ValueError(f"No records found for testing={testing} (criterion={criterion}).")

    df_sub = df.loc[mask].drop(columns=["for_testing"]).copy()
    df_train = df.loc[~df["for_testing"]].drop(columns=["for_testing"]).copy()

    all_records = trauma_dataset.get_records()
    y = np.array(
        [r.y.get(criterion, 0) for r, keep in zip(all_records, df["for_testing"].eq(testing)) if keep]
    )
    y = np.nan_to_num(y, nan=0).astype(int)

    binary_cols, categorical_cols, continuous_cols = _get_feature_column_groups(trauma_dataset)

    if binary_cols:
        X_binary = (
            df_sub[binary_cols]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
            .values
        )
    else:
        X_binary = np.zeros((len(df_sub), 0), dtype=float)

    if continuous_cols:
        X_continuous = (
            df_sub[continuous_cols]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
            .values
        )
    else:
        X_continuous = np.zeros((len(df_sub), 0), dtype=float)

    if categorical_cols:
        missing_token = "__MISSING__"
        df_cat_train = df_train[categorical_cols].copy()
        df_cat_train = df_cat_train.fillna(missing_token).astype(str)

        df_cat_sub = df_sub[categorical_cols].copy()
        df_cat_sub = df_cat_sub.fillna(missing_token).astype(str)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df_cat_train)
        X_categorical = encoder.transform(df_cat_sub).astype(int)
    else:
        X_categorical = np.zeros((len(df_sub), 0), dtype=int)

    return X_binary, X_categorical, X_continuous, y
