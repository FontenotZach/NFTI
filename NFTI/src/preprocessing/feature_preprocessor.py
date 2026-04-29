from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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
    # Column order must match the order used in TraumaDataset.add_record.
    headers = [
        header.name
        for header in trauma_dataset.get_headers()
        if (header.data_type and header.usage == "1")
    ]

    all_rows = [list(record.data.values()) for record in trauma_dataset.get_records()]
    return pd.DataFrame(all_rows, columns=headers)


def preprocess_data_for_criterion(
    trauma_dataset,
    criterion: str,
    *,
    testing: bool,
):
    """
    Shared preprocessing used by training and ensemble evaluation.

    Returns:
      X_binary: (n_samples, n_binary)
      X_categorical: (n_samples, n_onehot)
      X_continuous: (n_samples, n_continuous)
      y: (n_samples,)
    """

    all_records = trauma_dataset.get_records()
    mask = np.array([r.for_testing == testing for r in all_records], dtype=bool)

    if mask.sum() == 0:
        raise ValueError(f"No records found for testing={testing} (criterion={criterion}).")

    df_all = _build_feature_frame(trauma_dataset)
    df_sub = df_all.loc[mask].copy()

    # Labels from the same record subset used for X.
    y = np.array([r.y.get(criterion, 0) for r, keep in zip(all_records, mask) if keep])
    y = np.nan_to_num(y, nan=0).astype(int)

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

