from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.paths import PICKLES_DIR, ensure_dirs


def get_imputation_feature_columns(trauma_dataset) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Columns to impute: Usage=1, Timing=1, excluding BIU headers.
    Returns (all_feature_cols, binary_cols, categorical_cols, continuous_cols).
    """
    binary_cols: List[str] = []
    categorical_cols: List[str] = []
    continuous_cols: List[str] = []

    for header in trauma_dataset.get_headers():
        if "BIU" in header.name:
            continue
        if header.usage != "1" or header.timing not in ["1"]:
            continue
        if header.data_type == "1":
            binary_cols.append(header.name)
        elif header.data_type == "2":
            categorical_cols.append(header.name)
        elif header.data_type == "3":
            continuous_cols.append(header.name)

    feature_cols = binary_cols + categorical_cols + continuous_cols
    return feature_cols, binary_cols, categorical_cols, continuous_cols


def _features_frame(records, feature_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame([record.data for record in records])
    if not feature_cols:
        return pd.DataFrame()
    return df[feature_cols].apply(pd.to_numeric, errors="coerce")


def _apply_imputed_values(
    df_full: pd.DataFrame,
    imputed_columns,
    imputed: np.ndarray,
    binary_cols: List[str],
    categorical_cols: List[str],
) -> None:
    imputed_df = pd.DataFrame(imputed, columns=imputed_columns, index=df_full.index)
    discrete = set(binary_cols + categorical_cols)

    for column in imputed_columns:
        series = imputed_df[column]
        if column in discrete:
            series = series.round().astype(int)
        df_full[column] = series


def _sync_imputed_record(record, row, feature_cols: List[str]) -> None:
    record.data = row.to_dict()
    if not hasattr(record, "base_data") or record.base_data is None:
        record.base_data = dict(record.data)
        return
    for column in feature_cols:
        if column in row:
            record.base_data[column] = row[column]


def impute_trauma_dataset_mice(
    trauma_dataset,
    *,
    max_iter: int = 10,
    random_state: int = 42,
    save_path: Path | None = None,
):
    """
    Impute missing values with MICE (sklearn IterativeImputer).

    The imputer is fit on training records only (for_testing=False) and applied
    to both training and test records to avoid leakage.
    """
    records = trauma_dataset.get_records()
    if not records:
        raise ValueError("Cannot impute an empty TraumaDataset.")

    if getattr(trauma_dataset, "cohort_state", None) is None:
        raise ValueError(
            "Apply the prehospital EMS cohort filter before MICE imputation."
        )

    if getattr(trauma_dataset, "transform_state", None) and trauma_dataset.transform_state.get(
        "applied"
    ):
        raise ValueError(
            "MICE imputation must run before normalize/encode (option 10). "
            "Reload a pre-transform pickle and run option 9 first."
        )

    train_records = [record for record in records if not record.for_testing]
    test_records = [record for record in records if record.for_testing]

    if not train_records:
        raise ValueError("No training records (for_testing=False) available to fit MICE.")

    feature_cols, binary_cols, categorical_cols, _continuous_cols = get_imputation_feature_columns(
        trauma_dataset
    )
    if not feature_cols:
        raise ValueError("No model input columns selected for imputation.")

    trauma_dataset.review_and_adjust_for_biu()

    df_train = pd.DataFrame([record.data for record in train_records])
    df_test = pd.DataFrame([record.data for record in test_records]) if test_records else None

    train_features = _features_frame(train_records, feature_cols)

    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
        keep_empty_features=True,
    )
    imputer.fit(train_features)

    train_imputed = imputer.transform(train_features)
    _apply_imputed_values(df_train, imputer.feature_names_in_, train_imputed, binary_cols, categorical_cols)

    if test_records and df_test is not None:
        test_features = _features_frame(test_records, feature_cols)
        test_imputed = imputer.transform(test_features)
        _apply_imputed_values(df_test, imputer.feature_names_in_, test_imputed, binary_cols, categorical_cols)
        for i, record in enumerate(test_records):
            _sync_imputed_record(record, df_test.iloc[i], feature_cols)

    for i, record in enumerate(train_records):
        _sync_imputed_record(record, df_train.iloc[i], feature_cols)

    trauma_dataset.records = train_records + test_records
    trauma_dataset.imputation_state = {
        "method": "mice",
        "max_iter": max_iter,
        "random_state": random_state,
        "feature_columns": feature_cols,
        "train_record_count": len(train_records),
        "test_record_count": len(test_records),
    }
    trauma_dataset.review_and_adjust_for_biu()
    trauma_dataset.recalculate_custom_features()

    if save_path is not None:
        ensure_dirs()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(trauma_dataset, f)

    return trauma_dataset
