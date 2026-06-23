from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.missing_values import is_missing
from src.preprocessing.feature_preprocessor import (
    _get_feature_column_groups,
    get_continuous_columns_for_scaling,
)


def _train_mask(records) -> np.ndarray:
    return np.array([not record.for_testing for record in records], dtype=bool)


def _records_to_frame(records, columns: List[str]) -> pd.DataFrame:
    rows = [{column: record.data.get(column) for column in columns} for record in records]
    return pd.DataFrame(rows, columns=columns)


def _missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda series: series.map(is_missing))


def _nan_standard_scale(cont_df: pd.DataFrame, train_mask: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Z-score continuous columns using training-set stats; NaN positions stay NaN."""
    train_df = cont_df.loc[train_mask]
    means = train_df.mean(skipna=True).to_numpy(dtype=float)
    stds = train_df.std(skipna=True).to_numpy(dtype=float)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where((np.isnan(stds)) | (stds == 0.0), 1.0, stds)

    scaled = cont_df.sub(means, axis=1).div(stds, axis=1).to_numpy(dtype=float)

    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = stds
    scaler.var_ = stds ** 2
    scaler.n_features_in_ = len(cont_df.columns)
    scaler.n_samples_seen_ = int(train_mask.sum())
    return scaled, scaler


def _ohe_indices_by_source(one_hot_names: List[str], categorical_cols: List[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {column: [] for column in categorical_cols}
    for index, feature_name in enumerate(one_hot_names):
        source_column = _source_column_for_ohe_name(feature_name, categorical_cols)
        grouped.setdefault(source_column, []).append(index)
    return grouped


def _is_biu_column(column: str) -> bool:
    return "BIU" in column


def _apply_ohe_missing_mask(
    encoded: np.ndarray,
    missing_mask: pd.DataFrame,
    categorical_cols: List[str],
    ohe_indices_by_source: Dict[str, List[int]],
) -> np.ndarray:
    """Leave one-hot columns as NaN when the source categorical value was missing.

    BIU fields always keep strict 0/1 encoding (missing → ``__MISSING__`` column = 1).
    """
    encoded = encoded.astype(float, copy=True)
    for column in categorical_cols:
        if _is_biu_column(column):
            continue
        missing_rows = missing_mask[column].to_numpy(dtype=bool)
        if not missing_rows.any():
            continue
        for ohe_index in ohe_indices_by_source.get(column, []):
            encoded[missing_rows, ohe_index] = np.nan
    return encoded


def _store_transformed_value(record, key: str, value) -> None:
    record.data[key] = float("nan") if pd.isna(value) else value


def _store_ohe_value(record, key: str, value, *, strict_binary: bool) -> None:
    if strict_binary:
        record.data[key] = 0 if pd.isna(value) else int(value)
        return
    _store_transformed_value(record, key, value)


def _source_column_for_ohe_name(feature_name: str, categorical_cols: List[str]) -> str:
    for column in sorted(categorical_cols, key=len, reverse=True):
        prefix = f"{column}_"
        if feature_name.startswith(prefix):
            return column
    return categorical_cols[0] if categorical_cols else ""


def _records_to_frame_from_base(records, columns: List[str]) -> pd.DataFrame:
    rows = [
        {column: record.base_data.get(column) if hasattr(record, "base_data") and record.base_data is not None else record.data.get(column)
         for column in columns}
        for record in records
    ]
    return pd.DataFrame(rows, columns=columns)


def transform_trauma_dataset(trauma_dataset, *, refit: bool = True):
    """
    Z-score normalize continuous input headers and one-hot encode categorical
    input headers, persisting the result into each TraumaRecord.data.

    Raw physiologic values stay in record.base_data. Derived features are
    recomputed from base_data immediately before scaling so ratios and diffs
    never mix standardized and raw inputs.

    Fitting uses training records only (for_testing=False). Transforms are
    applied to all records. Fitted scaler/encoder state is stored on the dataset
    in ``transform_state``.
    """
    if getattr(trauma_dataset, "transform_state", None) and trauma_dataset.transform_state.get(
        "applied"
    ):
        if not refit:
            warnings.warn(
                "Dataset transforms were already applied. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            return trauma_dataset
        warnings.warn(
            "Re-fitting transforms on a dataset that was already transformed.",
            UserWarning,
            stacklevel=2,
        )

    records = trauma_dataset.get_records()
    if not records:
        raise ValueError("Cannot transform an empty TraumaDataset.")

    train_mask = _train_mask(records)
    if not train_mask.any():
        raise ValueError("No training records (for_testing=False) available to fit transforms.")

    for record in records:
        if not hasattr(record, "base_data") or record.base_data is None:
            record.base_data = dict(record.data)

    # Engineered features must use pre-scale base_data before any z-scoring.
    trauma_dataset.recalculate_custom_features()

    binary_cols, categorical_cols, _ = _get_feature_column_groups(trauma_dataset)
    continuous_cols = get_continuous_columns_for_scaling(trauma_dataset)
    header_by_name = {header.name: header for header in trauma_dataset.get_headers()}

    transform_state: Dict[str, object] = {
        "applied": True,
        "binary_columns": binary_cols,
        "continuous_columns": continuous_cols,
        "categorical_columns": categorical_cols,
    }

    # --- Continuous: z-score from base_data; write scaled values to record.data only ---
    if continuous_cols:
        cont_df = _records_to_frame_from_base(records, continuous_cols)
        cont_df = cont_df.apply(pd.to_numeric, errors="coerce")

        transformed_cont, scaler = _nan_standard_scale(cont_df, train_mask)

        for row_index, record in enumerate(records):
            for col_index, column in enumerate(continuous_cols):
                _store_transformed_value(record, column, transformed_cont[row_index, col_index])

        transform_state["continuous_scaler"] = scaler
        print(
            f"Z-score normalized {len(continuous_cols)} continuous columns "
            f"(fit on {int(train_mask.sum())} training records; base_data kept raw)."
        )
    else:
        print("No continuous columns to normalize.")

    # --- Categorical: one-hot encoding (preserve NaN on missing source values) ---
    if categorical_cols:
        missing_token = "__MISSING__"
        cat_df_raw = _records_to_frame_from_base(records, categorical_cols)
        cat_missing_mask = _missing_mask(cat_df_raw)
        cat_df = cat_df_raw.fillna(missing_token).astype(str)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(cat_df.loc[train_mask])
        encoded = encoder.transform(cat_df).astype(float)
        one_hot_names = list(encoder.get_feature_names_out(categorical_cols))
        encoded = _apply_ohe_missing_mask(
            encoded,
            cat_missing_mask,
            categorical_cols,
            _ohe_indices_by_source(one_hot_names, categorical_cols),
        )

        for column in categorical_cols:
            header = header_by_name.get(column)
            if header is not None:
                header.usage = "0"

        for feature_name in one_hot_names:
            source_column = _source_column_for_ohe_name(feature_name, categorical_cols)
            source_header = header_by_name.get(source_column)
            trauma_dataset.add_header(
                feature_name,
                ntds_page=source_header.ntds_page if source_header else "",
                definition=f"One-hot encoding of {source_column}",
                timing=source_header.timing if source_header else "1",
                data_type="1",
                usage="1",
                one_hot_grouping=source_column,
                y="",
            )

        for row_index, record in enumerate(records):
            for column in categorical_cols:
                record.data.pop(column, None)
            for col_index, feature_name in enumerate(one_hot_names):
                source_column = _source_column_for_ohe_name(feature_name, categorical_cols)
                _store_ohe_value(
                    record,
                    feature_name,
                    encoded[row_index, col_index],
                    strict_binary=_is_biu_column(source_column),
                )

        transform_state["one_hot_encoder"] = encoder
        transform_state["one_hot_columns"] = one_hot_names
        print(
            f"One-hot encoded {len(categorical_cols)} categorical columns into "
            f"{len(one_hot_names)} binary columns (BIU columns always 0/1)."
        )
    else:
        print("No categorical columns to one-hot encode.")

    trauma_dataset.transform_state = transform_state
    return trauma_dataset


def describe_transform_state(trauma_dataset) -> str:
    state = getattr(trauma_dataset, "transform_state", None) or {}
    if not state.get("applied"):
        return "No transforms have been applied to this dataset."

    lines = [
        "=== Dataset Transform State ===",
        f"Continuous columns z-scored: {len(state.get('continuous_columns', []))}",
        f"Categorical columns encoded: {len(state.get('categorical_columns', []))}",
        f"One-hot columns created: {len(state.get('one_hot_columns', []))}",
    ]
    return "\n".join(lines)
