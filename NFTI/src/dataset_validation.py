from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Sequence


def _format_name_list(names: Sequence[str], *, limit: int = 20) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... (+{len(names) - limit} more)"


def warn_schema_data_coverage(header_info: Dict[str, dict], data_columns: Iterable[str]) -> None:
    """
    Warn when schema-defined headers are missing from the data CSV, or when the
    data CSV contains columns that are not described in the schema.
    """
    data_set = set(data_columns)
    schema_names = set(header_info.keys())

    missing_from_data: List[str] = []
    for name, meta in header_info.items():
        if not meta.get("data_type"):
            continue
        if meta.get("usage") == "1" or meta.get("y") == "1":
            if name not in data_set:
                missing_from_data.append(name)

    if missing_from_data:
        warnings.warn(
            "Schema headers with Usage=1 or Y=1 are not present in the data CSV "
            f"({len(missing_from_data)}): {_format_name_list(sorted(missing_from_data))}",
            UserWarning,
            stacklevel=2,
        )

    unknown_columns = sorted(col for col in data_set if col not in schema_names)
    if unknown_columns:
        warnings.warn(
            "Data CSV columns are missing from header_definitions.csv "
            f"({len(unknown_columns)}): {_format_name_list(unknown_columns)}",
            UserWarning,
            stacklevel=2,
        )


def warn_data_column_schema_gaps(header_info: Dict[str, dict], data_columns: Iterable[str]) -> None:
    """Warn for data columns present in the CSV but missing a Type in the schema."""
    missing_type = sorted(
        column
        for column in data_columns
        if column in header_info and not header_info[column].get("data_type")
    )
    if missing_type:
        warnings.warn(
            "Data CSV columns have no Type in header_definitions.csv and will not be "
            f"registered as headers ({len(missing_type)}): "
            f"{_format_name_list(missing_type)}",
            UserWarning,
            stacklevel=2,
        )


def warn_custom_feature_availability(custom_features: Sequence[dict], headers) -> None:
    """
    Warn when a custom feature cannot be calculated because dependencies are
    missing from the dataset or will not be populated into TraumaRecord.data.
    """
    header_by_name = {header.name: header for header in headers}
    existing = set(header_by_name.keys())

    for feature in custom_features:
        feature_name = feature["header"]
        dependencies = feature["dependencies"]

        missing = [dep for dep in dependencies if dep not in existing]
        if missing:
            warnings.warn(
                f"Cannot calculate custom feature '{feature_name}': "
                f"dependencies not in dataset: {_format_name_list(missing, limit=10)}",
                UserWarning,
                stacklevel=2,
            )
            continue

        unpopulated = [
            dep
            for dep in dependencies
            if header_by_name[dep].usage != "1"
        ]
        if unpopulated:
            warnings.warn(
                f"Custom feature '{feature_name}' depends on headers that are not "
                f"populated into TraumaRecord.data (Usage=0): "
                f"{_format_name_list(unpopulated, limit=10)}",
                UserWarning,
                stacklevel=2,
            )
