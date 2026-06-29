"""Shared TraumaDataset construction helpers.

Centralizes the dataset-build logic that was previously duplicated between
``app.py`` (interactive pickling) and ``smoke_check.py`` so both paths build the
analytic cohort identically: load header schema, register headers and custom
features, populate records, apply the prehospital EMS cohort filter, and assign
the stable train/validation/holdout split.
"""

from __future__ import annotations

import csv
from typing import Optional

import pandas as pd

from src.TraumaDataset import TraumaDataset
from src.preprocessing.cohort_filter import (
    apply_prehospital_ems_cohort_filter_to_dataset,
)


def load_header_definitions(csv_file_path: str) -> dict:
    """Load the header schema (``data/schemas/header_definitions.csv``)."""
    headers_info: dict = {}
    with open(csv_file_path, mode="r") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            headers_info[row["Header"]] = {
                "ntds_page": row.get("NTDS_Page", ""),
                "definition": row.get("Definition", ""),
                "timing": row.get("Timing", ""),
                "data_type": row.get("Type", ""),
                "load": row.get("Load", ""),
                "usage": row.get("Usage", ""),
                "y": row.get("Y", ""),
            }
    return headers_info


def build_trauma_dataset(
    df: pd.DataFrame,
    header_info: dict,
    *,
    customs_path: Optional[str] = None,
    write_cohort_report: bool = True,
    random_state: int = 42,
) -> TraumaDataset:
    """Build a cohort-filtered, split-assigned TraumaDataset from a raw frame.

    Steps (identical for the interactive pipeline and the smoke test):
      1. Register every column as a Header using the schema metadata.
      2. Register custom/derived features when a customs CSV is provided.
      3. Validate the build against the schema and data columns.
      4. Populate one TraumaRecord per row (split assigned later).
      5. Apply the prehospital EMS cohort filter in place.
      6. Assign the train/validation/holdout split with a stable seed.
    """
    dataset = TraumaDataset()

    for column in df.columns:
        details = header_info.get(column, {})
        dataset.add_header(
            column,
            ntds_page=details.get("ntds_page", ""),
            definition=details.get("definition", ""),
            timing=details.get("timing", ""),
            data_type=details.get("data_type", ""),
            load=details.get("load", ""),
            usage=details.get("usage", ""),
            y=details.get("y", ""),
        )

    if customs_path:
        dataset.add_custom_features(customs_path)

    dataset.validate_build(header_info, df.columns)

    for _, row in df.iterrows():
        dataset.add_record(row, assign_split=False)

    apply_prehospital_ems_cohort_filter_to_dataset(
        dataset, write_report=write_cohort_report
    )
    dataset.assign_train_test_split(random_state=random_state)

    return dataset
