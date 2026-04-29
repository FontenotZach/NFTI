"""
Single source of truth for train/holdout assignment via TraumaRecord.for_testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

if TYPE_CHECKING:
    from src.config import TrainingConfig


def _stratify_labels_for_split(y: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 for stratification (binary labels)."""
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    return y.astype(int)


def assign_for_testing(trauma_dataset, *, config: "TrainingConfig") -> None:
    """
    Set record.for_testing using one stratified shuffle split.

    Mutates records in place. Call once after all records are loaded.
    """
    records = trauma_dataset.get_records()
    if not records:
        raise ValueError("assign_for_testing: no records in dataset.")

    key = config.split_stratify_label
    y_split = np.array([r.y.get(key, np.nan) for r in records], dtype=float)
    y_strat = _stratify_labels_for_split(y_split)

    n_splits = 1
    sss = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    train_idx, test_idx = next(sss.split(np.zeros((len(records), 1)), y_strat))
    test_set: Set[int] = set(test_idx.tolist())

    for i, r in enumerate(records):
        r.for_testing = i in test_set

    setattr(trauma_dataset, "_holdout_assigned", True)

    assert_disjoint_train_test_ids(trauma_dataset, train_idx, test_idx, config=config)


def assert_disjoint_train_test_ids(
    trauma_dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    config: "TrainingConfig",
) -> None:
    """Guardrail: train and test ID sets must not overlap when record_id_column is set."""
    col = config.record_id_column
    if not col:
        return

    records = trauma_dataset.get_records()
    train_ids = [records[i].data.get(col) for i in train_idx]
    test_ids = [records[i].data.get(col) for i in test_idx]

    # Skip assert if IDs are missing everywhere
    if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in train_ids + test_ids):
        return

    set_train = {x for x in train_ids if x is not None and not (isinstance(x, float) and np.isnan(x))}
    set_test = {x for x in test_ids if x is not None and not (isinstance(x, float) and np.isnan(x))}

    overlap = set_train & set_test
    if overlap:
        raise ValueError(
            f"Train/test ID overlap on column {col!r}: {len(overlap)} overlapping values "
            f"(example: {next(iter(overlap))!r})."
        )


def ensure_assign_for_testing(trauma_dataset, *, config: "TrainingConfig") -> None:
    """
    Ensure train and holdout are both non-empty.

    Assigns using ``assign_for_testing`` only when the split is missing or degenerate
    (e.g. fresh records all ``for_testing=False``).
    """
    records = trauma_dataset.get_records()
    if not records:
        raise ValueError("ensure_assign_for_testing: empty dataset.")

    n_test = sum(1 for r in records if r.for_testing)
    n_train = len(records) - n_test
    if n_train == 0 or n_test == 0:
        assign_for_testing(trauma_dataset, config=config)
