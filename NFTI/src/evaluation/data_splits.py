from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation.binary_metrics import record_id_from_trauma_record
from src.preprocessing.feature_preprocessor import build_model_design_matrix


@dataclass
class NftiPositiveSplits:
    """Train / validation / holdout matrices with aligned labels and record IDs."""

    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    train_record_ids: List[str]
    X_validation: np.ndarray
    y_validation: np.ndarray
    validation_record_ids: List[str]
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    holdout_record_ids: List[str]
    n_cohort: int
    n_train_pool: int
    validation_fraction: float
    holdout_fraction: float
    random_state: int


def build_nfti_positive_splits(
    trauma_dataset,
    *,
    criterion: str = "nfti_positive",
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> NftiPositiveSplits:
    """
    Build consistent train / validation / holdout splits.

    Holdout uses record.for_testing (cohort-level holdout).
    Validation is carved from the training pool only.
    """
    all_records = trauma_dataset.get_records()
    holdout_mask = np.array([r.for_testing for r in all_records], dtype=bool)

    train_record_pairs = [(idx, record) for idx, record in enumerate(all_records) if not record.for_testing]
    holdout_record_pairs = [(idx, record) for idx, record in enumerate(all_records) if record.for_testing]

    X_train_pool, y_train_pool, feature_names = build_model_design_matrix(
        trauma_dataset, criterion, testing=False
    )
    X_holdout, y_holdout, holdout_feature_names = build_model_design_matrix(
        trauma_dataset, criterion, testing=True
    )
    if feature_names != holdout_feature_names:
        raise ValueError("Training and holdout feature names do not match.")

    train_pool_ids = [
        record_id_from_trauma_record(record, idx) for idx, record in train_record_pairs
    ]
    holdout_ids = [
        record_id_from_trauma_record(record, idx) for idx, record in holdout_record_pairs
    ]

    labeled_train_mask = ~np.isnan(np.asarray(y_train_pool, dtype=float))
    X_train_pool = X_train_pool[labeled_train_mask]
    y_train_pool = y_train_pool[labeled_train_mask].astype(int)
    train_pool_ids = [rid for rid, keep in zip(train_pool_ids, labeled_train_mask) if keep]

    labeled_holdout_mask = ~np.isnan(np.asarray(y_holdout, dtype=float))
    X_holdout = X_holdout[labeled_holdout_mask]
    y_holdout = y_holdout[labeled_holdout_mask].astype(int)
    holdout_ids = [rid for rid, keep in zip(holdout_ids, labeled_holdout_mask) if keep]

    if len(y_train_pool) == 0:
        raise ValueError("No labeled training-pool records available for nfti_positive.")
    if len(y_holdout) == 0:
        raise ValueError("No labeled holdout records available for nfti_positive.")

    X_train, X_validation, y_train, y_validation, train_ids, validation_ids = train_test_split(
        X_train_pool,
        y_train_pool,
        train_pool_ids,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y_train_pool,
    )

    holdout_fraction = float(holdout_mask.mean()) if len(all_records) else 0.0

    return NftiPositiveSplits(
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        train_record_ids=list(train_ids),
        X_validation=X_validation,
        y_validation=y_validation,
        validation_record_ids=list(validation_ids),
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        holdout_record_ids=holdout_ids,
        n_cohort=len(all_records),
        n_train_pool=len(y_train_pool),
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
        random_state=random_state,
    )
