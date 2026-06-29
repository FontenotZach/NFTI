from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data.dataset_builder import build_trauma_dataset, load_header_definitions
from src.preprocessing.feature_preprocessor import preprocess_data_for_criterion
from src.models.xgboost_model import train_xgboost_model
from src.paths import SCHEMAS_DIR, SAMPLES_DATA_DIR, ensure_dirs


def build_smoke_dataset(dataset_csv_path: str, *, n_rows: int = 250, add_custom: bool = True):
    ensure_dirs()
    header_info = load_header_definitions(str(SCHEMAS_DIR / "header_definitions.csv"))
    customs_path = str(SCHEMAS_DIR / "customs.csv") if add_custom else None

    df = pd.read_csv(dataset_csv_path, low_memory=False).head(n_rows)
    return build_trauma_dataset(
        df, header_info, customs_path=customs_path, write_cohort_report=False
    )


def main():
    ensure_dirs()
    dataset_csv_path = str(SAMPLES_DATA_DIR / "dat5_limited.csv")

    if not os.path.exists(dataset_csv_path):
        raise FileNotFoundError(f"Missing dataset CSV: {dataset_csv_path}")

    print("Building smoke dataset...")
    ds = build_smoke_dataset(dataset_csv_path, n_rows=200, add_custom=True)
    print(ds)

    criterion = "nfti_positive"

    print("Preprocessing (training subset)...")
    Xb, Xc, Xcont, y = preprocess_data_for_criterion(ds, criterion, testing=False)
    X_train = np.hstack((Xb, Xc, Xcont))
    print(f"X_train shape: {X_train.shape}, y shape: {y.shape}")

    print("Training a fast XGBoost model (no grid search)...")
    model = train_xgboost_model(
        ds,
        criterion,
        grid_search=False,
        use_smote=False,
        test_size=0.2,
        random_state=42,
        log_filename=None,
    )

    print("Preprocessing (testing subset) + XGBoost predict_proba shape checks...")
    Xb_t, Xc_t, Xcont_t, _ = preprocess_data_for_criterion(ds, criterion, testing=True)
    X_test = np.hstack((Xb_t, Xc_t, Xcont_t))
    prob = model.predict_proba(X_test)[:, 1]
    print(f"X_test shape: {X_test.shape}, prob shape: {prob.shape}")

    print("Smoke check complete (no crashes).")


if __name__ == "__main__":
    main()
