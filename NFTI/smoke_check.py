from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd

from src.TraumaDataset import TraumaDataset
from src.preprocessing.feature_preprocessor import preprocess_data_for_criterion
from src.models.xgboost_model import train_xgboost_model
from src import Ensemble
from src.paths import SCHEMAS_DIR, SAMPLES_DATA_DIR, ensure_dirs


def load_header_definitions(csv_file_path: str) -> dict:
    headers_info = {}
    with open(csv_file_path, mode="r") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            headers_info[row["Header"]] = {
                "ntds_page": row.get("NTDS_Page", ""),
                "definition": row.get("Definition", ""),
                "timing": row.get("Timing", ""),
                "data_type": row.get("Type", ""),
                "usage": row.get("Usage", ""),
                "one_hot_grouping": row.get("One_Hot_Grouping", ""),
                "y": row.get("Y", ""),
            }
    return headers_info


def build_smoke_dataset(dataset_csv_path: str, *, n_rows: int = 250, add_custom: bool = True):
    app_dir = os.path.dirname(__file__)
    ensure_dirs()
    header_csv_path = str(SCHEMAS_DIR / "header_definitions.csv")
    customs_csv_path = str(SCHEMAS_DIR / "customs.csv")

    df = pd.read_csv(dataset_csv_path, low_memory=False).head(n_rows)
    header_info = load_header_definitions(header_csv_path)

    ds = TraumaDataset()
    for column in df.columns:
        h = header_info.get(column, {})
        ds.add_header(
            column,
            ntds_page=h.get("ntds_page", ""),
            definition=h.get("definition", ""),
            timing=h.get("timing", ""),
            data_type=h.get("data_type", ""),
            usage=h.get("usage", ""),
            one_hot_grouping=h.get("one_hot_grouping", ""),
            y=h.get("y", ""),
        )

    if add_custom:
        ds.add_custom_features(customs_csv_path)

    for _, row in df.iterrows():
        ds.add_record(row)

    return ds


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
    Xb_t, Xc_t, Xcont_t, _ = Ensemble.preprocess_data_for_criterion(criterion, ds, testing=True)
    X_test = np.column_stack((Xb_t, Xc_t, Xcont_t))
    prob = model.predict_proba(X_test)[:, 1]
    print(f"X_test shape: {X_test.shape}, prob shape: {prob.shape}")

    # Ensemble xgboost-only mode: if NN is None, meta-input should be (n_samples, n_criteria_cols)
    meta_input = prob.reshape(-1, 1)
    print(f"meta_input shape (NN disabled): {meta_input.shape}")

    # Minimal final-model shape check (untrained model is fine for this smoke test)
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    final_in = Input(shape=(meta_input.shape[1],))
    final_out = Dense(1, activation="sigmoid")(final_in)
    final_model = Model(inputs=final_in, outputs=final_out)
    _ = final_model.predict(meta_input, verbose=0)

    print("Smoke check complete (no crashes).")


if __name__ == "__main__":
    main()

