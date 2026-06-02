from __future__ import annotations

import csv
import json
import os

import pandas as pd

from src.TraumaDataset import TraumaDataset
from src.preprocessing.feature_preprocessor import build_features_dataframe
from src.models.xgboost_model import train_xgboost_model
from src.config import TrainingConfig
from src.paths import SCHEMAS_DIR, SAMPLES_DATA_DIR, REPORTS_DIR, ensure_dirs
from src.data.temporal_derived import register_temporal_derived_headers
from src.data.clinical_derived import register_clinical_derived_headers


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

    register_temporal_derived_headers(ds, header_info)
    register_clinical_derived_headers(ds, header_info)

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

    cfg = TrainingConfig(
        random_seed=42,
        test_size=0.15,
        grid_search=False,
        cv_folds=3,
    )

    print("Training a fast XGBoost pipeline (no grid search)...")
    model = train_xgboost_model(
        ds,
        criterion,
        config=cfg,
        log_filename=None,
    )

    reports = sorted(
        REPORTS_DIR.glob(f"xgb_metrics_{criterion}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    assert reports, "Expected metrics JSON under artifacts/reports/"
    with open(reports[-1], encoding="utf-8") as f:
        rep = json.load(f)
    ts = rep["threshold_selection"]
    assert ts["threshold_selected_on"] == "train_cv_oof_predictions", (
        f"Expected OOF threshold selection; got {ts!r}. "
        "Try more rows / both classes in training."
    )
    assert ts["threshold_policy"] == "youden_j"

    print("Holdout predict_proba shape checks...")
    df = build_features_dataframe(ds)
    X_test = df[df["for_testing"]].drop(columns=["for_testing"])
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
