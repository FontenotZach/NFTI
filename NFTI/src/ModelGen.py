from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.evaluation.binary_metrics import (
    clean_binary_eval_inputs,
    evaluate_binary_classifier,
    format_metrics_for_log,
    record_id_from_trauma_record,
    save_metrics_row,
    save_row_level_predictions,
    select_threshold_f1,
    select_threshold_youden,
)
from src.paths import FIGURES_DIR, LOGS_DIR, ensure_dirs
from src.preprocessing.feature_preprocessor import (
    _extract_outcome,
    preprocess_data_for_criterion as preprocess_data_for_criterion_shared,
    print_preprocessing_sanity,
)

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def _holdout_record_ids(trauma_dataset) -> list:
    ids = []
    for idx, record in enumerate(trauma_dataset.get_records()):
        if record.for_testing:
            ids.append(record_id_from_trauma_record(record, idx))
    return ids


def _run_eval_and_save(
    *,
    y_true,
    y_prob,
    record_ids,
    timestamp: str,
    model_type: str,
    metric_to_predict: str,
    split_name: str,
    threshold_type: str,
    threshold: float,
    model_path: str,
    log_filename_value: str,
    save_predictions: bool,
) -> dict:
    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=threshold)
    row = {
        "timestamp": timestamp,
        "model_type": model_type,
        "metric_to_predict": metric_to_predict,
        "split_name": split_name,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "model_path": model_path,
        "log_filename": log_filename_value,
        **metrics,
    }
    save_metrics_row(row)

    log_text = format_metrics_for_log(row)
    print(log_text)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / log_filename_value, "a", encoding="utf-8") as log_file:
        log_file.write(log_text)

    if save_predictions:
        y_clean, p_clean, ids_clean, _ = clean_binary_eval_inputs(y_true, y_prob, record_ids)
        pred_rows = []
        y_pred = (p_clean >= threshold).astype(int)
        for rid, yt, yp, yhat in zip(ids_clean, y_clean, p_clean, y_pred):
            pred_rows.append(
                {
                    "record_id": rid,
                    "y_true": int(yt),
                    "y_pred_prob": float(yp),
                    "y_pred": int(yhat),
                    "timestamp": timestamp,
                    "model_type": model_type,
                    "metric_to_predict": metric_to_predict,
                    "split_name": split_name,
                }
            )
        if pred_rows:
            save_row_level_predictions(pred_rows)

    return row


def train_trauma_model_xgboost(trauma_dataset, metric_to_predict):
    global log_filename
    from src.models.xgboost_model import train_xgboost_model

    best_xgb_model = train_xgboost_model(
        trauma_dataset,
        metric_to_predict,
        log_filename=log_filename,
    )
    if best_xgb_model is None:
        return None, None, None

    holdout_result = test_all_testing_records_xgboost(
        best_xgb_model, trauma_dataset, metric_to_predict
    )
    if holdout_result is None:
        return best_xgb_model, None, None
    X_holdout, y_holdout = holdout_result
    return best_xgb_model, X_holdout, y_holdout


def plot_roc_curve(best_xgb_model, X_test, y_test, metric_to_predict):
    y_pred_prob = best_xgb_model.predict_proba(X_test)[:, 1]
    y_eval, p_eval, _, _ = clean_binary_eval_inputs(y_test, y_pred_prob)
    if len(y_eval) == 0 or len(np.unique(y_eval)) < 2:
        print(
            f"Skipping ROC plot for {metric_to_predict}: holdout has only one labeled class "
            f"(positive={int((y_eval == 1).sum())}, negative={int((y_eval == 0).sum())})."
        )
        return

    fpr, tpr, _ = roc_curve(y_eval, p_eval)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic for {metric_to_predict}")
    plt.legend(loc="lower right")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURES_DIR / f"{metric_to_predict}_ROC_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve plot saved to {plot_path}")


def test_with_threshold(
    threshold,
    threshold_name,
    metric_to_predict,
    testing_records,
    y_pred_prob,
    *,
    model_type="xgboost",
    split_name="holdout",
    model_path="",
    timestamp=None,
    record_ids=None,
):
    global log_filename

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if record_ids is None:
        record_ids = [
            record_id_from_trauma_record(record, idx)
            for idx, record in enumerate(testing_records)
        ]

    y_true = np.array([_extract_outcome(record, metric_to_predict) for record in testing_records], dtype=float)

    save_predictions = threshold_name == "fixed_0.5"
    _run_eval_and_save(
        y_true=y_true,
        y_prob=y_pred_prob,
        record_ids=record_ids,
        timestamp=timestamp,
        model_type=model_type,
        metric_to_predict=metric_to_predict,
        split_name=split_name,
        threshold_type=threshold_name,
        threshold=float(threshold),
        model_path=model_path,
        log_filename_value=log_filename,
        save_predictions=save_predictions,
    )


def test_all_testing_records_xgboost(model, trauma_dataset, metric_to_predict, *, model_path=""):
    global log_filename

    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]
    if not testing_records:
        print("No testing records available.")
        return None

    print(f"\n--- Testing on {len(testing_records)} Records ---\n")

    X_binary, X_categorical, X_continuous, y_actual = preprocess_data_for_criterion_shared(
        trauma_dataset, metric_to_predict, testing=True
    )
    print_preprocessing_sanity("holdout", X_binary, X_categorical, X_continuous, y_actual)
    X_test_data = np.hstack((X_binary, X_categorical, X_continuous))

    y_pred_prob = model.predict_proba(X_test_data)[:, 1]
    record_ids = _holdout_record_ids(trauma_dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    test_with_threshold(
        0.5,
        "fixed_0.5",
        metric_to_predict,
        testing_records,
        y_pred_prob,
        model_type="xgboost",
        split_name="holdout",
        model_path=model_path,
        timestamp=timestamp,
        record_ids=record_ids,
    )

    y_clean, _, _, _ = clean_binary_eval_inputs(y_actual, y_pred_prob)
    if len(np.unique(y_clean)) >= 2:
        posthoc_f1 = select_threshold_f1(y_actual, y_pred_prob)
        posthoc_youden = select_threshold_youden(y_actual, y_pred_prob)
        print(f"Post-hoc holdout threshold (f1): {posthoc_f1:.4f}")
        print(f"Post-hoc holdout threshold (youden): {posthoc_youden:.4f}")

        test_with_threshold(
            posthoc_f1,
            "posthoc_holdout_f1",
            metric_to_predict,
            testing_records,
            y_pred_prob,
            model_type="xgboost",
            split_name="holdout",
            model_path=model_path,
            timestamp=timestamp,
            record_ids=record_ids,
        )
        test_with_threshold(
            posthoc_youden,
            "posthoc_holdout_youden",
            metric_to_predict,
            testing_records,
            y_pred_prob,
            model_type="xgboost",
            split_name="holdout",
            model_path=model_path,
            timestamp=timestamp,
            record_ids=record_ids,
        )
    else:
        print(
            f"Holdout for {metric_to_predict} has only one labeled class after cleaning; "
            "skipping post-hoc thresholds."
        )

    plot_roc_curve(model, X_test_data, y_actual, metric_to_predict)
    return X_test_data, y_actual
