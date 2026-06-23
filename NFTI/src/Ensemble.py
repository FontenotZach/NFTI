from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

log_filename = f"log_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
from src.preprocessing.feature_preprocessor import (
    filter_labeled_samples,
    preprocess_data_for_criterion as preprocess_data_for_criterion_shared,
)
from src.paths import LOGS_DIR, MODELS_ENSEMBLE_DIR, ensure_dirs

# Populated by build_final_nfti_model; mirrored in app.py globals when training completes.
final_cutoff_metadata: Dict[str, str | float] = {}


def preprocess_data_for_criterion(criterion, trauma_dataset, testing):
    """Preprocess binary/categorical/continuous features and outcome labels for one criterion."""
    return preprocess_data_for_criterion_shared(trauma_dataset, criterion, testing=testing)


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
        y_pred = (p_clean >= threshold).astype(int)
        pred_rows = [
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
            for rid, yt, yp, yhat in zip(ids_clean, y_clean, p_clean, y_pred)
        ]
        if pred_rows:
            save_row_level_predictions(pred_rows)

    return row


def _posthoc_holdout_thresholds(y_true, y_prob) -> List[Tuple[str, float]]:
    y_clean, _, _, _ = clean_binary_eval_inputs(y_true, y_prob)
    if len(y_clean) == 0 or len(np.unique(y_clean)) < 2:
        return []
    return [
        ("posthoc_holdout_youden", select_threshold_youden(y_true, y_prob)),
        ("posthoc_holdout_f1", select_threshold_f1(y_true, y_prob)),
    ]


def _evaluate_threshold_suite(
    *,
    y_true,
    y_prob,
    record_ids,
    timestamp: str,
    model_type: str,
    metric_to_predict: str,
    split_name: str,
    model_path: str,
    log_filename_value: str,
    thresholds: List[Tuple[str, float]],
    include_posthoc_holdout: bool = False,
    save_predictions_threshold_types: Optional[set] = None,
) -> None:
    if save_predictions_threshold_types is None:
        save_predictions_threshold_types = {"fixed_0.5"}

    eval_thresholds = list(thresholds)
    if include_posthoc_holdout:
        eval_thresholds.extend(_posthoc_holdout_thresholds(y_true, y_prob))

    for threshold_type, threshold in eval_thresholds:
        _run_eval_and_save(
            y_true=y_true,
            y_prob=y_prob,
            record_ids=record_ids,
            timestamp=timestamp,
            model_type=model_type,
            metric_to_predict=metric_to_predict,
            split_name=split_name,
            threshold_type=threshold_type,
            threshold=float(threshold),
            model_path=model_path,
            log_filename_value=log_filename_value,
            save_predictions=(threshold_type in save_predictions_threshold_types),
        )


def _record_ids_for_split(trauma_dataset, *, testing: bool) -> list:
    return [
        record_id_from_trauma_record(record, idx)
        for idx, record in enumerate(trauma_dataset.get_records())
        if record.for_testing == testing
    ]


def _record_ids_for_labeled_y(trauma_dataset, *, testing: bool, y) -> list:
    ids = _record_ids_for_split(trauma_dataset, testing=testing)
    y_arr = np.asarray(y, dtype=float)
    if len(ids) != len(y_arr):
        raise ValueError(
            f"Record id count ({len(ids)}) does not match label count ({len(y_arr)})."
        )
    mask = ~np.isnan(y_arr)
    return [rid for rid, keep in zip(ids, mask) if keep]


def build_final_meta_input(models_xgboost, models_nn, trauma_dataset, *, testing: bool):
    """
    Stack per-criterion XGBoost (+ optional NN) probabilities for the final NFTI-positive model.
    """
    criterion_cols = []
    for criterion in models_xgboost.keys():
        X_binary, X_cat, X_cont, _ = preprocess_data_for_criterion(
            criterion, trauma_dataset, testing=testing
        )
        xgb_pred_prob = models_xgboost[criterion].predict_proba(
            np.column_stack((X_binary, X_cat, X_cont))
        )[:, 1]
        parts = [xgb_pred_prob.reshape(-1, 1)]
        nn_model = models_nn.get(criterion)
        if nn_model is not None:
            nn_pred_prob = nn_model.predict([X_binary, X_cat, X_cont]).reshape(-1, 1)
            parts.append(nn_pred_prob)
        criterion_cols.append(np.hstack(parts))
    return np.hstack(criterion_cols)


def build_ensemble_model_meta(models_xgboost, models_nn, trauma_dataset):
    global log_filename

    ensure_dirs()
    print("Building meta-model ensemble...")

    meta_models = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    holdout_record_ids = _record_ids_for_split(trauma_dataset, testing=True)

    for criterion in models_xgboost.keys():
        print(f"Building meta-model for {criterion}...")

        X_train_binary, X_train_cat, X_train_cont, y_train = preprocess_data_for_criterion(
            criterion, trauma_dataset, testing=False
        )

        xgb_pred_prob = models_xgboost[criterion].predict_proba(
            np.column_stack((X_train_binary, X_train_cat, X_train_cont))
        )[:, 1]

        meta_parts = [xgb_pred_prob.reshape(-1, 1)]
        nn_model = models_nn.get(criterion)
        if nn_model is not None:
            nn_pred_prob = nn_model.predict([X_train_binary, X_train_cat, X_train_cont]).reshape(-1, 1)
            meta_parts.append(nn_pred_prob)

        meta_input = np.hstack(meta_parts)
        input_shape = meta_input.shape[1]

        meta_model = Sequential()
        meta_model.add(Dense(64, activation="relu", input_shape=(input_shape,)))
        meta_model.add(Dropout(0.3))
        meta_model.add(Dense(32, activation="relu"))
        meta_model.add(Dropout(0.3))
        meta_model.add(Dense(1, activation="sigmoid"))
        meta_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

        (meta_input,), y_labeled = filter_labeled_samples(meta_input, y=y_train)
        meta_model.fit(meta_input, y_labeled, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

        MODELS_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_ENSEMBLE_DIR / f"nfti_model_{timestamp}_{criterion}.h5"
        meta_model.save(str(model_path))
        print(f"Model saved at {model_path}")

        meta_models[criterion] = meta_model

        X_hold_binary, X_hold_cat, X_hold_cont, y_holdout = preprocess_data_for_criterion(
            criterion, trauma_dataset, testing=True
        )
        xgb_hold_prob = models_xgboost[criterion].predict_proba(
            np.column_stack((X_hold_binary, X_hold_cat, X_hold_cont))
        )[:, 1]

        hold_parts = [xgb_hold_prob.reshape(-1, 1)]
        if nn_model is not None:
            nn_hold_prob = nn_model.predict([X_hold_binary, X_hold_cat, X_hold_cont]).reshape(-1, 1)
            hold_parts.append(nn_hold_prob)

        hold_meta_input = np.hstack(hold_parts)
        meta_pred_prob = meta_model.predict(hold_meta_input, verbose=0).flatten()

        y_clean, _, _, _ = clean_binary_eval_inputs(y_holdout, meta_pred_prob)
        if len(y_clean) == 0:
            print(f"Skipping ensemble holdout evaluation for {criterion}: no labeled records.")
            continue

        _evaluate_threshold_suite(
            y_true=y_holdout,
            y_prob=meta_pred_prob,
            record_ids=holdout_record_ids,
            timestamp=timestamp,
            model_type="ensemble_meta",
            metric_to_predict=criterion,
            split_name="holdout",
            model_path=str(model_path),
            log_filename_value=log_filename,
            thresholds=[("fixed_0.5", 0.5)],
            include_posthoc_holdout=True,
            save_predictions_threshold_types={"fixed_0.5"},
        )

    print(f"Evaluation results saved to {LOGS_DIR / log_filename}")
    print("Meta-model ensemble building and testing complete.")
    return meta_models


def build_final_nfti_model(models_xgboost, models_nn, trauma_dataset):
    global log_filename, final_cutoff_metadata

    ensure_dirs()
    print("Building final meta-model for NFTI positive...")

    _, _, _, y_train_nfti = preprocess_data_for_criterion(
        "nfti_positive", trauma_dataset, testing=False
    )

    final_meta_input = build_final_meta_input(
        models_xgboost, models_nn, trauma_dataset, testing=False
    )
    print(f"Final meta-input shape (training): {final_meta_input.shape}")

    input_shape = final_meta_input.shape[1]
    final_input = Input(shape=(input_shape,), name="final_input")
    x = Dense(64, activation="relu")(final_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    final_output = Dense(1, activation="sigmoid")(x)

    final_model = Model(inputs=final_input, outputs=final_output)
    final_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    (final_meta_input,), y_labeled = filter_labeled_samples(final_meta_input, y=y_train_nfti)
    final_model.fit(final_meta_input, y_labeled, epochs=20, batch_size=64, validation_split=0.2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_ENSEMBLE_DIR / f"nfti_model_{timestamp}_final_meta.keras"
    final_model.save(str(model_path))
    print(f"Final meta-model saved at {model_path}")

    final_pred_prob = final_model.predict(final_meta_input, verbose=0).flatten()

    y_clean, p_clean, _, _ = clean_binary_eval_inputs(y_labeled, final_pred_prob)
    if len(y_clean) == 0 or len(np.unique(y_clean)) < 2:
        training_youden = 0.5
        print("Only one labeled class on training split; using training_youden=0.5.")
    else:
        training_youden = select_threshold_youden(y_labeled, final_pred_prob)
        print(f"Training Youden threshold (training_youden): {training_youden:.4f}")

    final_cutoff_metadata = {
        "final_cutoff": float(training_youden),
        "final_cutoff_type": "training_youden",
        "cutoff_derivation_split": "training",
        "model_path": str(model_path),
    }

    training_record_ids = _record_ids_for_labeled_y(
        trauma_dataset, testing=False, y=y_train_nfti
    )

    _evaluate_threshold_suite(
        y_true=y_labeled,
        y_prob=final_pred_prob,
        record_ids=training_record_ids,
        timestamp=timestamp,
        model_type="ensemble_final",
        metric_to_predict="nfti_positive",
        split_name="training",
        model_path=str(model_path),
        log_filename_value=log_filename,
        thresholds=[
            ("fixed_0.5", 0.5),
            ("training_youden", training_youden),
        ],
        include_posthoc_holdout=False,
        save_predictions_threshold_types={"fixed_0.5"},
    )

    print(f"Evaluation results saved to {LOGS_DIR / log_filename}")
    print("Final NFTI positive model building and testing complete.")

    return final_model, training_youden, final_cutoff_metadata


def evaluate_final_ensemble_holdout(
    trauma_dataset,
    final_model,
    models_xgboost,
    models_nn,
    *,
    final_cutoff: float,
    final_cutoff_type: str = "training_youden",
    model_path: str = "",
    log_filename_value: Optional[str] = None,
    include_posthoc_holdout: bool = True,
) -> None:
    """Evaluate the saved final ensemble model on holdout records."""
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]
    if not testing_records:
        print("No testing records available.")
        return

    if log_filename_value is None:
        log_filename_value = f"log_ensemble_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    holdout_record_ids = _record_ids_for_split(trauma_dataset, testing=True)

    final_meta_input = build_final_meta_input(
        models_xgboost, models_nn, trauma_dataset, testing=True
    )
    print(f"Final meta-input shape (holdout): {final_meta_input.shape}")

    expected_features = final_model.input_shape[1]
    if final_meta_input.shape[1] != expected_features:
        print(
            f"Shape mismatch for final model. Expected {expected_features} input features, "
            f"got {final_meta_input.shape[1]}."
        )
        print(
            "Re-run model training (option 4) to rebuild the final ensemble model "
            "with the current criterion stack."
        )
        return

    final_pred_prob = final_model.predict(final_meta_input, verbose=0).flatten()
    y_holdout = preprocess_data_for_criterion("nfti_positive", trauma_dataset, testing=True)[3]

    y_clean, _, _, _ = clean_binary_eval_inputs(y_holdout, final_pred_prob)
    if len(y_clean) == 0:
        print("Skipping final ensemble holdout evaluation: no labeled records.")
        return

    _evaluate_threshold_suite(
        y_true=y_holdout,
        y_prob=final_pred_prob,
        record_ids=holdout_record_ids,
        timestamp=timestamp,
        model_type="ensemble_final",
        metric_to_predict="nfti_positive",
        split_name="holdout",
        model_path=model_path,
        log_filename_value=log_filename_value,
        thresholds=[
            ("fixed_0.5", 0.5),
            (final_cutoff_type, final_cutoff),
        ],
        include_posthoc_holdout=include_posthoc_holdout,
        save_predictions_threshold_types={"fixed_0.5"},
    )

    print(f"Evaluation results saved to {LOGS_DIR / log_filename_value}")
