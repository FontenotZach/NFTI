from datetime import datetime

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
from src.paths import FIGURES_DIR, LOGS_DIR, MODELS_KERAS_DIR, TUNING_DIR, ensure_dirs
from src.preprocessing.feature_preprocessor import (
    _extract_outcome,
    filter_labeled_samples,
    preprocess_data_for_criterion as preprocess_data_for_criterion_shared,
    print_preprocessing_sanity,
)

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

KERAS_METRICS = [
    AUC(name="auroc"),
    AUC(name="auprc", curve="PR"),
    Precision(name="precision"),
    Recall(name="recall"),
]


def _has_both_classes(y) -> bool:
    y_arr = np.asarray(y, dtype=float)
    labeled = y_arr[~np.isnan(y_arr)].astype(int)
    return len(labeled) > 0 and len(np.unique(labeled)) >= 2


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


def preprocess_data(trauma_dataset, metric_to_predict):
    X_binary, X_categorical, X_continuous, y = preprocess_data_for_criterion_shared(
        trauma_dataset, metric_to_predict, testing=False
    )
    print_preprocessing_sanity("train", X_binary, X_categorical, X_continuous, y)
    print("Training model...")
    return X_binary, X_categorical, X_continuous, y


def build_model(hp, binary_input_dim, categorical_input_dim, continuous_input_dim):
    binary_input = Input(shape=(binary_input_dim,), name="binary_input")
    categorical_input = Input(shape=(categorical_input_dim,), name="categorical_input")
    continuous_input = Input(shape=(continuous_input_dim,), name="continuous_input")

    normalized_continuous = BatchNormalization()(continuous_input)
    concatenated = Concatenate()([binary_input, categorical_input, normalized_continuous])

    x = concatenated
    for i in range(hp.Int("num_layers", 1, 4)):
        x = Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
            activation="relu",
            kernel_regularizer=l2(0.001),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(f"dropout_{i}", min_value=0.2, max_value=0.5, step=0.1))(x)

    nfti_output = Dense(1, activation="sigmoid", name="nfti_output")(x)

    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=[binary_input, categorical_input, continuous_input], outputs=nfti_output)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=KERAS_METRICS)
    return model


def _evaluate_keras_split(
    model,
    X_binary,
    X_categorical,
    X_continuous,
    y,
    record_ids,
    *,
    metric_to_predict: str,
    split_name: str,
    model_path: str,
    timestamp: str,
    include_posthoc: bool,
):
    y_prob = model.predict([X_binary, X_categorical, X_continuous], verbose=0).ravel()

    thresholds = [("fixed_0.5", 0.5)]
    if include_posthoc:
        y_clean, _, _, _ = clean_binary_eval_inputs(y, y_prob)
        if len(np.unique(y_clean)) >= 2:
            prefix = "posthoc_holdout" if split_name == "holdout" else f"posthoc_{split_name}"
            thresholds.extend(
                [
                    (f"{prefix}_youden", select_threshold_youden(y, y_prob)),
                    (f"{prefix}_f1", select_threshold_f1(y, y_prob)),
                ]
            )

    for idx, (threshold_type, threshold) in enumerate(thresholds):
        _run_eval_and_save(
            y_true=y,
            y_prob=y_prob,
            record_ids=record_ids,
            timestamp=timestamp,
            model_type="keras",
            metric_to_predict=metric_to_predict,
            split_name=split_name,
            threshold_type=threshold_type,
            threshold=threshold,
            model_path=model_path,
            log_filename_value=log_filename,
            save_predictions=(idx == 0),
        )


def tune_and_train_model(X_binary, X_categorical, X_continuous, y, metric_to_predict, trauma_dataset):
    global log_filename
    ensure_dirs()

    if not _has_both_classes(y):
        print(f"Skipping {metric_to_predict}: training data has fewer than two outcome classes.")
        return None

    (X_binary, X_categorical, X_continuous), y = filter_labeled_samples(
        X_binary, X_categorical, X_continuous, y=y
    )

    X_train_binary, X_test_binary, X_train_cat, X_test_cat, X_train_cont, X_test_cont, y_train, y_test = (
        train_test_split(
            X_binary,
            X_categorical,
            X_continuous,
            y,
            test_size=0.15,
            random_state=42,
            stratify=y,
        )
    )

    classes = np.unique(y_train)
    class_weights = dict(
        zip(classes, compute_class_weight("balanced", classes=classes, y=y_train))
    )

    tuner = kt.RandomSearch(
        lambda hp: build_model(
            hp, X_train_binary.shape[1], X_train_cat.shape[1], X_train_cont.shape[1]
        ),
        objective="val_auprc",
        max_trials=5,
        executions_per_trial=1,
        directory=str(TUNING_DIR / metric_to_predict),
        project_name="nfti_tuning",
    )

    tuner.search(
        [X_train_binary, X_train_cat, X_train_cont],
        y_train,
        epochs=10,
        validation_split=0.15,
        class_weight=class_weights,
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    loss, *metric_values = best_model.evaluate(
        [X_test_binary, X_test_cat, X_test_cont], y_test, verbose=0
    )
    print(f"Internal test loss: {loss}, metrics: {dict(zip(best_model.metrics_names, metric_values))}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_KERAS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_KERAS_DIR / f"nfti_model_{timestamp}_{metric_to_predict}.keras"
    best_model.save(model_path)
    print(f"Model saved at {model_path}")

    internal_record_ids = [f"internal_test_{i}" for i in range(len(y_test))]
    _evaluate_keras_split(
        best_model,
        X_test_binary,
        X_test_cat,
        X_test_cont,
        y_test,
        internal_record_ids,
        metric_to_predict=metric_to_predict,
        split_name="internal_test",
        model_path=str(model_path),
        timestamp=timestamp,
        include_posthoc=True,
    )

    print(f"\n--- Holdout evaluation for {metric_to_predict} ---\n")
    X_h_bin, X_h_cat, X_h_cont, y_holdout = preprocess_data_for_criterion_shared(
        trauma_dataset, metric_to_predict, testing=True
    )
    print_preprocessing_sanity("holdout", X_h_bin, X_h_cat, X_h_cont, y_holdout)

    holdout_record_ids = _holdout_record_ids(trauma_dataset)
    _evaluate_keras_split(
        best_model,
        X_h_bin,
        X_h_cat,
        X_h_cont,
        y_holdout,
        holdout_record_ids,
        metric_to_predict=metric_to_predict,
        split_name="holdout",
        model_path=str(model_path),
        timestamp=timestamp,
        include_posthoc=True,
    )

    return best_model


def build_and_train_model(X_binary, X_categorical, X_continuous, y, metric_to_predict, trauma_dataset):
    global log_filename

    print("SMOTE is disabled; training on original labeled samples.")

    if not _has_both_classes(y):
        print(f"Skipping {metric_to_predict}: training data has fewer than two outcome classes.")
        return None

    (X_binary, X_categorical, X_continuous), y = filter_labeled_samples(
        X_binary, X_categorical, X_continuous, y=y
    )

    binary_input = Input(shape=(X_binary.shape[1],), name="binary_input")
    categorical_input = Input(shape=(X_categorical.shape[1],), name="categorical_input")
    continuous_input = Input(shape=(X_continuous.shape[1],), name="continuous_input")

    normalized_continuous = BatchNormalization()(continuous_input)
    concatenated = Concatenate()([binary_input, categorical_input, normalized_continuous])

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    nfti_output = Dense(1, activation="sigmoid", name="nfti_output")(x)

    optimizer = Adam(learning_rate=0.001)
    model = Model(inputs=[binary_input, categorical_input, continuous_input], outputs=nfti_output)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=KERAS_METRICS)

    X_train_binary, X_test_binary, X_train_cat, X_test_cat, X_train_cont, X_test_cont, y_train, y_test = (
        train_test_split(
            X_binary,
            X_categorical,
            X_continuous,
            y,
            test_size=0.15,
            random_state=42,
            stratify=y,
        )
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    def lr_scheduler(epoch, lr):
        if epoch > 10:
            return lr * 0.1
        return lr

    lr_schedule = LearningRateScheduler(lr_scheduler)

    model.fit(
        [X_train_binary, X_train_cat, X_train_cont],
        y_train,
        epochs=10,
        batch_size=16,
        validation_split=0.15,
        callbacks=[early_stopping, lr_schedule],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_KERAS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_KERAS_DIR / f"nfti_model_{timestamp}_{metric_to_predict}.keras"
    model.save(model_path)
    print(f"Model saved at {model_path}")

    internal_record_ids = [f"internal_test_{i}" for i in range(len(y_test))]
    _evaluate_keras_split(
        model,
        X_test_binary,
        X_test_cat,
        X_test_cont,
        y_test,
        internal_record_ids,
        metric_to_predict=metric_to_predict,
        split_name="internal_test",
        model_path=str(model_path),
        timestamp=timestamp,
        include_posthoc=True,
    )

    print(f"\n--- Holdout evaluation for {metric_to_predict} ---\n")
    X_h_bin, X_h_cat, X_h_cont, y_holdout = preprocess_data_for_criterion_shared(
        trauma_dataset, metric_to_predict, testing=True
    )
    print_preprocessing_sanity("holdout", X_h_bin, X_h_cat, X_h_cont, y_holdout)

    holdout_record_ids = _holdout_record_ids(trauma_dataset)
    _evaluate_keras_split(
        model,
        X_h_bin,
        X_h_cat,
        X_h_cont,
        y_holdout,
        holdout_record_ids,
        metric_to_predict=metric_to_predict,
        split_name="holdout",
        model_path=str(model_path),
        timestamp=timestamp,
        include_posthoc=True,
    )

    return model


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


def train_trauma_model(trauma_dataset, metric_to_predict):
    global log_filename
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / log_filename

    output = f"\n\n----------- Creating models for {metric_to_predict} -----------"
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(output)

    X_binary, X_categorical, X_continuous, y = preprocess_data(trauma_dataset, metric_to_predict)
    model = tune_and_train_model(
        X_binary, X_categorical, X_continuous, y, metric_to_predict, trauma_dataset
    )
    return model
