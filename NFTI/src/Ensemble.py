import numpy as np
import pandas as pd
import os
from datetime import datetime
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

log_filename = f"log_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

from src.preprocessing.feature_preprocessor import preprocess_data_for_criterion as preprocess_data_for_criterion_shared
from src.paths import LOGS_DIR, MODELS_ENSEMBLE_DIR, ensure_dirs

def build_ensemble_model_meta(models_xgboost, models_nn, trauma_dataset):
    global log_filename

    ensure_dirs()
    print("Building meta-model ensemble...")

    meta_models = {}

    # For each criterion, use the outputs from XGBoost and Neural Network to build a meta-model
    for criterion in models_xgboost.keys():
        print(f'Building meta-model for {criterion}...')

        # Preprocess data for the criterion
        X_test_binary, X_test_cat, X_test_cont, y_test = preprocess_data_for_criterion(criterion, trauma_dataset, testing=False)

        # Get predictions from XGBoost
        xgb_pred_prob = models_xgboost[criterion].predict_proba(np.column_stack((X_test_binary, X_test_cat, X_test_cont)))[:, 1]

        meta_inputs = [xgb_pred_prob.reshape(-1, 1)]

        # Optionally add predictions from the Neural Network.
        nn_model = models_nn.get(criterion)
        if nn_model is not None:
            nn_pred_prob = nn_model.predict([X_test_binary, X_test_cat, X_test_cont])
            meta_inputs.append(nn_pred_prob.reshape(-1, 1))

        # Stack the predictions as input features for the meta-model
        meta_input = np.hstack(meta_inputs)

        # Build a simple neural network meta-model
        input_shape = meta_input.shape[1]
        meta_model = Sequential()
        meta_model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        meta_model.add(Dropout(0.3))
        meta_model.add(Dense(32, activation='relu'))
        meta_model.add(Dropout(0.3))
        meta_model.add(Dense(1, activation='sigmoid'))

        # Compile the meta-model
        meta_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the meta-model on the stacked inputs
        meta_model.fit(meta_input, y_test, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

        # Save the meta-model
        MODELS_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = MODELS_ENSEMBLE_DIR / f'nfti_model_{timestamp}_{criterion}.h5'
        meta_model.save(str(model_path))
        print(f"Model saved at {model_path}")

        meta_models[criterion] = meta_model

        X_test_binary, X_test_cat, X_test_cont, y_test = preprocess_data_for_criterion(criterion, trauma_dataset, testing=True)

        # Get predictions from XGBoost
        xgb_pred_prob = models_xgboost[criterion].predict_proba(np.column_stack((X_test_binary, X_test_cat, X_test_cont)))[:, 1]

        meta_inputs = [xgb_pred_prob.reshape(-1, 1)]
        nn_model = models_nn.get(criterion)
        if nn_model is not None:
            nn_pred_prob = nn_model.predict([X_test_binary, X_test_cat, X_test_cont])
            meta_inputs.append(nn_pred_prob.reshape(-1, 1))

        meta_input = np.hstack(meta_inputs)

        # Predict using the meta-model
        meta_pred_prob = meta_model.predict(meta_input).flatten()

        # Calculate ROC curve and determine optimal threshold
        try:
            fpr, tpr, thresholds = roc_curve(y_test, meta_pred_prob)
            optimal_idx = np.argmax(tpr - fpr)  # Maximize the difference between TPR and FPR
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal threshold for {criterion}: {optimal_threshold}")
        except:
            print("Threshold error occured, defaulting to 0.")
            optimal_threshold = 0

        # Apply the optimal threshold for predictions
        meta_pred = (meta_pred_prob >= optimal_threshold).astype(int)

        # Evaluate the ensemble model
        accuracy = accuracy_score(y_test, meta_pred)
        precision = precision_score(y_test, meta_pred)
        recall = recall_score(y_test, meta_pred)
        f1 = f1_score(y_test, meta_pred)
        try:    
            auc = roc_auc_score(y_test, meta_pred_prob)
        except:
            auc = 0

        evaluation_results = (
            f"\n\n--- Ensemble " + criterion + " Model Evaluation ---\n"
            f"Accuracy: {accuracy * 100:.2f}%\n"
            f"Precision: {precision * 100:.2f}%\n"
            f"Recall (Sensitivity): {recall * 100:.2f}%\n"
            f"F1 Score: {f1 * 100:.2f}%\n"
            f"ROC AUC: {auc:.4f}\n"
        )
        
        print(evaluation_results)

        # Append the evaluation results to a log file
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file_path = LOGS_DIR / log_filename
        with open(log_file_path, 'a') as log_file:
            log_file.write(evaluation_results)

    print(f"Evaluation results saved to {log_file_path}")
    print("Meta-model ensemble building and testing complete.")
    return meta_models

def build_final_nfti_model(models_xgboost, models_nn, trauma_dataset):
    global log_filename

    ensure_dirs()
    print("Building final meta-model for NFTI positive...")

    # Preprocess the data for the final NFTI positive prediction
    X_test_binary, X_test_cat, X_test_cont, y_test_nfti = preprocess_data_for_criterion('nfti_positive', trauma_dataset, testing=False)

    # Initialize list to hold the meta-model inputs (XGBoost + NN predictions)
    meta_inputs = []

    # Iterate through the criteria models for XGBoost and NN, and collect their predictions
    for criterion in models_xgboost.keys():
        print(f'Getting meta-model predictions for {criterion}...')

        # Preprocess data for each criterion (same method for each one)
        X_test_binary_criterion, X_test_cat_criterion, X_test_cont_criterion, _ = preprocess_data_for_criterion(criterion, trauma_dataset, testing=False)

        # Get predictions from XGBoost
        xgb_pred_prob = models_xgboost[criterion].predict_proba(np.column_stack((X_test_binary_criterion, X_test_cat_criterion, X_test_cont_criterion)))[:, 1]

        meta_inputs = [xgb_pred_prob.reshape(-1, 1)]

        nn_model = models_nn.get(criterion)
        if nn_model is not None:
            nn_pred_prob = nn_model.predict(
                [X_test_binary_criterion, X_test_cat_criterion, X_test_cont_criterion]
            )
            meta_inputs.append(nn_pred_prob.reshape(-1, 1))

        # Combine predictions from XGBoost and optional NN into the meta input
        meta_input = np.hstack(meta_inputs)

        print(f"Shape of meta: {meta_input.shape}")

        # Add to the list of meta inputs
        meta_inputs.append(meta_input)

    # Stack all the meta-model predictions as input for the final model
    final_meta_input = np.hstack(meta_inputs)

    # Build a neural network as the final model for NFTI positive prediction
    input_shape = final_meta_input.shape[1]

    final_input = Input(shape=(input_shape,), name='final_input')

    # Dense layers for the final meta-model
    x = Dense(64, activation='relu')(final_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    final_output = Dense(1, activation='sigmoid')(x)

    final_model = Model(inputs=final_input, outputs=final_output)
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the final model on the meta-input and NFTI positive labels
    final_model.fit(final_meta_input, y_test_nfti, epochs=20, batch_size=64, validation_split=0.2)

    # Save the final neural network model
    MODELS_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_ENSEMBLE_DIR / f'nfti_model_{timestamp}_final_meta.keras'
    final_model.save(str(model_path))

    print(f"Final meta-model saved at {model_path}")

    # Get predicted probabilities for the final model
    final_pred_prob = final_model.predict(final_meta_input).flatten()

    # Compute ROC curve to get the optimal threshold
    try:
        fpr, tpr, thresholds = roc_curve(y_test_nfti, final_pred_prob)
        optimal_idx = np.argmax(tpr - fpr)  # Maximize the difference between TPR and FPR
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal threshold for final NFTI positive model: {optimal_threshold}")
    except:
        print("Threshold error occured, defaulting to 0.")
        optimal_threshold = 0

    # Apply the optimal threshold for final prediction
    final_pred = (final_pred_prob >= optimal_threshold).astype(int)

    # Evaluate the final model
    accuracy = accuracy_score(y_test_nfti, final_pred)
    precision = precision_score(y_test_nfti, final_pred)
    recall = recall_score(y_test_nfti, final_pred)
    f1 = f1_score(y_test_nfti, final_pred)
    try:
        auc = roc_auc_score(y_test_nfti, final_pred_prob)
    except:
        auc = 0

    evaluation_results = (
        f"\n\n--- Ensemble NFTI Positive Test Set Model Evaluation ---\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall (Sensitivity): {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%\n"
        f"ROC AUC: {auc:.4f}\n"
    )
    
    print(evaluation_results)

    # Append the evaluation results to a log file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / log_filename
    with open(log_file_path, 'a') as log_file:
        log_file.write(evaluation_results)
    
    print(f"Evaluation results saved to {log_file_path}")
    print("Final NFTI positive model building and testing complete.")

    return final_model, optimal_threshold

def preprocess_data_for_criterion(criterion, trauma_dataset, testing):
    """
    Preprocess the data for a specific criterion (e.g., 'nfti_ICU') to use in model evaluation.
    The function will return the binary, categorical, continuous input matrices and the output labels (y).
    """

    # Centralized preprocessing so training + inference are aligned.
    return preprocess_data_for_criterion_shared(trauma_dataset, criterion, testing=testing)
