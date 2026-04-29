import os
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, BatchNormalization, Embedding
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

from src.preprocessing.feature_preprocessor import preprocess_data_for_criterion as preprocess_data_for_criterion_shared
from src.paths import FIGURES_DIR, LOGS_DIR, MODELS_KERAS_DIR, TUNING_DIR, ensure_dirs

def press_enter():
    print("\n\nPress enter to continue.")
    input()
    return

# Helper function to preprocess the data
def preprocess_data(trauma_dataset, metric_to_predict):
    # Centralized preprocessing to ensure training/ensemble use identical logic.
    X_binary, X_categorical, X_continuous, y = preprocess_data_for_criterion_shared(
        trauma_dataset, metric_to_predict, testing=False
    )
    print("Training model...")
    return X_binary, X_categorical, X_continuous, y

def build_model(hp, binary_input_dim, categorical_input_dim, continuous_input_dim):
    # Input layers
    binary_input = Input(shape=(binary_input_dim,), name='binary_input')
    categorical_input = Input(shape=(categorical_input_dim,), name='categorical_input')
    continuous_input = Input(shape=(continuous_input_dim,), name='continuous_input')

    embedding_dim = hp.Int('embedding_dim', min_value=4, max_value=32, step=4)
    embedding = Embedding(input_dim=30, output_dim=embedding_dim)(categorical_input)

    flattened_embedding = Flatten()(embedding)

    normalized_continuous = BatchNormalization()(continuous_input)

    concatenated = Concatenate()([binary_input, flattened_embedding, normalized_continuous])

    for i in range(hp.Int('num_layers', 1, 4)):
        x = Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), 
                  activation='relu', kernel_regularizer='l2')(concatenated if i == 0 else x)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1))(x)

    nfti_output = Dense(1, activation='sigmoid', name='nfti_output')(x)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=[binary_input, categorical_input, continuous_input], outputs=nfti_output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def tune_and_train_model(X_binary, X_categorical, X_continuous, y, metric_to_predict, trauma_dataset):

    global log_filename
    ensure_dirs()

    # Train/test split
    X_train_binary, X_test_binary, X_train_cat, X_test_cat, X_train_cont, X_test_cont, y_train, y_test = train_test_split(
        X_binary, X_categorical, X_continuous, y, test_size=0.15, random_state=42
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))

    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, X_train_binary.shape[1], X_train_cat.shape[1], X_train_cont.shape[1]),
        objective='val_accuracy',
        max_trials=5,  # number of hyperparameter combinations to try
        executions_per_trial=1,  # number of times to repeat each trial
        directory=str(TUNING_DIR / metric_to_predict),
        project_name='nfti_tuning'
    )

    tuner.search([X_train_binary, X_train_cat, X_train_cont], y_train, 
                 epochs=10, validation_split=0.15, class_weight=class_weights)

    best_model = tuner.get_best_models(num_models=1)[0]

    loss, accuracy = best_model.evaluate([X_test_binary, X_test_cat, X_test_cont], y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred_prob = best_model.predict([X_test_binary, X_test_cat, X_test_cont])

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")

    MODELS_KERAS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_KERAS_DIR / f'nfti_model_{timestamp}_{metric_to_predict}.keras'
    best_model.save(model_path)

    print(f"Model saved at {model_path}")

    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    print(f"\n--- Testing on {len(testing_records)} Records ---\n")

    # Convert the testing records into a DataFrame for easier processing
    df = pd.DataFrame([record.data for record in testing_records])

    # Extract the binary, categorical, and continuous columns
    binary_cols = []
    categorical_cols = []
    continuous_cols = []

    for header in trauma_dataset.get_headers():
        if header.usage == '1' and header.timing in ['1']:  # Filter based on usage and timing
            if header.data_type == '1':  # Binary
                binary_cols.append(header.name)
            elif header.data_type == '2':  # Categorical
                categorical_cols.append(header.name)
            elif header.data_type == '3':  # Continuous
                continuous_cols.append(header.name)

    # Convert binary and continuous columns to numeric
    X_binary = df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values
    X_continuous = df[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values

    # Convert categorical variables into one-hot encoded format
    X_categorical = pd.get_dummies(df[categorical_cols], dummy_na=True).fillna(0).values

    if len(X_binary) == 0 or len(X_categorical) == 0 or len(X_continuous) == 0:
        print("No valid testing data found.")
        return

    # Combine binary, categorical, and continuous features
    X_test_data = np.hstack((X_binary, X_categorical, X_continuous))

    # Get actual NFTI positive status
    y_actual = np.array([record.y.get(metric_to_predict, 0) for record in testing_records])
    y_actual = np.nan_to_num(y_actual, nan=0).astype(int)

    y_pred_prob = best_model.predict([X_binary, X_categorical, X_continuous])

    # Calculate ROC curve and determine the optimal threshold (Youden's J)
    try:
        fpr, tpr, thresholds = roc_curve(y_actual, y_pred_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal threshold: {optimal_threshold}")
    except:
        print("Threshold error occured, defaulting to 0.")
        optimal_threshold = 0

    nfti_positive_predictions = []
    nfti_negative_predictions = []

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Loop through all testing records and make predictions using the optimal threshold
    for i, record in enumerate(testing_records):
        prediction_prob = y_pred_prob[i]
        prediction = 1 if prediction_prob >= optimal_threshold else 0

        actual_nfti_positive = record.y.get(metric_to_predict, 0)

        if actual_nfti_positive == 1:
            nfti_positive_predictions.append(prediction_prob)
            if prediction == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            nfti_negative_predictions.append(prediction_prob)
            if prediction == 0:
                true_negative += 1
            else:
                false_positive += 1

    # Calculate averages
    avg_positive_prediction = np.mean(nfti_positive_predictions) if nfti_positive_predictions else 0
    avg_negative_prediction = np.mean(nfti_negative_predictions) if nfti_negative_predictions else 0

    # Calculate sensitivity (recall) and precision
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    # Calculate overall accuracy
    total_records = len(testing_records)
    correct_predictions = true_positive + true_negative
    accuracy = correct_predictions / total_records * 100

    auc = roc_auc_score(y_test, y_pred_prob)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Display results
    print(f"Results for Neural netowrk ")
    print(f"Average prediction for {metric_to_predict} positive records: {avg_positive_prediction:.2f}")
    print(f"Average prediction for {metric_to_predict} negative records: {avg_negative_prediction:.2f}")
    print(f"Overall accuracy based on optimal threshold {optimal_threshold}: {accuracy:.2f}%")
    print(f"Recall (Sensitivity) for {metric_to_predict} positive cases: {recall * 100:.2f}%")
    print(f"Precision for {metric_to_predict} positive cases: {precision * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    evaluation_results = (
        f"\n--- {metric_to_predict} Nueral network Validation Set Model Evaluation ---\n"
        f"Threshold Type: ROC\n"
        f"Optimal Threshold: {optimal_threshold:.4f}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall (Sensitivity): {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%\n"
        f"ROC AUC: {auc:.4f}\n"
        f"Confusion matrix: tp - {true_positive}, tn - {true_negative}, fp - {false_positive}, fn - {false_negative}"
    )

    print(evaluation_results)

    # Append the evaluation results to a log file
    log_file_path = os.path.join('Logs', log_filename)
    os.makedirs('Logs', exist_ok=True)  # Create the directory if it doesn't exist

    with open(log_file_path, 'a') as log_file:
        log_file.write(evaluation_results)
    
    print(f"Evaluation results saved to {log_file_path}")

    return best_model

def build_and_train_model(X_binary, X_categorical, X_continuous, y, metric_to_predict, trauma_dataset):
    global log_filename
    # Apply SMOTE for oversampling of the minority class
    print("Applying SMOTE to the training data...")
    
    # Concatenate the binary, categorical, and continuous features to apply SMOTE
    X_combined = np.concatenate([X_binary, X_categorical, X_continuous], axis=1)

    print("Pre-SMOTE shape: " + str(X_combined.shape))

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)
    X_resampled, y_resampled = X_combined, y

    print("Post-SMOTE shape: " + str(X_resampled.shape))

    # Split the resampled data back into binary, categorical, and continuous inputs
    X_resampled_binary = X_resampled[:, :X_binary.shape[1]]
    X_resampled_categorical = X_resampled[:, X_binary.shape[1]:X_binary.shape[1] + X_categorical.shape[1]]
    X_resampled_continuous = X_resampled[:, -X_continuous.shape[1]:]

    # Input layers
    binary_input = Input(shape=(X_binary.shape[1],), name='binary_input')
    categorical_input = Input(shape=(X_categorical.shape[1],), name='categorical_input')
    continuous_input = Input(shape=(X_continuous.shape[1],), name='continuous_input')

    embedding = Embedding(input_dim=30, output_dim=5)(categorical_input)
    flattened_embedding = Flatten()(embedding)
    normalized_continuous = BatchNormalization()(continuous_input)

    concatenated = Concatenate()([binary_input, flattened_embedding, normalized_continuous])

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    nfti_output = Dense(1, activation='sigmoid', name='nfti_output')(x)

    optimizer = Adam(learning_rate=0.001)
    model = Model(inputs=[binary_input, categorical_input, continuous_input], outputs=nfti_output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train/test split (80/20 split)
    X_train_binary, X_test_binary, X_train_cat, X_test_cat, X_train_cont, X_test_cont, y_train, y_test = train_test_split(
        X_resampled_binary, X_resampled_categorical, X_resampled_continuous, y_resampled, test_size=0.15, random_state=42
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    def lr_scheduler(epoch, lr):
        if epoch > 10:
            return lr * 0.1
        return lr

    lr_schedule = LearningRateScheduler(lr_scheduler)

    model.fit([X_train_binary, X_train_cat, X_train_cont], y_train,
              epochs=10, batch_size=64, validation_split=0.15,
              callbacks=[early_stopping, lr_schedule])

    # Evaluate the model
    loss, accuracy = model.evaluate([X_test_binary, X_test_cat, X_test_cont], y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred_prob = model.predict([X_test_binary, X_test_cat, X_test_cont])

    # Calculate ROC curve and determine the optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")

    # Apply the optimal threshold for predictions
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    # Evaluate accuracy, precision, recall, F1, and AUC
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    evaluation_results = (
        f"\n--- {metric_to_predict} Test Set Model Evaluation ---\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall (Sensitivity): {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%\n"
        f"ROC AUC: {auc:.4f}\n"
    )
    
    print(evaluation_results)

    # Append the evaluation results to a log file
    log_file_path = os.path.join('Logs', log_filename)
    os.makedirs('Logs', exist_ok=True)  # Create the directory if it doesn't exist

    with open(log_file_path, 'a') as log_file:
        log_file.write(evaluation_results)
    
    print(f"Evaluation results saved to {log_file_path}")

    # Save the model
    model_dir = 'Models'
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'nfti_model_{timestamp}_{metric_to_predict}.keras')
    model.save(model_path)

    print(f"Model saved at {model_path}")

    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    print(f"\n--- Testing on {len(testing_records)} Records ---\n")

    # Convert the testing records into a DataFrame for easier processing
    df = pd.DataFrame([record.data for record in testing_records])

    # Extract the binary, categorical, and continuous columns
    binary_cols = []
    categorical_cols = []
    continuous_cols = []

    for header in trauma_dataset.get_headers():
        if header.usage == '1' and header.timing in ['1']:  # Filter based on usage and timing
            if header.data_type == '1':  # Binary
                binary_cols.append(header.name)
            elif header.data_type == '2':  # Categorical
                categorical_cols.append(header.name)
            elif header.data_type == '3':  # Continuous
                continuous_cols.append(header.name)

    # Convert binary and continuous columns to numeric
    X_binary = df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values
    X_continuous = df[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values

    # Convert categorical variables into one-hot encoded format
    X_categorical = pd.get_dummies(df[categorical_cols], dummy_na=True).fillna(0).values

    if len(X_binary) == 0 or len(X_categorical) == 0 or len(X_continuous) == 0:
        print("No valid testing data found.")
        return

    # Combine binary, categorical, and continuous features
    X_test_data = np.hstack((X_binary, X_categorical, X_continuous))

    # Get actual NFTI positive status
    y_actual = np.array([record.y.get(metric_to_predict, 0) for record in testing_records])
    y_actual = np.nan_to_num(y_actual, nan=0).astype(int)

    y_pred_prob = model.predict([X_binary, X_categorical, X_continuous])

    # Calculate ROC curve and determine the optimal threshold (Youden's J)
    try:
        fpr, tpr, thresholds = roc_curve(y_actual, y_pred_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal threshold: {optimal_threshold}")
    except:
        print("Threshold error occured, defaulting to 0.")
        optimal_threshold = 0

    # Apply the optimal threshold for predictions
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    # Evaluate accuracy, precision, recall, F1, and AUC
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    try:
        auc = roc_auc_score(y_actual, y_pred_prob)
    except:
        auc = 0

    evaluation_results = (
        f"\n--- {metric_to_predict} Validation Set Model Evaluation ---\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall (Sensitivity): {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%\n"
        f"ROC AUC: {auc:.4f}\n"
    )
    
    print(evaluation_results)

    # Append the evaluation results to a log file
    log_file_path = os.path.join('Logs', log_filename)
    os.makedirs('Logs', exist_ok=True)  # Create the directory if it doesn't exist

    with open(log_file_path, 'a') as log_file:
        log_file.write(evaluation_results)
    
    print(f"Evaluation results saved to {log_file_path}")

    return model

def compute_shap_values(model, X_test_binary, X_test_cat, X_test_cont, binary_cols, categorical_cols, continuous_cols, metric_to_predict, depth):
    """
    Compute and display SHAP values for the trained model. This version is optimized for speed during testing.
    """
    # Concatenate the test data to create the test data tuple
    test_data_tuple = np.concatenate([X_test_binary, X_test_cat, X_test_cont], axis=1)

    background_data = shap.sample(test_data_tuple, depth)  

    # Define the feature names
    binary_feature_names = binary_cols
    categorical_feature_names = list(pd.get_dummies(pd.DataFrame(columns=categorical_cols), dummy_na=True).columns)
    continuous_feature_names = continuous_cols

    all_feature_names = binary_feature_names + categorical_feature_names + continuous_feature_names

    print(all_feature_names)

    # Create the prediction function wrapper for SHAP
    def predict_fn(X):
        binary_input = X[:, :X_test_binary.shape[1]]
        categorical_input = X[:, X_test_binary.shape[1]:X_test_binary.shape[1] + X_test_cat.shape[1]]
        continuous_input = X[:, -X_test_cont.shape[1]:]
        return model.predict([binary_input, categorical_input, continuous_input]).flatten()

    # Initialize the SHAP KernelExplainer with the background data
    explainer = shap.KernelExplainer(predict_fn, background_data)

    shap_values = explainer.shap_values(test_data_tuple[:depth], nsamples=2000) 

    # Ensure that shap_values is a list (for multiple outputs)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Assuming a single output

    # Plot SHAP summary for the small subset
    # Save the SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, features=test_data_tuple[:depth], feature_names=all_feature_names, max_display=10, show=False)

    # Save the plot as a PNG file
    ensure_dirs()
    plot_path = FIGURES_DIR / ('shap_summary_plot' + metric_to_predict + '.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def random_forest(trauma_dataset):
    # Extract records from TraumaDataset
    records = [record for record in trauma_dataset.get_records() if not record.for_testing]
    X_data = [list(record.data.values()) for record in records]  # X inputs
    y_data = [record.y.get('nfti_positive', 0) for record in records]  # Y outputs

    # Convert X_data to a NumPy array
    X = np.array(X_data)
    y = np.array(y_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))

    # Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")
    
    # Save the model to the RandomForest directory
    model_dir = 'RandomForest'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    
    joblib.dump(rf_model, model_path)  # Save the model
    print(f"Random Forest model saved at {model_path}")

def train_trauma_model_xgboost(trauma_dataset, metric_to_predict):
    global log_filename
    """
    Train an XGBoost model on the specified metric.
    """
    from src.models.xgboost_model import train_xgboost_model

    # Train (and save) the XGBoost model in the new module.
    best_xgb_model = train_xgboost_model(
        trauma_dataset,
        metric_to_predict,
        log_filename=log_filename,
    )

    # Preserve the legacy behavior for holdout evaluation/ROC plotting.
    X_holdout, y_holdout = test_all_testing_records_xgboost(
        best_xgb_model, trauma_dataset, metric_to_predict
    )
    return best_xgb_model, X_holdout, y_holdout

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(best_xgb_model, X_test, y_test, metric_to_predict):
    """
    Generates and saves a ROC curve plot for the given XGBoost model.

    Parameters:
    - best_xgb_model: The trained XGBoost model.
    - X_test: The features of the testing dataset.
    - y_test: The true labels of the testing dataset.
    - metric_to_predict: The name of the metric being predicted, used for file naming.
    """
    # Predict probabilities for the positive class
    y_pred_prob = best_xgb_model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)  # Calculate area under the curve

    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {metric_to_predict}')
    plt.legend(loc="lower right")
    
    # Ensure the directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURES_DIR / f'{metric_to_predict}_ROC_curve.png'
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

    print(f"ROC curve plot saved to {plot_path}")
def test_with_threshold(threshold, threshold_name, metric_to_predict, testing_records, y_pred_prob):
    global log_filename

    # Prepare true labels and predicted probabilities
    y_true = np.array([record.y.get(metric_to_predict, 0) for record in testing_records])
    y_score = y_pred_prob

    # Filter out records where y_true is nan
    valid_indices = ~np.isnan(y_true)
    y_true = y_true[valid_indices]
    y_score = y_score[valid_indices]
    testing_records = [record for idx, record in enumerate(testing_records) if valid_indices[idx]]

    print(f"\n[DEBUG] After filtering invalid y_true for {metric_to_predict}")
    print(f"Length of y_true: {len(y_true)}")
    print(f"Length of y_score: {len(y_score)}")
    print(f"Unique values in y_true: {set(y_true)}")
    print(f"Any NaNs in y_score: {np.isnan(y_score).any()}")
    print(f"Any NaNs in y_true: {np.isnan(y_true).any()}\n")
    
    # Safely calculate AUC
    if np.isnan(y_score).any() or len(set(y_true)) < 2:
        auc = float('nan')
        print(f"[WARNING] Skipping AUC calculation for {metric_to_predict} - invalid data or single class present.\n")
    else:
        auc = roc_auc_score(y_true, y_score)
        print(f"[DEBUG] ROC AUC for {metric_to_predict}: {auc:.4f}\n")

    # Store the predictions and actual outcomes
    nfti_positive_predictions = []
    nfti_negative_predictions = []

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Loop through all testing records and make predictions using the threshold
    for i, record in enumerate(testing_records):
        prediction_prob = y_pred_prob[i]
        prediction = 1 if prediction_prob >= threshold else 0

        actual = record.y.get(metric_to_predict, 0)

        if actual == 1:
            nfti_positive_predictions.append(prediction_prob)
            if prediction == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            nfti_negative_predictions.append(prediction_prob)
            if prediction == 0:
                true_negative += 1
            else:
                false_positive += 1

    # Calculate averages
    avg_positive_prediction = np.mean(nfti_positive_predictions) if nfti_positive_predictions else 0
    avg_negative_prediction = np.mean(nfti_negative_predictions) if nfti_negative_predictions else 0

    # Calculate sensitivity (recall) and precision
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    # Calculate overall accuracy
    total_records = len(testing_records)
    correct_predictions = true_positive + true_negative
    accuracy = correct_predictions / total_records * 100

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Display results
    print(f"Results for XGBoost testing with threshold {threshold_name}")
    print(f"Average prediction for {metric_to_predict} positive records: {avg_positive_prediction:.2f}")
    print(f"Average prediction for {metric_to_predict} negative records: {avg_negative_prediction:.2f}")
    print(f"Overall accuracy based on threshold {threshold}: {accuracy:.2f}%")
    print(f"Recall (Sensitivity) for {metric_to_predict} positive cases: {recall * 100:.2f}%")
    print(f"Precision for {metric_to_predict} positive cases: {precision * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"ROC AUC: {auc if not np.isnan(auc) else 'Not Available'}")

    evaluation_results = (
        f"\n--- {metric_to_predict} XGBoost Validation Set Model Evaluation ---\n"
        f"Threshold Type: {threshold_name}\n"
        f"Optimal Threshold: {threshold:.4f}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall (Sensitivity): {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%\n"
        f"ROC AUC: {auc if not np.isnan(auc) else 'Not Available'}\n"
        f"Confusion matrix: tp - {true_positive}, tn - {true_negative}, fp - {false_positive}, fn - {false_negative}"
    )

    # Ensure the Logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / log_filename

    # Save the evaluation results to a log file
    with open(log_file_path, 'a') as log_file:
        log_file.write(evaluation_results)

    print(f"Evaluation results saved to {log_file_path}")
    print("\n")

def get_middle_threshold(testing_records, metric_to_predict, y_pred_prob):
    # Store the predictions and actual outcomes
    nfti_positive_predictions = []
    nfti_negative_predictions = []

    for i, record in enumerate(testing_records):
        prediction_prob = y_pred_prob[i]

        actual_nfti_positive = record.y.get(metric_to_predict, 0)

        if actual_nfti_positive == 1:
            nfti_positive_predictions.append(prediction_prob)

        else:
            nfti_negative_predictions.append(prediction_prob)


    # Calculate averages
    avg_positive_prediction = np.mean(nfti_positive_predictions) if nfti_positive_predictions else 0
    avg_negative_prediction = np.mean(nfti_negative_predictions) if nfti_negative_predictions else 0

    return (avg_positive_prediction + avg_negative_prediction) / 2

def test_all_testing_records_xgboost(model, trauma_dataset, metric_to_predict):
    global log_filename
    # Get records that are flagged for testing
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    if not testing_records:
        print("No testing records available.")
        return

    print(f"\n--- Testing on {len(testing_records)} Records ---\n")

    # Use the shared preprocessing so XGBoost sees the same feature columns
    # during training and evaluation.
    X_binary, X_categorical, X_continuous, y_actual = preprocess_data_for_criterion_shared(
        trauma_dataset, metric_to_predict, testing=True
    )
    X_test_data = np.hstack((X_binary, X_categorical, X_continuous))

    # Predict probabilities using the XGBoost model
    y_pred_prob = model.predict_proba(X_test_data)[:, 1]  # Probabilities for the positive class

    # Calculate the ROC curve and find the optimal threshold

    # Alternative threshold selection using F1-maximization
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred_prob)
    f1_scores = [f1_score(y_actual, y_pred_prob >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_f = thresholds[optimal_idx]
    print(f"Optimal threshold (f1): {optimal_threshold_f:.4f}")

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_ROC = thresholds[optimal_idx]
    print(f"Optimal threshold (ROC): {optimal_threshold_ROC:.4f}")

    middle_threshold = get_middle_threshold(testing_records, metric_to_predict, y_pred_prob)

    print("\n")

    test_with_threshold(optimal_threshold_f, "f-score", metric_to_predict, testing_records, y_pred_prob)

    test_with_threshold(optimal_threshold_ROC, "ROC", metric_to_predict, testing_records, y_pred_prob)

    test_with_threshold(middle_threshold, "Middle", metric_to_predict, testing_records, y_pred_prob)

    plot_roc_curve(model, X_test_data, y_actual, metric_to_predict)

    return X_test_data, y_actual

def train_trauma_model(trauma_dataset, metric_to_predict):
    global log_filename
    log_file_path = os.path.join('Logs', log_filename)
    os.makedirs('Logs', exist_ok=True)  # Create the directory if it doesn't exist

    output = f"\n\n----------- Creating models for {metric_to_predict} -----------"

    with open(log_file_path, 'a') as log_file:
        log_file.write(output)

    X_binary, X_categorical, X_continuous, y = preprocess_data(trauma_dataset, metric_to_predict)
    model = tune_and_train_model(X_binary, X_categorical, X_continuous, y, metric_to_predict, trauma_dataset)

    binary_cols = []
    categorical_cols = []
    continuous_cols = []

    for header in trauma_dataset.get_headers():
        if header.usage == '1' and header.timing in ['1']: #TODO
            if header.data_type == '1':  # Binary
                binary_cols.append(header.name)
            elif header.data_type == '2':  # Categorical
                categorical_cols.append(header.name)
            elif header.data_type == '3':  # Continuous
                continuous_cols.append(header.name)

    # compute_shap_values(model, X_binary, X_categorical, X_continuous, binary_cols, categorical_cols, continuous_cols, metric_to_predict, 400)

    return model

