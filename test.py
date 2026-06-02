import numpy as np
import pandas as pd
from src.TraumaDataset import generate_json_from_headers, load_from_json
from tensorflow.keras.models import load_model
import random
import pickle
import os
import tkinter as tk
from tkinter import filedialog
import json
from src.paths import PICKLES_DIR, MODELS_KERAS_DIR, ensure_dirs

def fill_missing_values(input_data, binary_cols, categorical_cols, continuous_cols):
    input_data[binary_cols] = input_data[binary_cols].fillna(0).astype(int)

    input_data[categorical_cols] = input_data[categorical_cols].fillna(0).astype(int)
    input_data[continuous_cols] = input_data[continuous_cols].fillna(0).astype(float)

    return input_data

# Function to test the model using random testing records from TraumaDataset
def test_with_random_record(model, trauma_dataset):
    # Get records that are flagged for testing
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    if not testing_records:
        print("No testing records available.")
        return

    # Randomly select a record for testing
    random_record = random.choice(testing_records)
    print("\n--- Testing with Random Record ---")
    print(random_record)

    X_data = list(random_record.data.values())
    headers = [header.name for header in trauma_dataset.get_headers() if header.data_type and header.usage == "1"]
    df = pd.DataFrame([X_data], columns=headers)  # DataFrame

    df = df.fillna(0)

    # Extract the binary, categorical, and continuous columns
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

    # Convert binary and continuous columns to numeric
    X_binary = df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values
    X_continuous = df[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values

    # Categorical variables
    X_categorical = pd.get_dummies(df[categorical_cols], dummy_na=True).fillna(0).values

    # Make the prediction using the model
    prediction = model.predict([X_binary, X_categorical, X_continuous])[0][0]  # Assuming single output
    print(f"\nPrediction (likelihood of NFTI positive): {prediction * 100:.2f}%")

    # Print the actual value of 'nfti_positive' for comparison
    actual_nfti_positive = random_record.y.get('nfti_positive', 0)
    print(f"Actual 'nfti_positive' value: {actual_nfti_positive}")

def test_all_testing_records(model, trauma_dataset):
    # Get records that are flagged for testing
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    if not testing_records:
        print("No testing records available.")
        return

    # Prompt the user to enter a cutoff value
    try:
        cutoff = float(input("Enter the cutoff threshold for classifying as NFTI positive (e.g., 0.5): "))
    except ValueError:
        print("Invalid cutoff value. Please enter a numeric value.")
        return

    print(f"\n--- Testing on {len(testing_records)} Records ---\n")

    # Convert the testing records into a DataFrame for easier processing
    df = pd.DataFrame([record.data for record in testing_records])

    # Extract the binary, categorical, and continuous columns
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

    # Convert binary and continuous columns to numeric
    X_binary = df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values
    X_continuous = df[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values

    # Convert categorical variables into one-hot encoded format
    X_categorical = pd.get_dummies(df[categorical_cols], dummy_na=True).fillna(0).values

    if len(X_binary) == 0 or len(X_categorical) == 0 or len(X_continuous) == 0:
        print("No valid testing data found.")
        return

    # Store the predictions and actual outcomes
    nfti_positive_predictions = []
    nfti_negative_predictions = []

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Loop through all testing records and make predictions
    for i, record in enumerate(testing_records):
        # Predict using the model
        prediction = model.predict([X_binary[i:i+1], X_categorical[i:i+1], X_continuous[i:i+1]])[0][0]

        # Get actual NFTI positive status
        actual_nfti_positive = record.y.get('nfti_OR', 0)

        # Append prediction to the appropriate list
        if actual_nfti_positive == 1:
            nfti_positive_predictions.append(prediction)
            if prediction >= cutoff:
                true_positive += 1
            else:
                false_negative += 1
        else:
            nfti_negative_predictions.append(prediction)
            if prediction < cutoff:
                true_negative += 1
            else:
                false_positive += 1

    # Calculate averages
    avg_positive_prediction = np.mean(nfti_positive_predictions) if nfti_positive_predictions else 0
    avg_negative_prediction = np.mean(nfti_negative_predictions) if nfti_negative_predictions else 0

    # Calculate sensitivity (recall)
    sensitivity_positive = true_positive / (true_positive + false_negative) * 100 if (true_positive + false_negative) > 0 else 0
    sensitivity_negative = true_negative / (true_negative + false_positive) * 100 if (true_negative + false_positive) > 0 else 0

    # Calculate overall accuracy
    total_records = len(testing_records)
    correct_predictions = true_positive + true_negative
    accuracy = correct_predictions / total_records * 100

    # Display results
    print(f"Average prediction for NFTI positive records: {avg_positive_prediction * 100:.2f}%")
    print(f"Average prediction for NFTI negative records: {avg_negative_prediction * 100:.2f}%")
    print(f"Overall accuracy based on cutoff {cutoff}: {accuracy:.2f}%")
    print(f"Sensitivity for NFTI positive cases: {sensitivity_positive:.2f}%")
    print(f"Sensitivity for NFTI negative cases: {sensitivity_negative:.2f}%")

    return accuracy

def test_all_testing_records_with_em_data(model, trauma_dataset):
    # Get records that are flagged for testing
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    if not testing_records:
        print("No testing records available.")
        return

    # Prompt the user to enter a cutoff value
    try:
        cutoff = float(input("Enter the cutoff threshold for classifying as NFTI positive (e.g., 0.5): "))
    except ValueError:
        print("Invalid cutoff value. Please enter a numeric value.")
        return

    print(f"\n--- Testing on {len(testing_records)} Records ---\n")

    # Convert the testing records into a DataFrame for easier processing
    df = pd.DataFrame([record.data for record in testing_records])

    # Extract the binary, categorical, and continuous columns
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

    # Convert binary and continuous columns to numeric
    X_binary = df[binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values
    X_continuous = df[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values

    # Convert categorical variables into one-hot encoded format
    X_categorical = pd.get_dummies(df[categorical_cols], dummy_na=True).fillna(0).values

    if len(X_binary) == 0 or len(X_categorical) == 0 or len(X_continuous) == 0:
        print("No valid testing data found.")
        return

    # Store the predictions and actual outcomes for records with EM data
    nfti_positive_predictions = []
    nfti_negative_predictions = []

    correct_predictions_with_em_data = 0

    # Initialize counters for records with EM data
    total_with_em_data = 0
    correct_with_em_data = 0
    nfti_positive_with_em_data = 0
    nfti_positive_correct_with_em_data = 0
    nfti_negative_with_em_data = 0
    nfti_negative_correct_with_em_data = 0

    # Loop through all testing records and make predictions for those with non-zero EMSSBP, EMSPULSERATE, and EMSRESPIRATORYRATE data
    for i, record in enumerate(testing_records):
        # Only consider records where EMSSBP, EMSPULSERATE, and EMSRESPIRATORYRATE are not zero
        if record.data.get('EMSSBP', 0) != 0 and record.data.get('EMSPULSERATE', 0) != 0 and record.data.get('EMSRESPIRATORYRATE', 0) != 0:
            total_with_em_data += 1

            # Predict using the model
            prediction = model.predict([X_binary[i:i+1], X_categorical[i:i+1], X_continuous[i:i+1]])[0][0]

            # Append prediction to the appropriate list
            if record.y.get('nfti_positive', 0) == 1:
                nfti_positive_predictions.append(prediction)
                nfti_positive_with_em_data += 1

                if prediction >= cutoff:
                    correct_with_em_data += 1
                    nfti_positive_correct_with_em_data += 1
            else:
                nfti_negative_predictions.append(prediction)
                nfti_negative_with_em_data += 1

                if prediction < cutoff:
                    correct_with_em_data += 1
                    nfti_negative_correct_with_em_data += 1

    # Calculate averages
    avg_positive_prediction = np.mean(nfti_positive_predictions) if nfti_positive_predictions else 0
    avg_negative_prediction = np.mean(nfti_negative_predictions) if nfti_negative_predictions else 0

    # Calculate overall accuracy and sensitivity for cases with EM data
    accuracy_with_em_data = correct_with_em_data / total_with_em_data * 100 if total_with_em_data > 0 else 0
    sensitivity_positive_with_em_data = nfti_positive_correct_with_em_data / nfti_positive_with_em_data * 100 if nfti_positive_with_em_data > 0 else 0
    sensitivity_negative_with_em_data = nfti_negative_correct_with_em_data / nfti_negative_with_em_data * 100 if nfti_negative_with_em_data > 0 else 0

    # Display results
    print(f"Average prediction for NFTI positive records with EM data: {avg_positive_prediction * 100:.2f}%")
    print(f"Average prediction for NFTI negative records with EM data: {avg_negative_prediction * 100:.2f}%")
    print(f"Overall accuracy for records with EM data based on cutoff {cutoff}: {accuracy_with_em_data:.2f}%")
    print(f"Sensitivity for NFTI positive cases with EM data: {sensitivity_positive_with_em_data:.2f}%")
    print(f"Sensitivity for NFTI negative cases with EM data: {sensitivity_negative_with_em_data:.2f}%")

    return accuracy_with_em_data

def show_testing_menu(model, trauma_dataset):
    while True:
        print("\n--- Testing Options ---")
        print("1. Test with random record from dataset")
        print("2. Manually input data for testing")
        print("3. Generate JSON template for user input")
        print("4. Run all testing records")
        print("5. Run all testing records with EMS vitals")
        print("6. Exit testing")
        choice = input("Enter your choice: ")

        if choice == '1':
            test_with_random_record(model, trauma_dataset)
        elif choice == '2':
            test_with_user_input(model, trauma_dataset.get_headers())
        elif choice == '3':
            generate_json_from_headers(trauma_dataset.get_headers())
        elif choice == '4':
            test_all_testing_records(model, trauma_dataset)
        elif choice == '5':
            test_all_testing_records_with_em_data(model, trauma_dataset)
        elif choice == '6':
            print("Exiting testing.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    ensure_dirs()
    model_path = MODELS_KERAS_DIR / "nfti_model.h5"
    model = load_model(model_path)
    
    # Load the TraumaDataset object
    datasets_dir = PICKLES_DIR / "datasets"
    with open(datasets_dir / 'trauma_dataset.pkl', 'rb') as f:
        trauma_dataset = pickle.load(f)

    show_testing_menu(model, trauma_dataset)