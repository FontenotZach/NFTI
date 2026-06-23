import pandas as pd
import pickle
import random
import os
from keras.models import load_model
from src.Header import Header
from src.TraumaRecord import TraumaRecord
from src.TraumaDataset import TraumaDataset
from test import show_testing_menu
import csv
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
from sklearn.metrics import PrecisionRecallDisplay

sys.path.append('src')
import ModelGen
import Ensemble

from src.paths import (
    ensure_dirs,
    FIGURES_DIR,
    LOGS_DIR,
    PICKLES_DIR,
    REPORTS_DIR,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
    SCHEMAS_DIR,
)
from src.dataset_audit import (
    audit_trauma_dataset,
    format_audit_summary,
    print_header_detail,
    write_audit_report,
)
from src.preprocessing.dataset_transform import describe_transform_state, transform_trauma_dataset
from src.preprocessing.cohort_filter import apply_prehospital_ems_cohort_filter_to_dataset
from src.preprocessing.mice_imputation import impute_trauma_dataset_mice

# Globals
trauma_dataset = None
model = None
models_xgboost = {}
models_nn = {}
meta_models = {}
final_model = None
customs = None
testing = False
final_cutoff = 0
final_cutoff_type = "training_youden"
cutoff_derivation_split = "training"
final_model_path = ""


def default_customs_path():
    path = SCHEMAS_DIR / "customs.csv"
    return str(path) if path.exists() else None


def ask_open_filename(title, filetypes, initialdir=None):
    """Open a file picker without blocking the terminal (Windows-safe tk setup)."""
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        kwargs = {"title": title, "filetypes": filetypes}
        if initialdir and os.path.isdir(initialdir):
            kwargs["initialdir"] = initialdir
        selected = filedialog.askopenfilename(**kwargs)
        return selected or ""
    except tk.TclError as exc:
        print(f"File dialog unavailable ({exc}).")
        return ""
    finally:
        if root is not None:
            root.destroy()


# Load header definitions and data fields from the CSV file
def load_header_definitions(csv_file_path):
    headers_info = {}
    print(f"Loading header definitions from {csv_file_path}...")
    with open(csv_file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            headers_info[row['Header']] = {
                'ntds_page': row['NTDS_Page'],
                'definition': row['Definition'],
                'timing': row['Timing'],
                'data_type': row['Type'],
                'usage': row['Usage'],
                'one_hot_grouping': row['One_Hot_Grouping'],
                'y': row['Y']
            }
    print("Header definitions loaded successfully.")
    return headers_info

def load_dataset(testing):
    print("Please select the dataset file to load.")
    data_file_path = ask_open_filename(
        title="Select Dataset File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=str(RAW_DATA_DIR),
    )
    
    if not data_file_path:
        print("No file selected. Using default dataset.")
        ensure_dirs()
        default_train = RAW_DATA_DIR / "dat5.csv"
        fallback = SAMPLES_DATA_DIR / "dat5_limited.csv"
        data_file_path = str(default_train if default_train.exists() else fallback)
    
    if testing:
        print("Testing mode enabled. Using a lighter dataset.")
        ensure_dirs()
        lite = SAMPLES_DATA_DIR / "dat5_limited.csv"
        if lite.exists():
            data_file_path = str(lite)
    
    print(f"Loading dataset from {data_file_path}...")
    df = pd.read_csv(data_file_path, low_memory=False)
    print("Dataset loaded successfully.")

    return df

def pickle_data():
    global trauma_dataset, customs, testing
    ensure_dirs()
    df = load_dataset(testing)
    if testing:
        print('DF head')
        print(df.head())
    print("Dataset loaded successfully.")

    # Load header definitions
    header_csv_path = str(SCHEMAS_DIR / 'header_definitions.csv')
    header_info = load_header_definitions(header_csv_path)

    # Initialize TraumaDataset
    trauma_dataset = TraumaDataset()

    print("Populating headers into TraumaDataset...")
    # Add header metadata
    for column in df.columns:
        header_details = header_info.get(column, {})
        ntds_page = header_details.get('ntds_page', '')
        definition = header_details.get('definition', '')
        timing = header_details.get('timing', '')
        data_type = header_details.get('data_type', '')
        usage = header_details.get('usage', '')
        one_hot_grouping = header_details.get('one_hot_grouping', '')
        y = header_details.get('y', '')

        # Add header to dataset
        trauma_dataset.add_header(
            column,
            ntds_page,
            definition,
            timing,
            data_type,
            usage,
            one_hot_grouping,
            y
        )

    customs_path = customs or default_customs_path()
    if customs_path:
        print(f"Loading custom headers from {customs_path}.")
        trauma_dataset.add_custom_features(customs_path)
    else:
        print("No custom features file set; skipping derived features.")

    trauma_dataset.validate_build(header_info, df.columns)

    print(f"Headers populated. Total headers: {len(trauma_dataset.get_headers())}")


    if testing:
        print("Testing mode enabled. Loading only the first 1000 records...")
        df = df.head(1000)
    
    # Populate records (train/test split assigned after cohort filtering).
    print("Populating records into TraumaDataset...")
    for _, row in df.iterrows():
        trauma_dataset.add_record(row, assign_split=False)
    print(f"Records populated. Total records: {len(trauma_dataset.get_records())}")

    # Primary prehospital EMS cohort: exclude non-ambulance transport and interfacility transfers.
    apply_prehospital_ems_cohort_filter_to_dataset(trauma_dataset)
    trauma_dataset.assign_train_test_split(random_state=42)
    print(
        f"Eligible prehospital EMS cohort ready for downstream steps: "
        f"{len(trauma_dataset.get_records())} records"
    )

    # Pickle dataset
    datasets_dir = PICKLES_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    pickle_file_path = str(datasets_dir / 'trauma_dataset.pkl')
    print(f"Saving pickled dataset to {pickle_file_path}...")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(trauma_dataset, f)
    print(f"Dataset pickled and saved successfully at {pickle_file_path}.")

def load_pickled_data():
    global trauma_dataset

    pickle_file_path = ask_open_filename(
        title="Select the pickled TraumaDataset file",
        filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")),
        initialdir=str(PICKLES_DIR / "datasets"),
    )

    if not pickle_file_path:
        print("No file selected. Please try again.")
        return

    print(f"Loading pickled dataset from {pickle_file_path}...")

    if os.path.exists(pickle_file_path):
        try:
            with open(pickle_file_path, 'rb') as f:
                trauma_dataset = pickle.load(f)
            print("Pickled dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    else:
        print(f"No pickle file found at {pickle_file_path}.")

def print_random_record():
    if trauma_dataset is None:
        print("No dataset loaded. Please load a pickled dataset first.")
    else:
        random_index = random.randint(0, len(trauma_dataset.records) - 1)
        print(f"Random Record [{random_index}]: {trauma_dataset.records[random_index]}")


def show_dataset_audit_menu():
    if trauma_dataset is None:
        print("No dataset loaded. Pickle or load a TraumaDataset first.")
        return

    while True:
        print("\n--- Dataset Audit / Verification ---")
        print("1. Run full audit report (summary + CSV export)")
        print("2. Inspect a specific header")
        print("3. Print a random record")
        print("4. Back to main menu")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            try:
                sample_size = int(input("Sample rows to export (default 10): ").strip() or "10")
            except ValueError:
                print("Invalid sample size. Using 10.")
                sample_size = 10

            audit = audit_trauma_dataset(trauma_dataset, sample_size=sample_size)
            print("\n" + format_audit_summary(audit))

            paths = write_audit_report(trauma_dataset, sample_size=sample_size)
            print("\nAudit files written:")
            for label, path in paths.items():
                if path.exists():
                    print(f"  {label}: {path}")
        elif choice == "2":
            header_name = input("Header name: ").strip()
            if header_name:
                print_header_detail(trauma_dataset, header_name)
            else:
                print("No header name provided.")
        elif choice == "3":
            print_random_record()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")


def train_model():

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    from src.evaluation.binary_metrics import clean_binary_eval_inputs

    if trauma_dataset is None:
        print("No dataset loaded. Please load a pickled dataset first.")
        return
    
    print("Starting model training using ModelGen...")

    # List to store models for ensemble
    models_xgboost_tuple = {}

    global models_xgboost
    global models_nn
    global meta_models
    global final_model
    global final_cutoff
    global final_cutoff_type
    global cutoff_derivation_split
    global final_model_path
    
    roc_data = {}
    pr_data = {}

    # Train models for each NFTI criterion
    for header in trauma_dataset.get_headers():
        if header.y == '1':
            print(f'Generating XGBoost model for header {header.name}')
            try:
                xgboost_model, X_test, y_test = ModelGen.train_trauma_model_xgboost(
                    trauma_dataset, header.name
                )
            except ValueError as exc:
                print(f"Skipped {header.name}: {exc}")
                continue

            if xgboost_model is None:
                continue
            
            # Store the model and test data
            models_xgboost_tuple[header.name] = (xgboost_model, X_test, y_test)
            models_xgboost[header.name] = xgboost_model

            print('Generating Neural Network model for header ' + header.name)
            # nn_model = ModelGen.train_trauma_model(trauma_dataset, header.name)
            nn_model = None
            models_nn[header.name] = nn_model

    if not models_xgboost_tuple:
        print("No models were trained (likely too few positive samples for one or more criteria).")
        return

    if "nfti_positive" in models_xgboost:
        try:
            from src.models.xgboost_shap import compute_nfti_positive_xgboost_shap

            compute_nfti_positive_xgboost_shap(trauma_dataset, models_xgboost["nfti_positive"])
        except Exception as exc:
            print(f"SHAP analysis for nfti_positive skipped: {exc}")

    # Plotting all ROC curves
    plt.figure(figsize=(10, 8))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(models_xgboost_tuple))))

    for header_name, (model, X_test, y_test) in models_xgboost_tuple.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_test_clean, y_pred_prob_clean, _, _ = clean_binary_eval_inputs(y_test, y_pred_prob)

        if len(y_test_clean) == 0:
            print(f"Skipping combined ROC curve for {header_name}: no labeled test records.")
            continue
        if len(np.unique(y_test_clean)) < 2:
            print(
                f"Skipping combined ROC curve for {header_name}: only one labeled class present "
                f"(positive={int((y_test_clean == 1).sum())}, "
                f"negative={int((y_test_clean == 0).sum())})."
            )
            continue

        fpr, tpr, _ = roc_curve(y_test_clean, y_pred_prob_clean)
        roc_auc = auc(fpr, tpr)
        roc_data[header_name] = (fpr, tpr, roc_auc)

        # Plot ROC curve for this model
        plt.plot(fpr, tpr, color=next(colors), lw=2, label=f'ROC curve of {header_name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiple Models')
    plt.legend(loc="lower right")
    plt.tight_layout()  # Adjust layout

    # Save the combined ROC plot
    ensure_dirs()
    roc_plot_path = FIGURES_DIR / 'combined_ROC_curve.png'
    plt.savefig(roc_plot_path)
    plt.close() 

    # Plotting all Precision-Recall curves
    plt.figure(figsize=(10, 8))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(models_xgboost_tuple))))

    for header_name, (model, X_test, y_test) in models_xgboost_tuple.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_test_clean, y_pred_prob_clean, _, _ = clean_binary_eval_inputs(y_test, y_pred_prob)

        if len(y_test_clean) == 0:
            print(f"Skipping combined PR curve for {header_name}: no labeled test records.")
            continue
        if len(np.unique(y_test_clean)) < 2:
            print(
                f"Skipping combined PR curve for {header_name}: only one labeled class present "
                f"(positive={int((y_test_clean == 1).sum())}, "
                f"negative={int((y_test_clean == 0).sum())})."
            )
            continue

        precision, recall, _ = precision_recall_curve(y_test_clean, y_pred_prob_clean)
        try:
            average_precision = average_precision_score(y_test_clean, y_pred_prob_clean)
        except ValueError:
            average_precision = float("nan")
        pr_data[header_name] = (precision, recall, average_precision)

        # Plot Precision-Recall curve for this model
        plt.plot(recall, precision, color=next(colors), lw=2, label=f'PR curve of {header_name} (AP = {average_precision:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Multiple Models')
    plt.legend(loc="lower left")
    plt.tight_layout()  # Adjust layout

    # Save the combined PR plot
    ensure_dirs()
    pr_plot_path = FIGURES_DIR / 'combined_PR_curve.png'
    plt.savefig(pr_plot_path)
    plt.close()

    print("Model training complete.")
    
    # Now build the ensemble model
    meta_models = Ensemble.build_ensemble_model_meta(models_xgboost, models_nn, trauma_dataset)

    final_model, final_cutoff, cutoff_metadata = Ensemble.build_final_nfti_model(
        models_xgboost, models_nn, trauma_dataset
    )
    final_cutoff_type = str(cutoff_metadata.get("final_cutoff_type", "training_youden"))
    cutoff_derivation_split = str(cutoff_metadata.get("cutoff_derivation_split", "training"))
    final_model_path = str(cutoff_metadata.get("model_path", ""))

    save_globals()

def generate_ems_judgement_roc():
    global trauma_dataset

    if trauma_dataset is None:
        print("No dataset loaded. Please load a pickled dataset first.")
        return

    print("Generating EMS Judgement ROC Curve...")

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from src.evaluation.binary_metrics import clean_binary_eval_inputs

    # Extract testing records from trauma_dataset
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    y_test_nfti = np.array([record.y.get("nfti_positive", np.nan) for record in testing_records], dtype=float)
    vpoemsjudge = pd.to_numeric(
        pd.Series([record.data.get("VPOEMSJUDGE") for record in testing_records]),
        errors="coerce",
    ).to_numpy(dtype=float)

    y_clean, score_clean, _, _ = clean_binary_eval_inputs(y_test_nfti, vpoemsjudge)
    if len(y_clean) == 0:
        print("Skipping EMS judgement ROC: no labeled records with valid VPOEMSJUDGE scores.")
        return
    if len(np.unique(y_clean)) < 2:
        print(
            "Skipping EMS judgement ROC: only one labeled class present "
            f"(positive={int((y_clean == 1).sum())}, negative={int((y_clean == 0).sum())})."
        )
        return

    fpr_human, tpr_human, _ = roc_curve(y_clean, score_clean)
    roc_auc_human = auc(fpr_human, tpr_human)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_human, tpr_human, color='blue', lw=2, label=f'EMS VPOEMSJUDGE (AUC = {roc_auc_human:.2f})')

    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: EMS Judgement vs NFTI Positive')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save and show the plot
    ensure_dirs()
    plt.savefig(FIGURES_DIR / 'ems_judgement_roc_curve.png')
    plt.show()

    print("EMS Judgement ROC Curve generated and saved.")

def browse_and_load_model():
    global model

    model_path = ask_open_filename(
        title="Select the pre-made model file",
        filetypes=(("HDF5 files", "*.h5"), ("Keras model files", "*.keras"), ("All files", "*.*")),
        initialdir=str(PICKLES_DIR),
    )

    if not model_path:
        print("No file selected. Please try again.")
        return

    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Error loading model: {e}")

def load_custom_feature_file():
    """Set the customs CSV path for the next run of option 1 (no dataset required)."""
    global customs

    default_path = default_customs_path()
    print(
        "Select a customs CSV. It will be applied the next time you run "
        "'Pickle the data' (option 1). No dataset needs to be loaded first."
    )
    if default_path:
        print(f"Default: {default_path}")

    selected = ask_open_filename(
        title="Select Custom Feature File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        initialdir=str(SCHEMAS_DIR),
    )

    if selected:
        if not os.path.isfile(selected):
            print(f"File not found: {selected}")
            return
        customs = selected
        print(f"Custom features file set: {customs}")
        print("Run option 1 (Pickle the data) to build the dataset with these features.")
        return

    print("No file selected in dialog.")
    if default_path:
        use_default = input(f"Use default customs file? [{default_path}] [Y/n]: ").strip().lower()
        if use_default in ("", "y", "yes"):
            customs = default_path
            print(f"Custom features file set: {customs}")
            print("Run option 1 (Pickle the data) to apply.")
            return

    manual = input("Enter path to customs CSV (or press Enter to cancel): ").strip()
    if manual and os.path.isfile(manual):
        customs = manual
        print(f"Custom features file set: {customs}")
        print("Run option 1 (Pickle the data) to apply.")
    elif manual:
        print(f"File not found: {manual}")
    else:
        print("Custom features file unchanged.")
        if customs:
            print(f"Current setting: {customs}")
        elif default_path:
            print(f"Option 1 will still use the default: {default_path}")

def compute_missing_list(trauma_dataset_arg=None):
    global trauma_dataset
    ds = trauma_dataset_arg if trauma_dataset_arg is not None else trauma_dataset
    if ds is None:
        raise ValueError("No `trauma_dataset` loaded. Load data before computing missing values.")
    """
    Computes the ratio of missing values for each header in the TraumaDataset, splits records into NFTI positive and negative groups, and saves the missing value ratios for each group.

    Args:
        trauma_dataset: An instance of the TraumaDataset containing the data and headers.
    """

        # Extract data and labels from trauma_dataset records
    records_data = []
    nfti_positive_labels = []

    for record in ds.get_records():
        records_data.append(record.data)
        nfti_positive_labels.append(record.y.get('nfti_positive', None))

    # Convert records to a DataFrame
    df = pd.DataFrame(records_data)
    df['nfti_positive'] = nfti_positive_labels

    # Ensure the nfti_positive column exists and is valid
    if 'nfti_positive' not in df.columns or df['nfti_positive'].isna().all():
        raise ValueError("The 'nfti_positive' field is missing or contains no valid data.")

    # Split the dataset based on nfti_positive field
    positive_records = df[df['nfti_positive'] == 1]
    negative_records = df[df['nfti_positive'] == 0]

    # Initialize dictionaries to store header names and their missing ratios for each group
    missing_ratios_positive = {}
    missing_ratios_negative = {}

    # Iterate over headers to calculate missing ratios for each group
    for header in ds.get_headers():
        header_name = header.name

        if header_name in df.columns:
            # Calculate the ratio of missing values for this column in both groups
            missing_ratios_positive[header_name] = positive_records[header_name].isna().mean()
            missing_ratios_negative[header_name] = negative_records[header_name].isna().mean()

    # Sort headers by the ratio of missing values in descending order for each group
    sorted_missing_ratios_positive = sorted(missing_ratios_positive.items(), key=lambda x: x[1], reverse=True)
    sorted_missing_ratios_negative = sorted(missing_ratios_negative.items(), key=lambda x: x[1], reverse=True)

    # Prepare output strings for each group
    output_lines_positive = [f"{header}: {missing_ratio:.2%}\n" for header, missing_ratio in sorted_missing_ratios_positive]
    output_lines_negative = [f"{header}: {missing_ratio:.2%}\n" for header, missing_ratio in sorted_missing_ratios_negative]

    # Define output file paths
    ensure_dirs()
    positive_output_path = REPORTS_DIR / 'missing_positive.txt'
    negative_output_path = REPORTS_DIR / 'missing_negative.txt'

    # Write the output for positive cases to the file
    with open(positive_output_path, 'w') as file:
        file.writelines(output_lines_positive)

    print(f"Missing value ratios for NFTI positive cases saved to {positive_output_path}")

    # Write the output for negative cases to the file
    with open(negative_output_path, 'w') as file:
        file.writelines(output_lines_negative)

    print(f"Missing value ratios for NFTI negative cases saved to {negative_output_path}")

    return sorted_missing_ratios_positive, sorted_missing_ratios_negative

def generate_univariate_analysis():
    """
    Generates univariate analysis tables for binary and continuous variables,
    sorted by effect size, and exports them as matplotlib figures.

    Args:
        trauma_dataset: An instance of the TraumaDataset containing data and headers.
    """
    global trauma_dataset
    import matplotlib.pyplot as plt
    # Split variables by type
    binary_cols = []
    continuous_cols = []

    for header in trauma_dataset.get_headers():
        if header.usage == '1' and header.timing in ['1']:
            if header.data_type == '1':  # Binary
                binary_cols.append(header.name)
            elif header.data_type == '3':  # Continuous
                continuous_cols.append(header.name)

    # Extract data
    records = trauma_dataset.get_records()
    df = pd.DataFrame([record.data for record in records])
    df['nfti_positive'] = [record.y['nfti_positive'] for record in records]

    # Split data into NFTI-positive and NFTI-negative groups
    neg_data = df[df['nfti_positive'] == 0]
    pos_data = df[df['nfti_positive'] == 1]

    # Initialize results tables
    binary_results = []
    continuous_results = []

    # Process binary variables
    for col in binary_cols:
        neg_mean = neg_data[col].mean() * 100
        pos_mean = pos_data[col].mean() * 100

        # Adjust SEX field to include separate rows for male and female with confidence intervals
        if col == 'SEX':
            neg_male = neg_mean
            neg_female = 100 - neg_male
            pos_male = pos_mean
            pos_female = 100 - pos_male

            # Calculate odds ratios and confidence intervals
            male_odds_ratio = ((pos_male / (100 - pos_male)) / (neg_male / (100 - neg_male))) if neg_male > 0 and pos_male > 0 else np.nan
            male_odds_se = np.sqrt(1 / neg_data[col].sum() + 1 / (len(neg_data) - neg_data[col].sum()) +
                                   1 / pos_data[col].sum() + 1 / (len(pos_data) - pos_data[col].sum()))
            male_ci_low = np.exp(np.log(male_odds_ratio) - 1.96 * male_odds_se) if not np.isnan(male_odds_ratio) else np.nan
            male_ci_high = np.exp(np.log(male_odds_ratio) + 1.96 * male_odds_se) if not np.isnan(male_odds_ratio) else np.nan

            female_odds_ratio = ((pos_female / (100 - pos_female)) / (neg_female / (100 - neg_female))) if neg_female > 0 and pos_female > 0 else np.nan
            female_odds_se = np.sqrt(1 / (len(neg_data) - neg_data[col].sum()) + 1 / neg_data[col].sum() +
                                     1 / (len(pos_data) - pos_data[col].sum()) + 1 / pos_data[col].sum())
            female_ci_low = np.exp(np.log(female_odds_ratio) - 1.96 * female_odds_se) if not np.isnan(female_odds_ratio) else np.nan
            female_ci_high = np.exp(np.log(female_odds_ratio) + 1.96 * female_odds_se) if not np.isnan(female_odds_ratio) else np.nan

            binary_results.append([
                'SEX (Male)',
                f"{neg_male:.2f}%",
                f"{pos_male:.2f}%",
                f"{male_odds_ratio:.2f} ({male_ci_low:.2f}, {male_ci_high:.2f})" if not np.isnan(male_odds_ratio) else "N/A",
                abs(np.log(male_odds_ratio)) if not np.isnan(male_odds_ratio) else 0
            ])

            binary_results.append([
                'SEX (Female)',
                f"{neg_female:.2f}%",
                f"{pos_female:.2f}%",
                f"{female_odds_ratio:.2f} ({female_ci_low:.2f}, {female_ci_high:.2f})" if not np.isnan(female_odds_ratio) else "N/A",
                abs(np.log(female_odds_ratio)) if not np.isnan(female_odds_ratio) else 0
            ])
            continue

        # Odds ratio calculation
        odds_ratio = ((pos_mean / (100 - pos_mean)) / (neg_mean / (100 - neg_mean))) if neg_mean > 0 and pos_mean > 0 else np.nan

        # Confidence interval for odds ratio
        odds_se = np.sqrt(1 / neg_data[col].sum() + 1 / (len(neg_data) - neg_data[col].sum()) +
                          1 / pos_data[col].sum() + 1 / (len(pos_data) - pos_data[col].sum()))
        ci_low = np.exp(np.log(odds_ratio) - 1.96 * odds_se) if not np.isnan(odds_ratio) else np.nan
        ci_high = np.exp(np.log(odds_ratio) + 1.96 * odds_se) if not np.isnan(odds_ratio) else np.nan

        binary_results.append([col, f"{neg_mean:.2f}%", f"{pos_mean:.2f}%", f"{odds_ratio:.2f} ({ci_low:.2f}, {ci_high:.2f})", abs(np.log(odds_ratio)) if not np.isnan(odds_ratio) else 0])

    # Process continuous variables
    for col in continuous_cols:
        neg_median = neg_data[col].median()
        neg_std = neg_data[col].std()
        pos_median = pos_data[col].median()
        pos_std = pos_data[col].std()

        # Cohen's d calculation
        pooled_std = np.sqrt(((len(neg_data[col]) - 1) * neg_std**2 + (len(pos_data[col]) - 1) * pos_std**2) /
                             (len(neg_data[col]) + len(pos_data[col]) - 2))
        cohens_d = (pos_data[col].mean() - neg_data[col].mean()) / pooled_std if pooled_std > 0 else np.nan

        # Confidence interval for Cohen's d
        n_neg, n_pos = len(neg_data[col]), len(pos_data[col])
        ci_low = cohens_d - 1.96 * np.sqrt((1 / n_neg + 1 / n_pos)) if not np.isnan(cohens_d) else np.nan
        ci_high = cohens_d + 1.96 * np.sqrt((1 / n_neg + 1 / n_pos)) if not np.isnan(cohens_d) else np.nan

        continuous_results.append([col, f"{neg_median:.2f} ± {neg_std:.2f}", f"{pos_median:.2f} ± {pos_std:.2f}", f"d: {cohens_d:.2f} ({ci_low:.2f}, {ci_high:.2f})", abs(cohens_d) if not np.isnan(cohens_d) else 0])

    # Convert results to DataFrames and sort by effect size
    binary_results_df = pd.DataFrame(binary_results, columns=['Variable', 'NFTI-Negative Composition', 'NFTI-Positive Composition', 'Odds Ratio (95% CI)', 'SortKey'])
    continuous_results_df = pd.DataFrame(continuous_results, columns=['Variable', 'NFTI-Negative \nMedian and Standard Deviation', 'NFTI-Positive \nMedian and Standard Deviation', 'Cohen\'s d (95% CI)', 'SortKey'])

    binary_results_df = binary_results_df.sort_values(by='SortKey', ascending=False).drop(columns=['SortKey'])
    continuous_results_df = continuous_results_df.sort_values(by='SortKey', ascending=False).drop(columns=['SortKey'])

    # Export to matplotlib figures
    ensure_dirs()
    for results_df, filename in zip(
        [binary_results_df, continuous_results_df],
        [FIGURES_DIR / 'binary_analysis_table.png', FIGURES_DIR / 'continuous_analysis_table.png'],
    ):
        fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=results_df.values,
                         colLabels=results_df.columns,
                         cellLoc='center',
                         loc='center')

        # Format the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(results_df.columns))))

        # Adjust row spacing for better padding
        for (row, col), cell in table.get_celld().items():
            cell.set_height(cell.get_height() + 0.003)

        # Save the figure with higher DPI and no surrounding whitespace
        plt.savefig(filename, bbox_inches='tight', dpi=500)
        print(f"Table saved as {filename}")

    return binary_results_df, continuous_results_df


def impute_missing_values(trauma_dataset_arg=None, max_iter=10):
    """
    Run MICE imputation on the loaded TraumaDataset and save an imputed pickle.
    """
    global trauma_dataset

    ds = trauma_dataset_arg if trauma_dataset_arg is not None else trauma_dataset
    if ds is None:
        print("No dataset loaded. Pickle or load a TraumaDataset first.")
        return None

    print("Computing missing values report...")
    compute_missing_list(ds)

    print(f"Applying MICE imputation (max_iter={max_iter})...")
    try:
        impute_trauma_dataset_mice(
            ds,
            max_iter=max_iter,
            save_path=PICKLES_DIR / "datasets" / "trauma_dataset_imputed.pkl",
        )
    except ValueError as exc:
        print(f"MICE imputation failed: {exc}")
        return None

    print("Missing values imputed with MICE successfully.")
    print(f"Saved imputed dataset to {PICKLES_DIR / 'datasets' / 'trauma_dataset_imputed.pkl'}")
    return ds


def normalize_and_encode_dataset():
    """
    Z-score normalize continuous columns and one-hot encode categorical columns
    on the loaded TraumaDataset, then save a transformed pickle.
    """
    global trauma_dataset

    if trauma_dataset is None:
        print("No dataset loaded. Pickle or load a TraumaDataset first.")
        return

    print("Applying z-score normalization and one-hot encoding...")
    try:
        trauma_dataset = transform_trauma_dataset(trauma_dataset)
    except ValueError as exc:
        print(f"Transform failed: {exc}")
        return

    print(describe_transform_state(trauma_dataset))

    ensure_dirs()
    datasets_dir = PICKLES_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    pickle_file_path = str(datasets_dir / "trauma_dataset_transformed.pkl")
    print(f"Saving transformed dataset to {pickle_file_path}...")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(trauma_dataset, f)
    print("Transformed dataset saved successfully.")


def evaluate_testing_records_with_ensemble(trauma_dataset):
    global models_xgboost
    global models_nn
    global final_model
    global final_cutoff
    global final_cutoff_type
    global cutoff_derivation_split
    global final_model_path

    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]
    if not testing_records:
        print("No testing records available.")
        return

    print(f"\n--- Running {len(testing_records)} Records through Ensemble Models ---\n")

    for criterion in models_xgboost.keys():
        print(f"Getting meta-model predictions for {criterion}...")

    Ensemble.evaluate_final_ensemble_holdout(
        trauma_dataset,
        final_model,
        models_xgboost,
        models_nn,
        final_cutoff=final_cutoff,
        final_cutoff_type=final_cutoff_type,
        model_path=final_model_path,
        include_posthoc_holdout=True,
    )

def save_globals():
    """
    Save global variables (models_xgboost, models_nn, meta_models, final_model) as pickled files in the Model_Pickle directory.
    """
    global models_xgboost, models_nn, meta_models, final_model, final_cutoff
    global final_cutoff_type, cutoff_derivation_split, final_model_path

    ensure_dirs()
    model_dir = PICKLES_DIR / "globals"
    model_dir.mkdir(parents=True, exist_ok=True)

    cutoff_metadata = {
        "final_cutoff": final_cutoff,
        "final_cutoff_type": final_cutoff_type,
        "cutoff_derivation_split": cutoff_derivation_split,
        "model_path": final_model_path,
    }

    # Save each global variable as a separate pickle file
    with open(model_dir / 'models_xgboost.pkl', 'wb') as f:
        pickle.dump(models_xgboost, f)
    with open(model_dir / 'models_nn.pkl', 'wb') as f:
        pickle.dump(models_nn, f)
    with open(model_dir / 'meta_models.pkl', 'wb') as f:
        pickle.dump(meta_models, f)
    with open(model_dir / 'final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(model_dir / 'final_cutoff.pkl', 'wb') as f:
        pickle.dump(final_cutoff, f)
    with open(model_dir / 'final_cutoff_metadata.pkl', 'wb') as f:
        pickle.dump(cutoff_metadata, f)

    print(f"Global models saved in {model_dir}")

def load_globals():
    """
    Load global variables (models_xgboost, models_nn, meta_models, final_model) from pickled files in the Model_Pickle directory.
    """
    global models_xgboost, models_nn, meta_models, final_model, final_cutoff
    global final_cutoff_type, cutoff_derivation_split, final_model_path

    ensure_dirs()
    model_dir = PICKLES_DIR / "globals"

    # Load each global variable from its respective pickle file
    with open(model_dir / 'models_xgboost.pkl', 'rb') as f:
        models_xgboost = pickle.load(f)
    with open(model_dir / 'models_nn.pkl', 'rb') as f:
        models_nn = pickle.load(f)
    with open(model_dir / 'meta_models.pkl', 'rb') as f:
        meta_models = pickle.load(f)
    with open(model_dir / 'final_model.pkl', 'rb') as f:
        final_model = pickle.load(f)
    with open(model_dir / 'final_cutoff.pkl', 'rb') as f:
        final_cutoff = pickle.load(f)

    metadata_path = model_dir / 'final_cutoff_metadata.pkl'
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            cutoff_metadata = pickle.load(f)
        final_cutoff_type = str(cutoff_metadata.get("final_cutoff_type", "training_youden"))
        cutoff_derivation_split = str(cutoff_metadata.get("cutoff_derivation_split", "training"))
        final_model_path = str(cutoff_metadata.get("model_path", ""))
    else:
        final_cutoff_type = "training_youden"
        cutoff_derivation_split = "training"
        final_model_path = ""

    print(f"Global models loaded from {model_dir}")

def show_menu():
    global trauma_dataset, model, customs, testing
    while True:
        print("\n--- Trauma Dataset Menu ---")
        print("1. Pickle the data")
        print("2. Load pickled TraumaDataset")
        print("3. Audit / verify dataset")
        print("4. Train the model using ModelGen")
        print("5. Open individual model testing interface")
        print("6. Browse and load pre-made model")
        print("7. Set custom features CSV (optional before 1; default is data/schemas/customs.csv)")
        print("8. Toggle testing mode")
        print("9. Run MICE imputation")
        print("10. Normalize and encode dataset (z-score + one-hot)")
        print("11. Run ensemble testing")
        print("12. Load globals")
        print("13. Generate EMS judgement comparison")
        print("14. Compute missing list")
        print("15. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            pickle_data()
        elif choice == '2':
            load_pickled_data()
        elif choice == '3':
            show_dataset_audit_menu()
        elif choice == '4':
            train_model()
        elif choice == '5':
            if model and trauma_dataset:
                show_testing_menu(model, trauma_dataset) 
            else:
                print("Error: Model or TraumaDataset not loaded.")
        elif choice == '6':
            browse_and_load_model()
        elif choice == '7':
            load_custom_feature_file()
        elif choice == '8':
            if testing:
                print('Testing mode off')
                testing = not testing
            else:
                print('Testing mode on')
                testing = not testing
        elif choice == '9':
            impute_missing_values()
        elif choice == '10':
            normalize_and_encode_dataset()
        elif choice == '11':
            evaluate_testing_records_with_ensemble(trauma_dataset)
        elif choice == '12':
            load_globals()
        elif choice == '13':
            generate_ems_judgement_roc()
        elif choice == '14':
            compute_missing_list()
        elif choice == '15':
            generate_univariate_analysis()
            print('Exiting...')
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    show_menu()
