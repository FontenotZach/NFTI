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
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import KNNImputer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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


# class IterativeImputerWithProgress(IterativeImputer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _impute_one_feature(self, X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, estimator=None):
#         with tqdm(total=self.max_iter, desc=f"Imputing feature {feat_idx+1}/{X_filled.shape[1]}") as progress_bar:
#             for i in range(self.max_iter):
#                 progress_bar.update(1)
#                 super()._impute_one_feature(
#                     X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, estimator=estimator
#                 )
#         return super()._impute_one_feature(X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, estimator=estimator)


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
    # Use tkinter to ask the user for the dataset file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    print("Please select the dataset file to load.")
    
    # Open a file dialog for the user to select their dataset
    data_file_path = filedialog.askopenfilename(
        title="Select Dataset File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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

    if customs:
        print('Loading custom headers.')
        trauma_dataset.add_custom_features(customs)

    print(f"Headers populated. Total headers: {len(trauma_dataset.get_headers())}")


    if testing:
        print("Testing mode enabled. Loading only the first 1000 records...")
        df = df.head(1000)
    
    # Populate records
    print("Populating records into TraumaDataset...")
    for _, row in df.iterrows():
        trauma_dataset.add_record(row)
    print(f"Records populated. Total records: {len(trauma_dataset.get_records())}")

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

    root = tk.Tk()
    root.withdraw()

    pickle_file_path = filedialog.askopenfilename(
        title="Select the pickled TraumaDataset file",
        filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
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

def train_model():

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

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
    
    roc_data = {}
    pr_data = {}

    # Train models for each NFTI criterion
    for header in trauma_dataset.get_headers():
        if header.y == '1':
            print(f'Generating XGBoost model for header {header.name}')
            xgboost_model, X_test, y_test = ModelGen.train_trauma_model_xgboost(trauma_dataset, header.name)
            
            # Store the model and test data
            models_xgboost_tuple[header.name] = (xgboost_model, X_test, y_test)
            models_xgboost[header.name] = xgboost_model

            print('Generating Neural Network model for header ' + header.name)
            # nn_model = ModelGen.train_trauma_model(trauma_dataset, header.name)
            nn_model = None
            models_nn[header.name] = nn_model

    # Plotting all ROC curves
    plt.figure(figsize=(10, 8))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(models_xgboost_tuple))))

    for header_name, (model, X_test, y_test) in models_xgboost_tuple.items():
        # Predict probabilities for the positive class
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
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
        # Predict probabilities for the positive class
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Calculate Precision-Recall data
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        average_precision = average_precision_score(y_test, y_pred_prob)
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

    final_model, final_cutoff = Ensemble.build_final_nfti_model(models_xgboost, models_nn, trauma_dataset)

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

    # Extract testing records from trauma_dataset
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    # Extract true labels and EMS judgments
    y_test_nfti = pd.Series([record.y['nfti_positive'] for record in testing_records])
    vpoemsjudge = pd.Series([record.data['VPOEMSJUDGE'] for record in testing_records])

    # Ensure data is clean
    if y_test_nfti.isnull().any() or vpoemsjudge.isnull().any():
        print("Missing data detected in the extracted fields. Please check the dataset.")
        return

    # Generate ROC curve for EMS judgments
    fpr_human, tpr_human, _ = roc_curve(y_test_nfti, vpoemsjudge)
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

    root = tk.Tk()
    root.withdraw()

    model_path = filedialog.askopenfilename(
        title="Select the pre-made model file",
        filetypes=(("HDF5 files", "*.h5"), ("Keras model files", "*.keras"), ("All files", "*.*"))
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
    """
    Load custom features from a file and integrate them into the trauma_dataset.
    """
    global trauma_dataset, customs
    from tkinter import filedialog

    customs = filedialog.askopenfilename(title="Select Custom Feature File", filetypes=(("CSV files", "*.csv"),))

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

class IterativeImputerWithProgress(IterativeImputer):
    def __init__(self, max_iter=10, tol=1e-3, random_state=None, **kwargs):
        super().__init__(max_iter=max_iter, tol=tol, random_state=random_state, **kwargs)

    def _impute_one_feature(self, X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, *args, **kwargs):
        """
        Override _impute_one_feature to include a progress bar.
        """
        total_features = X_filled.shape[1]
        with tqdm(total=self.max_iter, desc=f"Imputing feature {feat_idx + 1}/{total_features}") as progress_bar:
            for i in range(self.max_iter):
                progress_bar.update(1)
                # Let the parent class handle the actual imputation
                result = super()._impute_one_feature(X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, *args, **kwargs)
        return result

def impute_missing_values_mice_with_progress(trauma_dataset, max_iter=10):
    """
    Impute missing values in the TraumaDataset using MICE (IterativeImputer) from sklearn,
    separately for training and testing datasets based on the `for_testing` flag.
    """

    print('Computing missing values and outputting to missing.txt')
    compute_missing_list()

    # Separate records into training and testing based on `for_testing` flag
    train_records = [record for record in trauma_dataset.get_records() if not record.for_testing]
    test_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    # Convert training and testing records into DataFrames for easier manipulation
    df_train = pd.DataFrame([record.data for record in train_records])
    df_test = pd.DataFrame([record.data for record in test_records])

    # Lists to store column names for binary, categorical, and continuous variables
    binary_cols = []
    categorical_cols = []
    continuous_cols = []

    # Separate headers based on their type
    for header in trauma_dataset.get_headers():
        if "BIU" not in header.name:
            if header.usage == '1' and header.timing in ['1']:
                if header.data_type == '1':  # Binary
                    binary_cols.append(header.name)
                elif header.data_type == '2':  # Categorical
                    categorical_cols.append(header.name)
                elif header.data_type == '3':  # Continuous
                    continuous_cols.append(header.name)

    # Collect the features to impute
    feature_cols = binary_cols + categorical_cols + continuous_cols

    # Extract the columns for imputation from both train and test datasets
    df_train_to_impute = df_train[feature_cols]
    df_test_to_impute = df_test[feature_cols]

    # Initialize IterativeImputer with Progress
    imputer_train = IterativeImputerWithProgress(max_iter=max_iter, random_state=42)
    imputer_test = IterativeImputerWithProgress(max_iter=max_iter, random_state=42)

    print("Applying MICE imputation with progress...")

    # Fit the imputer on training data
    imputer_train.fit(df_train_to_impute)
    imputer_test.fit(df_train_to_impute)


    # Transform training and testing data separately to avoid data leakage
    train_imputed = imputer_train.transform(df_train_to_impute)
    test_imputed = imputer_test.transform(df_test_to_impute)

    # Convert imputed arrays back to DataFrames
    df_train_imputed = pd.DataFrame(train_imputed, columns=feature_cols)
    df_test_imputed = pd.DataFrame(test_imputed, columns=feature_cols)

    # Round binary and categorical columns to integers
    df_train_imputed[binary_cols + categorical_cols] = df_train_imputed[binary_cols + categorical_cols].round(0).astype(int)
    df_test_imputed[binary_cols + categorical_cols] = df_test_imputed[binary_cols + categorical_cols].round(0).astype(int)

    # Update the original DataFrames with imputed values
    df_train[feature_cols] = df_train_imputed
    df_test[feature_cols] = df_test_imputed

    # Reassign imputed values back to their corresponding records in TraumaDataset
    for i, record in enumerate(train_records):
        record.data = df_train.iloc[i].to_dict()
    for i, record in enumerate(test_records):
        record.data = df_test.iloc[i].to_dict()

    # Update TraumaDataset by combining training and testing records
    trauma_dataset.records = train_records + test_records

    print(f"Missing values have been imputed using MICE successfully with progress tracking.")

    # Save the imputed dataset as a pickle file
    ensure_dirs()
    datasets_dir = PICKLES_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    pickle_file_path = str(datasets_dir / 'trauma_dataset_imputation_mice.pkl')
    print(f"Saving the imputed pickled dataset to {pickle_file_path}...")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(trauma_dataset, f)
    print(f"Dataset pickled and saved successfully at {pickle_file_path}.")

    return trauma_dataset

def impute_missing_values_knn(trauma_dataset, n_neighbors=50, batch_size=50000):
    """
    Impute missing values in the TraumaDataset using KNN Imputer from sklearn,
    separately for training and testing datasets based on the `for_testing` flag.
    """

    print('Computing missing values and outputting to missing.txt')
    compute_missing_list(trauma_dataset)

    # Separate records into training and testing based on `for_testing` flag
    train_records = [record for record in trauma_dataset.get_records()  if not record.for_testing]
    test_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    # Convert training and testing records into DataFrames for easier manipulation
    df_train = pd.DataFrame([record.data for record in train_records])
    df_test = pd.DataFrame([record.data for record in test_records])

    # Lists to store column names for binary, categorical, and continuous variables
    binary_cols = []
    categorical_cols = []
    continuous_cols = []

    # Separate headers based on their type
    for header in trauma_dataset.get_headers():
        if "BIU" not in header.name:
            if header.usage == '1' and header.timing in ['1']: #TODO
                if header.data_type == '1':  # Binary
                    binary_cols.append(header.name)
                elif header.data_type == '2':  # Categorical
                    categorical_cols.append(header.name)
                elif header.data_type == '3':  # Continuous
                    continuous_cols.append(header.name)

    # Collect the features to impute
    feature_cols = binary_cols + categorical_cols + continuous_cols

    # Extract the columns for imputation from both train and test datasets
    df_train_to_impute = df_train[feature_cols]
    df_test_to_impute = df_test[feature_cols]

    # Initialize KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Apply KNN imputation in batches for training data
    num_batches_train = (len(df_train_to_impute) // batch_size) + 1
    train_imputed = np.empty(df_train_to_impute.shape)

    for i in tqdm(range(num_batches_train), desc="Imputing training data batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_train_to_impute))
        
        # Impute each batch of training data
        train_batch = df_train_to_impute.iloc[start_idx:end_idx]
        train_imputed[start_idx:end_idx, :] = imputer.fit_transform(train_batch)

    # Apply imputation to the testing data without refitting the imputer
    test_imputed = imputer.transform(df_test_to_impute)

    # Convert imputed arrays back to DataFrames
    df_train_imputed = pd.DataFrame(train_imputed, columns=feature_cols)
    df_test_imputed = pd.DataFrame(test_imputed, columns=feature_cols)

    df_train_imputed[binary_cols + categorical_cols] = df_train_imputed[binary_cols + categorical_cols].round(0).astype(int)
    df_test_imputed[binary_cols + categorical_cols] = df_test_imputed[binary_cols + categorical_cols].round(0).astype(int)

    # Update the original DataFrames with imputed values
    df_train[feature_cols] = df_train_imputed
    df_test[feature_cols] = df_test_imputed

    # Reassign imputed values back to their corresponding records in TraumaDataset
    for i, record in enumerate(train_records):
        record.data = df_train.iloc[i].to_dict()
    for i, record in enumerate(test_records):
        record.data = df_test.iloc[i].to_dict()

    # Update TraumaDataset by combining training and testing records
    trauma_dataset.records = train_records + test_records
    # trauma_dataset.records = train_records

    print(f"Missing values have been imputed using KNN Imputer (k={n_neighbors}) successfully.")

    # Save the imputed dataset as a pickle file
    ensure_dirs()
    datasets_dir = PICKLES_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    pickle_file_path = str(datasets_dir / 'trauma_dataset_imputation.pkl')
    print(f"Saving the imputed pickled dataset to {pickle_file_path}...")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(trauma_dataset, f)
    print(f"Dataset pickled and saved successfully at {pickle_file_path}.")

    return trauma_dataset

def one_hot_encode_trauma_dataset(trauma_dataset):
    """
    One-hot encode the categorical columns PLACEOFINJURYCODE and PRIMARYECODEICD10 in the TraumaDataset.
    """
    df = pd.DataFrame([record.data for record in trauma_dataset.get_records()])

    # Define the columns to one-hot encode
    categorical_cols = ['PLACEOFINJURYCODE', 'PRIMARYECODEICD10']

    # Filter out categorical columns that exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    if not categorical_cols:
        print("No categorical columns to one-hot encode.")
        return trauma_dataset

    # Apply one-hot encoding
    for col in categorical_cols:
        if df[col].dtype == 'object':
            # Convert the column to a category if not already categorical
            df[col] = df[col].astype('category')

        # One-hot encode the categorical columns
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)

    # Update the records in the TraumaDataset with the one-hot encoded DataFrame
    for i, record in enumerate(trauma_dataset.get_records()):
        record.data = df.iloc[i].to_dict()

    print("One-hot encoding completed successfully for PLACEOFINJURYCODE and PRIMARYECODEICD10.")

    return trauma_dataset

def evaluate_testing_records_with_ensemble(trauma_dataset):
    global models_xgboost
    global models_nn
    global meta_models
    global final_model
    global final_cutoff

    from datetime import datetime

    log_filename = f"log_ensemble_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Extract the records flagged for testing
    testing_records = [record for record in trauma_dataset.get_records() if record.for_testing]

    if not testing_records:
        print("No testing records available.")
        return

    print(f"\n--- Running {len(testing_records)} Records through Ensemble Models ---\n")

    # Initialize list to hold the meta-model inputs (XGBoost + NN predictions)
    meta_inputs = []

    # Iterate through the criteria models for XGBoost and NN
    for criterion in models_xgboost.keys():
        print(f'Getting meta-model predictions for {criterion}...')

        # Preprocess data for each criterion using the same method as during training
        X_test_binary, X_test_cat, X_test_cont, _ = Ensemble.preprocess_data_for_criterion(criterion, trauma_dataset, testing=True)

        # Get predictions from XGBoost
        xgb_pred_prob = models_xgboost[criterion].predict_proba(np.column_stack((X_test_binary, X_test_cat, X_test_cont)))[:, 1]

        # Combine predictions from XGBoost and optional NN into the meta input
        meta_parts = [xgb_pred_prob.reshape(-1, 1)]
        nn_model = models_nn.get(criterion)
        if nn_model is not None:
            nn_pred_prob = nn_model.predict([X_test_binary, X_test_cat, X_test_cont]).flatten()
            meta_parts.append(nn_pred_prob.reshape(-1, 1))

        meta_input = np.hstack(meta_parts)

        # Add to the list of meta inputs for the final ensemble
        meta_inputs.append(meta_input)

    # Stack all the meta-model predictions as input for the final ensemble model
    final_meta_input = np.hstack(meta_inputs)

    # Ensure the final meta input shape matches the expected input shape for the final model
    if final_meta_input.shape[1] != final_model.input_shape[1]:
        print(f"Shape mismatch for final model. Expected {final_model.input_shape[1]}, got {final_meta_input.shape[1]}")
        return

    print(f"Shape mismatch for final model. Expected {final_model.input_shape[1]}, got {final_meta_input.shape[1]}")

    # Get predicted probabilities for the final ensemble model
    final_pred_prob = final_model.predict(final_meta_input).flatten()

    # Print the shape of final_pred_prob
    print(f"Shape of final_pred_prob: {final_pred_prob.shape}")

    # Apply the default threshold for binary classification
    final_pred = (final_pred_prob >= final_cutoff).astype(int)

    # Print the shape of final_pred
    print(f"Shape of final_pred: {final_pred.shape}")

    # Extract the actual 'nfti_positive' labels from the records and tile them to match meta-input length
    y_test_nfti = pd.Series([record.y['nfti_positive'] for record in testing_records])

    # Print the shape of y_test_nfti
    print(f"Shape of y_test_nfti: {y_test_nfti.shape}")

    # Print the shape of final_meta_input for reference
    print(f"Shape of final_meta_input: {final_meta_input.shape}")
    # Evaluate the final ensemble predictions
    accuracy = accuracy_score(y_test_nfti, final_pred)
    precision = precision_score(y_test_nfti, final_pred)
    recall = recall_score(y_test_nfti, final_pred)
    f1 = f1_score(y_test_nfti, final_pred)
    auc = roc_auc_score(y_test_nfti, final_pred_prob)

    evaluation_results = (
        f"\n--- Final Ensemble Model Evaluation on Validation Records ---\n"
        f"Cutoff: {final_cutoff * 100:.2f}%\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall (Sensitivity): {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%\n"
        f"ROC AUC: {auc:.4f}\n"
    )
    
    print(evaluation_results)

    # Save the evaluation results to a log file
    ensure_dirs()
    log_file_path = LOGS_DIR / log_filename
    with open(log_file_path, 'a') as log_file:
        log_file.write(evaluation_results)

    print(f"Evaluation results saved to {log_file_path}")

def save_globals():
    """
    Save global variables (models_xgboost, models_nn, meta_models, final_model) as pickled files in the Model_Pickle directory.
    """
    global models_xgboost, models_nn, meta_models, final_model, final_cutoff

    ensure_dirs()
    model_dir = PICKLES_DIR / "globals"
    model_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Global models saved in {model_dir}")

def load_globals():
    """
    Load global variables (models_xgboost, models_nn, meta_models, final_model) from pickled files in the Model_Pickle directory.
    """
    global models_xgboost, models_nn, meta_models, final_model, final_cutoff

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

    print(f"Global models loaded from {model_dir}")

def show_menu():
    global trauma_dataset, model, customs, testing
    while True:
        print("\n--- Trauma Dataset Menu ---")
        print("1. Pickle the data")
        print("2. Load pickled TraumaDataset")
        print("3. Print a random record from TraumaDataset")
        print("4. Train the model using ModelGen")
        print("5. Open individual model testing interface")
        print("6. Browse and load pre-made model")
        print("7. Load custom feature file")
        print("8. Toggle testing mode")
        print("9. Run imputation algorithm")
        print("10. Run ensemble testing")
        print("11. Load globals")
        print("12. Generate EMS judgement comparison")
        print("13. Compute missing list")
        print("14. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            pickle_data()
        elif choice == '2':
            load_pickled_data()
        elif choice == '3':
            print_random_record()
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
            impute_missing_values_knn(trauma_dataset)
        elif choice == '10':
            evaluate_testing_records_with_ensemble(trauma_dataset)
        elif choice == '11':
            load_globals()
        elif choice == '12':
            generate_ems_judgement_roc()
        elif choice == '13':
            compute_missing_list()
        elif choice == '14':
            generate_univariate_analysis()
            print('Exiting...')
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    show_menu()
