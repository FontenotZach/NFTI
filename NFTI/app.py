"""Interactive entry point for the NFTI prehospital trauma-triage pipeline.

The menu walks the reproducible analysis in order:

  Data      -> build the analytic cohort and verify it.
  Modeling  -> train the primary XGBoost vs. logistic-regression models.
  Analyses  -> fidelity, missingness, and the field-triage guideline proxy.

Manuscript tables and figures are rendered by the standalone scripts in
``scripts/`` (see README). This module deliberately holds only the steps that
build state in memory or write the primary analysis artifacts.
"""

import os
import pickle
import random
import sys

import pandas as pd
import tkinter as tk
from tkinter import filedialog

sys.path.append("src")
import ModelGen

from src.TraumaDataset import TraumaDataset
from src.data.dataset_builder import build_trauma_dataset, load_header_definitions
from src.dataset_audit import (
    audit_trauma_dataset,
    format_audit_summary,
    print_header_detail,
    write_audit_report,
)
from src.paths import (
    ensure_dirs,
    PICKLES_DIR,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
    SCHEMAS_DIR,
)

# In-memory pipeline state.
trauma_dataset = None
models_xgboost = {}
customs = None
testing = False


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
        print("Testing mode enabled. Loading only the first 1000 records...")
        df = df.head(1000)

    header_info = load_header_definitions(str(SCHEMAS_DIR / "header_definitions.csv"))

    customs_path = customs or default_customs_path()
    if customs_path:
        print(f"Loading custom headers from {customs_path}.")
    else:
        print("No custom features file set; skipping derived features.")

    print("Building TraumaDataset (headers, records, cohort filter, split)...")
    trauma_dataset = build_trauma_dataset(df, header_info, customs_path=customs_path)
    print(
        f"Eligible prehospital EMS cohort ready: "
        f"{len(trauma_dataset.get_records())} records, "
        f"{len(trauma_dataset.get_headers())} headers."
    )

    datasets_dir = PICKLES_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    pickle_file_path = str(datasets_dir / "trauma_dataset.pkl")
    print(f"Saving pickled dataset to {pickle_file_path}...")
    with open(pickle_file_path, "wb") as f:
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
            with open(pickle_file_path, "rb") as f:
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
    from src.evaluation.nfti_positive_primary import run_nfti_positive_primary_evaluation
    from src.models.xgboost_shap import compute_nfti_positive_xgboost_shap

    if trauma_dataset is None:
        print("No dataset loaded. Please load a pickled dataset first.")
        return

    print("Starting nfti_positive primary model training (XGBoost + LR baseline)...")

    global models_xgboost

    try:
        result = run_nfti_positive_primary_evaluation(
            trauma_dataset,
            log_filename=ModelGen.log_filename,
        )
    except ValueError as exc:
        print(f"Primary model training failed: {exc}")
        return

    models_xgboost["nfti_positive"] = result["xgboost_model"]

    try:
        compute_nfti_positive_xgboost_shap(trauma_dataset, result["xgboost_model"])
    except Exception as exc:
        print(f"SHAP analysis for nfti_positive skipped: {exc}")

    print("Primary model training and evaluation complete.")
    save_globals()


def load_custom_feature_file():
    """Set the customs CSV path applied the next time the dataset is built."""
    global customs

    default_path = default_customs_path()
    print(
        "Select a customs CSV. It will be applied the next time you build the "
        "dataset. No dataset needs to be loaded first."
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
        print("Run 'Build dataset' to apply these features.")
        return

    print("No file selected in dialog.")
    if default_path:
        use_default = input(f"Use default customs file? [{default_path}] [Y/n]: ").strip().lower()
        if use_default in ("", "y", "yes"):
            customs = default_path
            print(f"Custom features file set: {customs}")
            return

    manual = input("Enter path to customs CSV (or press Enter to cancel): ").strip()
    if manual and os.path.isfile(manual):
        customs = manual
        print(f"Custom features file set: {customs}")
    elif manual:
        print(f"File not found: {manual}")
    else:
        print("Custom features file unchanged.")
        if customs:
            print(f"Current setting: {customs}")
        elif default_path:
            print(f"The build will still use the default: {default_path}")


def run_vital_fidelity_audit():
    """Audit prehospital EMS vital-sign fidelity on the loaded TraumaDataset.

    Run BEFORE imputation/one-hot/normalization so figures reflect raw
    documented values. The audit never mutates the dataset and writes to
    artifacts/figures/fidelity and artifacts/tables/fidelity.
    """
    if trauma_dataset is None:
        print("No dataset loaded. Pickle or load a TraumaDataset first.")
        return

    from src.evaluation.fidelity_audit import run_prehospital_vital_fidelity_audit

    transform_state = getattr(trauma_dataset, "transform_state", None)
    if transform_state and transform_state.get("applied"):
        print(
            "WARNING: this dataset was already normalized/one-hot encoded. "
            "For a fully raw audit, load the pre-transform pickle "
            "(trauma_dataset.pkl) instead."
        )

    print("Running prehospital vital-sign fidelity audit (raw documented values)...")
    try:
        run_prehospital_vital_fidelity_audit(trauma_dataset)
    except ValueError as exc:
        print(f"Fidelity audit failed: {exc}")


def run_missingness_data_audit():
    """Run the robust missing-data audit from the CLI.

    Reads the RAW cohort CSV (data/raw/dat5.csv) directly and re-applies the
    prehospital EMS cohort filter, so it is independent of the in-memory dataset
    state and never uses normalized/one-hot values. Analysis 5 (model
    performance by missingness burden) reads the saved holdout predictions, so
    run this AFTER training.
    """
    from src.evaluation.missingness_audit import run_missingness_audit

    print("Running missing-data audit on RAW cohort CSV (data/raw/dat5.csv).")
    print("Analysis 5 needs holdout predictions, so run AFTER training.")
    try:
        run_missingness_audit()
    except Exception as exc:
        print(f"Missingness audit failed: {exc}")


def run_guideline_proxy_benchmark_menu():
    """Run the exploratory 2021 Field Triage Guideline proxy benchmark.

    Reads the RAW cohort CSV (data/raw/dat5.csv) directly, re-applies the
    prehospital EMS cohort filter, and inner-joins the saved XGBoost holdout
    predictions on record_id, so it uses only raw prehospital values. Requires
    holdout predictions, so run this AFTER primary training. VPOEMSJUDGE and
    post-arrival variables are never used.
    """
    from src.evaluation.guideline_proxy_benchmark import run_guideline_proxy_benchmark

    print("Running 2021 field triage guideline proxy benchmark (exploratory).")
    print("Requires holdout predictions, so run AFTER training.")
    try:
        run_guideline_proxy_benchmark()
    except Exception as exc:
        print(f"Guideline proxy benchmark failed: {exc}")


def register_one_hot_feature_labels():
    """Register the model's one-hot feature names in human_readable_headers.csv
    with BLANK labels so they can be named via the label manager.
    """
    if trauma_dataset is None:
        print("No dataset loaded. Pickle or load a TraumaDataset first.")
        return

    from src.preprocessing.feature_preprocessor import (
        _get_feature_column_groups,
        get_model_feature_names,
    )
    from src.data.human_readable import (
        HUMAN_READABLE_PATH,
        append_human_readable_entries,
    )

    binary_cols, categorical_cols, continuous_cols = _get_feature_column_groups(trauma_dataset)
    if not categorical_cols:
        print("No categorical model columns found; nothing to register.")
        return

    raw = set(binary_cols) | set(continuous_cols)
    one_hot_names = [n for n in get_model_feature_names(trauma_dataset) if n not in raw]
    if not one_hot_names:
        print("No one-hot model features found (no categorical columns).")
        return

    added = append_human_readable_entries({name: "" for name in one_hot_names})
    print(f"Registered {added} one-hot name(s) in {HUMAN_READABLE_PATH} (labels blank).")
    print("Now run the human-readable label manager to name them interactively.")


def manage_human_readable_labels():
    """Interactively name headers that lack a human-readable label.

    Walks two groups: model headers (usage=1 / y=1) absent from the label file,
    and rows already in the file whose label is blank (e.g. one-hot feature
    names). For one-hot-pattern names a best-guess label is suggested; pressing
    Enter accepts it. Entries are persisted with an upsert.
    """
    if trauma_dataset is None:
        print("No dataset loaded. Pickle or load a TraumaDataset first.")
        return

    from src.data.human_readable import (
        HUMAN_READABLE_PATH,
        build_one_hot_label,
        find_blank_label_headers,
        find_untracked_headers,
        load_human_readable_map,
        set_human_readable_labels,
    )
    from src.preprocessing.feature_preprocessor import _get_feature_column_groups
    from src.preprocessing.dataset_transform import _source_column_for_ohe_name

    candidates = [
        header.name
        for header in trauma_dataset.get_headers()
        if header.usage == "1" or header.y == "1"
    ]
    untracked = find_untracked_headers(candidates)
    blanks = find_blank_label_headers()

    seen = set()
    to_name = []
    for header in [*untracked, *blanks]:
        if header in seen:
            continue
        seen.add(header)
        to_name.append(header)

    if not to_name:
        print("All model headers and registered one-hot names already have labels.")
        return

    _, categorical_cols, _ = _get_feature_column_groups(trauma_dataset)
    hr_map = load_human_readable_map()

    def _suggestion_for(header: str) -> str:
        if not categorical_cols:
            return ""
        source_col = _source_column_for_ohe_name(header, categorical_cols)
        if source_col and header.startswith(f"{source_col}_"):
            return build_one_hot_label(header, source_col, hr_map)
        return ""

    print(f"\n{len(to_name)} header(s) need a human-readable label.")
    print(
        "Type a label to set it; press Enter to accept a suggested label "
        "(or skip when none); 'q' = stop and save.\n"
    )

    new_entries = {}
    for header in to_name:
        suggestion = _suggestion_for(header)
        prompt = f"  {header}" + (f" [Enter = '{suggestion}']: " if suggestion else ": ")
        try:
            response = input(prompt).strip()
        except EOFError:
            break
        if response.lower() == "q":
            break
        value = response or suggestion
        if value:
            new_entries[header] = value

    changed = set_human_readable_labels(new_entries)
    skipped = len(to_name) - len(new_entries)
    print(f"\nSaved {changed} label(s) to {HUMAN_READABLE_PATH}.")
    if skipped:
        print(f"{skipped} header(s) left unlabeled (you can run this option again later).")


def reset_output_artifacts():
    """Delete generated output artifacts for a clean run, preserving the dataset
    pickle (artifacts/pickles/datasets) and all inputs.
    """
    import shutil
    from src.paths import (
        FIGURES_DIR,
        TABLES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        PREDICTIONS_ARTIFACTS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        EXPORT_DIR,
        ARCHIVE_DIR,
        RESULTS_DIR,
        PICKLES_DIR,
    )

    targets = [
        FIGURES_DIR,
        TABLES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        PREDICTIONS_ARTIFACTS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        EXPORT_DIR,
        ARCHIVE_DIR,
        RESULTS_DIR,
        PICKLES_DIR / "globals",
    ]

    print("This deletes generated outputs but KEEPS artifacts/pickles/datasets/ and all inputs.")
    if input("Type 'reset' to confirm: ").strip().lower() != "reset":
        print("Reset cancelled.")
        return

    for d in targets:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    ensure_dirs()
    print("Output artifacts reset. Dataset pickle preserved.")


def save_globals():
    """Save trained XGBoost models as a pickled file."""
    global models_xgboost

    ensure_dirs()
    model_dir = PICKLES_DIR / "globals"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "models_xgboost.pkl", "wb") as f:
        pickle.dump(models_xgboost, f)

    print(f"XGBoost models saved in {model_dir}")


def load_globals():
    """Load trained XGBoost models from pickled file."""
    global models_xgboost

    ensure_dirs()
    model_dir = PICKLES_DIR / "globals"

    with open(model_dir / "models_xgboost.pkl", "rb") as f:
        models_xgboost = pickle.load(f)

    print(f"XGBoost models loaded from {model_dir}")


MENU_ACTIONS = {
    "1": pickle_data,
    "2": load_pickled_data,
    "3": show_dataset_audit_menu,
    "4": load_custom_feature_file,
    "6": train_model,
    "7": load_globals,
    "8": run_vital_fidelity_audit,
    "9": run_missingness_data_audit,
    "10": run_guideline_proxy_benchmark_menu,
    "11": register_one_hot_feature_labels,
    "12": manage_human_readable_labels,
    "13": reset_output_artifacts,
}


def show_menu():
    global testing
    while True:
        print("\n=== NFTI Prehospital Trauma-Triage Pipeline ===")
        print("Data")
        print("  1. Build dataset from raw CSV (cohort filter + split, then pickle)")
        print("  2. Load pickled dataset")
        print("  3. Audit / verify dataset")
        print("  4. Set custom-features CSV (optional; default: data/schemas/customs.csv)")
        print(f"  5. Toggle testing mode (currently: {'ON' if testing else 'OFF'})")
        print("Modeling")
        print("  6. Train primary NFTI models (XGBoost vs LR) + SHAP")
        print("  7. Load saved XGBoost models")
        print("Analyses")
        print("  8. EMS vital-sign fidelity audit (run before transforms)")
        print("  9. Missing-data audit (run after training)")
        print("  10. 2021 field-triage guideline proxy benchmark (run after training)")
        print("Labels & maintenance")
        print("  11. Register model one-hot feature names")
        print("  12. Manage human-readable header labels")
        print("  13. Reset output artifacts (keeps dataset pickle)")
        print("  0. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Exiting...")
            break
        if choice == "5":
            testing = not testing
            print(f"Testing mode {'on' if testing else 'off'}")
            continue

        action = MENU_ACTIONS.get(choice)
        if action is None:
            print("Invalid choice. Please try again.")
        else:
            action()


if __name__ == "__main__":
    show_menu()
