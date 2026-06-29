"""Missing-data audit for the trauma triage (NFTI) ML manuscript.

This module implements a robust, manuscript/supplement-ready missingness audit
that answers four questions:

    1. Is missingness isolated or block-patterned?
    2. Does missingness differ by clinical context?
    3. Is missingness predictable / systematic (vs completely random)?
    4. Does the NFTI model perform differently when EMS vitals are incomplete?

Design principles (mirroring ``src/evaluation/fidelity_audit.py``):

* It is intentionally separate from the model training pipeline. It never
  retrains the primary model, never mutates ``record.data``, and only writes
  under ``artifacts/{figures,tables,metrics}/missingness``.
* Missingness is computed on RAW (pre-imputation / pre-scaling / pre-one-hot)
  values. By default it loads the raw cohort CSV and re-applies the same
  prehospital EMS cohort filter used by the pipeline, so record IDs align with
  the saved holdout predictions. A pre-transform pickle is also supported.
* Every column is resolved against the columns that are actually present.
  Missing expected columns produce a warning and are skipped (never crash).

Outputs:
    Figures -> artifacts/figures/missingness
    Tables  -> artifacts/tables/missingness
    Metrics -> artifacts/metrics/missingness
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: write figures without requiring a display
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # SciPy ships with scikit-learn; degrade gracefully if unavailable.
    from scipy.stats import chi2_contingency, fisher_exact

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _SCIPY_AVAILABLE = False

from src.data.human_readable import load_human_readable_map
from src.evaluation.binary_metrics import (
    evaluate_binary_classifier,
    record_id_from_trauma_record,
)

# Reuse the publication-quality plotting defaults already used by the fidelity
# audit so figures share one visual language across the manuscript.
from src.evaluation.fidelity_audit import FIGURE_DPI, RANDOM_SEED, _PLOT_RC
from src.plotting import apply_manuscript_grid
from src.paths import (
    MISSINGNESS_FIGURES_DIR,
    MISSINGNESS_METRICS_DIR,
    MISSINGNESS_TABLES_DIR,
    NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH,
    NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# Column definitions.
#
# Names follow the documented NTDB/TQIP headers used throughout this project
# (see data/schemas/header_definitions.csv). UPDATE THESE if your dataset build
# uses different column names. Every name is resolved against the columns that
# are actually present at runtime; absent columns are reported and skipped.
# ---------------------------------------------------------------------------
OUTCOME_COLUMN = "nfti_positive"

# Core EMS prehospital vitals (used for burden / pattern analyses).
EMS_VITAL_COLUMNS: List[str] = [
    "EMSSBP",
    "EMSPULSERATE",
    "EMSRESPIRATORYRATE",
    "EMSPULSEOXIMETRY",
    "EMSTOTALGCS",
]

# ED/hospital arrival vitals matched to the EMS vitals.
HOSPITAL_VITAL_COLUMNS: List[str] = [
    "SBP",
    "PULSERATE",
    "RESPIRATORYRATE",
    "PULSEOXIMETRY",
    "TOTALGCS",
    "TEMPERATURE",
]

NFTI_SUBCRITERIA_COLUMNS: List[str] = [
    "nfti_prbc",
    "nfti_OR",
    "nfti_ICU",
    "nfti_IR",
    "nfti_death",
    "nfti_intubate",
]

# Race one-hot columns (0 = no, 1 = yes) and ETHNICITY (1 = Hispanic,
# 2 = non-Hispanic). These are used ONLY for descriptive missingness
# stratification (is missingness dependent on race/ethnicity?). They are
# deliberately NEVER used as predictors in any model -- race/ethnicity are
# excluded from all models to avoid encoding racial bias.
RACE_COLUMNS: List[str] = [
    "AMERICANINDIAN",
    "ASIAN",
    "BLACK",
    "PACIFICISLANDER",
    "RACEOTHER",
    "WHITE",
]
RACE_LABELS: Dict[str, str] = {
    "AMERICANINDIAN": "American Indian",
    "ASIAN": "Asian",
    "BLACK": "Black",
    "PACIFICISLANDER": "Pacific Islander",
    "RACEOTHER": "Other race",
    "WHITE": "White",
}

# Clinically meaningful feature groups. Resolved against available columns.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "EMS vitals": list(EMS_VITAL_COLUMNS) + ["EMSGCSEYE", "EMSGCSMOTOR", "EMSGCSVERBAL"],
    "Hospital vitals": list(HOSPITAL_VITAL_COLUMNS),
    "Demographics": [
        "SEX",
        "AGEyears",
        "ETHNICITY",
        "AMERICANINDIAN",
        "ASIAN",
        "BLACK",
        "PACIFICISLANDER",
        "RACEOTHER",
        "WHITE",
        "RACE_NA",
        "RACE_UK",
        "PRIMARYMETHODPAYMENT",
    ],
    "Mechanism": ["MECHANISM", "TRAUMATYPE", "INTENT", "PRIMARYECODEICD10"],
    "Transport / transfer": [
        "TRANSPORTMODE",
        "INTERFACILITYTRANSFER",
        "TM_GROUNDAMBULANCE",
        "TM_HELICOPTERAMBULANCE",
        "TM_FIXEDWINGAMBULANCE",
        "PREHOSPITALCARDIACARREST",
    ],
    "Injury / severity proxies": ["ISS", "ISSVersion", "LOWESTSBP", "TEMPERATURE"],
    "Outcomes / NFTI criteria": [OUTCOME_COLUMN] + list(NFTI_SUBCRITERIA_COLUMNS),
}

# Columns that define the prehospital EMS cohort filter (must be read so the
# filter reproduces the exact modeling cohort + record ordering).
COHORT_COLUMNS: List[str] = ["TRANSPORTMODE", "INTERFACILITYTRANSFER"]

# Extra columns used for clinical-context stratification / prediction models.
EXTRA_CONTEXT_COLUMNS: List[str] = [
    "AGEyears",
    "SEX",
    "MECHANISM",
    "TRAUMATYPE",
    "TRANSPORTMODE",
    "INTERFACILITYTRANSFER",
    "ETHNICITY",
    "PREHOSPITALCARDIACARREST",
    "ISS",
    "LOWESTSBP",
]

# Baseline (non-outcome, non-EMS-vital) predictors for the "missingness as a
# prediction target" models. EMS vitals are deliberately excluded to avoid
# trivially predicting block co-missingness, and outcome / NFTI subcriteria are
# excluded to avoid leakage. Race AND ethnicity are intentionally excluded from
# every model so missingness models cannot encode racial/ethnic bias; race and
# ethnicity are analyzed descriptively (as context strata) only.
PREDICTOR_CONTINUOUS: List[str] = [
    "AGEyears",
    "ISS",
    "LOWESTSBP",
] + list(HOSPITAL_VITAL_COLUMNS)
PREDICTOR_CATEGORICAL: List[str] = [
    "SEX",
    "MECHANISM",
    "TRAUMATYPE",
    "TRANSPORTMODE",
    "INTERFACILITYTRANSFER",
    "PREHOSPITALCARDIACARREST",
]

# Compact, plot-friendly labels for the core EMS vitals.
EMS_SHORT_LABELS: Dict[str, str] = {
    "EMSSBP": "EMS SBP",
    "EMSPULSERATE": "EMS HR",
    "EMSRESPIRATORYRATE": "EMS RR",
    "EMSPULSEOXIMETRY": "EMS SpO2",
    "EMSTOTALGCS": "EMS GCS",
}
HOSPITAL_SHORT_LABELS: Dict[str, str] = {
    "SBP": "ED SBP",
    "PULSERATE": "ED HR",
    "RESPIRATORYRATE": "ED RR",
    "PULSEOXIMETRY": "ED SpO2",
    "TOTALGCS": "ED GCS",
    "TEMPERATURE": "ED Temp",
}

SMALL_N_THRESHOLD = 30
# Cap rows used for prediction-model fitting (full tables/metrics still computed
# on the entire dataset). Keeps 7 logistic regressions tractable on large data.
MODEL_FIT_MAX_N = 150_000
# Maximum rows drawn (seeded) when rendering row-level missingness heatmaps.
HEATMAP_MAX_ROWS = 8_000
PRIMARY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MissingnessAuditResult:
    data_source: str
    source_path: str
    n_records: int
    processed_data_warning: bool = False
    ems_vitals_found: List[str] = field(default_factory=list)
    hospital_vitals_found: List[str] = field(default_factory=list)
    outcome_available: bool = False
    figures_saved: List[Path] = field(default_factory=list)
    tables_saved: List[Path] = field(default_factory=list)
    metrics_saved: List[Path] = field(default_factory=list)
    missing_expected_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped_analyses: List[str] = field(default_factory=list)
    # Headline numbers captured for the summary report.
    headline: Dict[str, object] = field(default_factory=dict)


@dataclass
class MissingnessData:
    df: pd.DataFrame  # raw, pre-imputation, cohort-filtered; includes record_id
    source: str
    source_path: str
    available_columns: set
    outcome_available: bool
    processed_warning: bool
    notes: List[str]


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------
def _save_fig(fig, path: Path, result: MissingnessAuditResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    result.figures_saved.append(path)
    return path


def _save_table(df: pd.DataFrame, path: Path, result: MissingnessAuditResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    result.tables_saved.append(path)
    return path


def _save_metric(df: pd.DataFrame, path: Path, result: MissingnessAuditResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    result.metrics_saved.append(path)
    return path


def _warn(result: MissingnessAuditResult, message: str) -> None:
    result.warnings.append(message)
    warnings.warn(message, UserWarning, stacklevel=2)


def _label(header: str, hr_map: Dict[str, str]) -> str:
    if header in EMS_SHORT_LABELS:
        return EMS_SHORT_LABELS[header]
    if header in HOSPITAL_SHORT_LABELS:
        return HOSPITAL_SHORT_LABELS[header]
    label = hr_map.get(header)
    return label if label else header


# ---------------------------------------------------------------------------
# load_data()
# ---------------------------------------------------------------------------
def _all_needed_columns() -> List[str]:
    needed: List[str] = []
    for cols in FEATURE_GROUPS.values():
        needed.extend(cols)
    needed.extend(EMS_VITAL_COLUMNS)
    needed.extend(HOSPITAL_VITAL_COLUMNS)
    needed.extend(NFTI_SUBCRITERIA_COLUMNS)
    needed.extend(COHORT_COLUMNS)
    needed.extend(EXTRA_CONTEXT_COLUMNS)
    needed.extend(PREDICTOR_CONTINUOUS)
    needed.extend(PREDICTOR_CATEGORICAL)
    needed.append(OUTCOME_COLUMN)
    # De-duplicate preserving order.
    return list(dict.fromkeys(needed))


def _resolve_default_input() -> Path:
    primary = RAW_DATA_DIR / "dat5.csv"
    if primary.exists():
        return primary
    sample = SAMPLES_DATA_DIR / "dat5_limited.csv"
    return sample


def _load_from_csv(path: Path, max_records: Optional[int]) -> MissingnessData:
    notes: List[str] = []
    header_only = pd.read_csv(path, nrows=0)
    available_all = list(header_only.columns)
    needed = [c for c in _all_needed_columns() if c in available_all]
    # Always read cohort columns when present.
    for col in COHORT_COLUMNS:
        if col in available_all and col not in needed:
            needed.append(col)

    df = pd.read_csv(path, usecols=needed, low_memory=False)
    if max_records is not None:
        # Mirror the pipeline's testing behaviour: head() BEFORE cohort filter.
        df = df.head(max_records).copy()
        notes.append(
            f"max_records={max_records} applied (head before cohort filter); "
            "record IDs will NOT align with the full-cohort holdout predictions."
        )

    # Re-apply the prehospital EMS cohort filter to reproduce the modeling
    # cohort and its record ordering (so record IDs match saved predictions).
    from src.preprocessing.cohort_filter import apply_prehospital_ems_cohort_filter

    eligible_df, stats = apply_prehospital_ems_cohort_filter(df)
    eligible_df = eligible_df.reset_index(drop=True)
    notes.append(
        "Prehospital EMS cohort filter applied "
        f"({stats.records_before:,} -> {stats.records_eligible:,} records)."
    )

    eligible_df.insert(0, "record_id", [f"row_{i}" for i in range(len(eligible_df))])
    outcome_available = (
        OUTCOME_COLUMN in eligible_df.columns
        and not eligible_df[OUTCOME_COLUMN].isna().all()
    )
    return MissingnessData(
        df=eligible_df,
        source="raw_csv",
        source_path=str(path),
        available_columns=set(eligible_df.columns),
        outcome_available=outcome_available,
        processed_warning=False,
        notes=notes,
    )


def _load_from_pickle(path: Path, max_records: Optional[int]) -> MissingnessData:
    notes: List[str] = []
    with open(path, "rb") as handle:
        trauma_dataset = pickle.load(handle)

    records = trauma_dataset.get_records()
    transform_applied = bool(
        getattr(trauma_dataset, "transform_state", None)
        and trauma_dataset.transform_state.get("applied")
    )
    imputation_applied = getattr(trauma_dataset, "imputation_state", None) is not None
    processed_warning = transform_applied or imputation_applied
    if imputation_applied:
        notes.append(
            "WARNING: dataset has an imputation_state; 'missing' counts reflect "
            "POST-imputation values and may understate true missingness."
        )
    if transform_applied:
        notes.append(
            "WARNING: dataset transforms (z-score/one-hot) were applied; reading "
            "pre-scale values from record.base_data where available."
        )

    if not records:
        empty = pd.DataFrame(columns=["record_id", OUTCOME_COLUMN])
        return MissingnessData(empty, "pickle", str(path), set(empty.columns), False, processed_warning, notes)

    def _store(record):
        base = getattr(record, "base_data", None)
        return base if base is not None else record.data

    available = set(_store(records[0]).keys()) | set(records[0].data.keys())
    needed = [c for c in _all_needed_columns() if c in available]

    iterator = records if max_records is None else records[:max_records]
    if max_records is not None:
        notes.append(
            f"max_records={max_records} applied; record IDs will not align with "
            "full-cohort holdout predictions."
        )

    rows = []
    for idx, record in enumerate(iterator):
        store = _store(record)
        row = {col: store.get(col, np.nan) for col in needed}
        row["record_id"] = record_id_from_trauma_record(record, idx)
        y = getattr(record, "y", None)
        row[OUTCOME_COLUMN] = (y or {}).get(OUTCOME_COLUMN, np.nan) if isinstance(y, dict) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["record_id"] + [c for c in df.columns if c != "record_id"]
    df = df[cols]
    outcome_available = OUTCOME_COLUMN in df.columns and not df[OUTCOME_COLUMN].isna().all()
    return MissingnessData(
        df=df,
        source="pickle",
        source_path=str(path),
        available_columns=set(df.columns),
        outcome_available=outcome_available,
        processed_warning=processed_warning,
        notes=notes,
    )


def load_data(
    input_path: Optional[Path] = None,
    *,
    max_records: Optional[int] = None,
) -> MissingnessData:
    """Load raw, pre-imputation data for the missingness audit.

    Defaults to the raw cohort CSV (``data/raw/dat5.csv``); a ``.pkl``
    pre-transform :class:`TraumaDataset` is also accepted.
    """
    path = Path(input_path) if input_path is not None else _resolve_default_input()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".pkl":
        return _load_from_pickle(path, max_records)
    return _load_from_csv(path, max_records)


# ---------------------------------------------------------------------------
# resolve_feature_groups()
# ---------------------------------------------------------------------------
def resolve_feature_groups(
    available_columns: set,
    result: MissingnessAuditResult,
    tables_dir: Path,
) -> Dict[str, List[str]]:
    rows: List[Dict] = []
    resolved: Dict[str, List[str]] = {}
    for group, expected_cols in FEATURE_GROUPS.items():
        present: List[str] = []
        for col in expected_cols:
            found = col in available_columns
            note = "" if found else "expected column not present in dataset"
            rows.append(
                {
                    "feature_group": group,
                    "column": col,
                    "found_in_dataset": bool(found),
                    "note": note,
                }
            )
            if found:
                present.append(col)
            else:
                result.missing_expected_columns.append(col)
        resolved[group] = present
        if not present:
            _warn(result, f"Feature group '{group}' has no available columns; skipping it.")

    _save_table(
        pd.DataFrame(rows, columns=["feature_group", "column", "found_in_dataset", "note"]),
        tables_dir / "resolved_feature_groups.csv",
        result,
    )
    return resolved


# ---------------------------------------------------------------------------
# Analysis 1: missingness by feature group
# ---------------------------------------------------------------------------
def _present(cols: Sequence[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]


def compute_variable_missingness(
    df: pd.DataFrame,
    resolved: Dict[str, List[str]],
    hr_map: Dict[str, str],
    result: MissingnessAuditResult,
    tables_dir: Path,
) -> pd.DataFrame:
    n_total = len(df)
    rows: List[Dict] = []
    for group, cols in resolved.items():
        for col in cols:
            n_missing = int(df[col].isna().sum())
            n_nonmissing = n_total - n_missing
            rows.append(
                {
                    "feature_group": group,
                    "variable": col,
                    "variable_label": _label(col, hr_map),
                    "n_total": n_total,
                    "n_missing": n_missing,
                    "pct_missing": (100.0 * n_missing / n_total) if n_total else np.nan,
                    "n_nonmissing": n_nonmissing,
                    "pct_nonmissing": (100.0 * n_nonmissing / n_total) if n_total else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    _save_table(out, tables_dir / "variable_missingness_by_group.csv", result)
    return out


def compute_group_missingness(
    df: pd.DataFrame,
    resolved: Dict[str, List[str]],
    result: MissingnessAuditResult,
    tables_dir: Path,
) -> pd.DataFrame:
    n_total = len(df)
    rows: List[Dict] = []
    ems_core = _present(EMS_VITAL_COLUMNS, df)
    for group, cols in resolved.items():
        if not cols:
            rows.append(
                {
                    "feature_group": group,
                    "n_variables_in_group": 0,
                    "n_records": n_total,
                    "pct_records_with_any_missing_in_group": np.nan,
                    "pct_records_with_all_missing_in_group": np.nan,
                    "median_n_missing_in_group": np.nan,
                    "iqr_n_missing_in_group": np.nan,
                    "mean_n_missing_in_group": np.nan,
                    "pct_complete_case_for_group": np.nan,
                    "pct_complete_case_for_required_core_group": np.nan,
                }
            )
            continue
        sub = df[cols]
        miss = sub.isna()
        n_missing_per_row = miss.sum(axis=1)
        n_in_group = len(cols)
        any_missing = (n_missing_per_row > 0).mean() * 100.0 if n_total else np.nan
        all_missing = (n_missing_per_row == n_in_group).mean() * 100.0 if n_total else np.nan
        complete = (n_missing_per_row == 0).mean() * 100.0 if n_total else np.nan
        core_complete = np.nan
        if group == "EMS vitals" and ems_core:
            core_complete = (df[ems_core].isna().sum(axis=1) == 0).mean() * 100.0
        rows.append(
            {
                "feature_group": group,
                "n_variables_in_group": n_in_group,
                "n_records": n_total,
                "pct_records_with_any_missing_in_group": any_missing,
                "pct_records_with_all_missing_in_group": all_missing,
                "median_n_missing_in_group": float(n_missing_per_row.median()),
                "iqr_n_missing_in_group": float(
                    n_missing_per_row.quantile(0.75) - n_missing_per_row.quantile(0.25)
                ),
                "mean_n_missing_in_group": float(n_missing_per_row.mean()),
                "pct_complete_case_for_group": complete,
                "pct_complete_case_for_required_core_group": core_complete,
            }
        )
    out = pd.DataFrame(rows)
    _save_table(out, tables_dir / "group_missingness_summary.csv", result)
    return out


def compute_ems_burden(
    df: pd.DataFrame,
    result: MissingnessAuditResult,
    tables_dir: Path,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return per-patient EMS missing count and a burden distribution table."""
    ems_cols = _present(EMS_VITAL_COLUMNS, df)
    if not ems_cols:
        _warn(result, "No EMS vital columns available; EMS burden analysis skipped.")
        return pd.Series(np.nan, index=df.index), pd.DataFrame()

    n_missing = df[ems_cols].isna().sum(axis=1)
    n_ems = len(ems_cols)
    dist_rows: List[Dict] = []
    outcome = df[OUTCOME_COLUMN] if result.outcome_available else None
    for k in range(n_ems + 1):
        mask = n_missing == k
        count = int(mask.sum())
        row = {
            "n_ems_vitals_missing": k,
            "count": count,
            "percent": (100.0 * count / len(df)) if len(df) else np.nan,
        }
        if outcome is not None:
            grp = pd.to_numeric(outcome[mask], errors="coerce")
            row["nfti_positive_rate"] = float(grp.mean()) if grp.notna().any() else np.nan
        dist_rows.append(row)
    dist = pd.DataFrame(dist_rows)
    _save_table(dist, tables_dir / "ems_vital_missingness_burden_distribution.csv", result)
    return n_missing, dist


def plot_variable_missingness(
    variable_df: pd.DataFrame, result: MissingnessAuditResult, figures_dir: Path
) -> None:
    if variable_df.empty:
        return
    data = variable_df.sort_values(["feature_group", "pct_missing"], ascending=[True, True])
    groups = list(dict.fromkeys(data["feature_group"]))
    cmap = plt.get_cmap("tab10")
    color_for = {g: cmap(i % 10) for i, g in enumerate(groups)}
    colors = [color_for[g] for g in data["feature_group"]]
    height = max(4.0, 0.28 * len(data))
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(10, height))
        y = np.arange(len(data))
        ax.barh(y, data["pct_missing"], color=colors, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(data["variable_label"], fontsize=8)
        ax.set_xlabel("Percent missing (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Missingness by variable, grouped by feature group")
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[g]) for g in groups]
        ax.legend(handles, groups, fontsize=8, loc="lower right", title="Feature group")
        fig.tight_layout()
        _save_fig(fig, figures_dir / "variable_missingness_by_group.png", result)


def plot_group_missingness(
    group_df: pd.DataFrame, result: MissingnessAuditResult, figures_dir: Path
) -> None:
    if group_df.empty:
        return
    data = group_df[group_df["n_variables_in_group"] > 0].copy()
    if data.empty:
        return
    data = data.sort_values("pct_complete_case_for_group", ascending=True)
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * len(data))))
        y = np.arange(len(data))
        ax.barh(y - 0.2, data["pct_complete_case_for_group"], height=0.4,
                color="#55A868", label="Complete-case (%)")
        ax.barh(y + 0.2, data["pct_records_with_any_missing_in_group"], height=0.4,
                color="#C44E52", label="Any missing (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(data["feature_group"])
        ax.set_xlabel("Percent of records (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Group-level missingness: complete-case vs any-missing")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        _save_fig(fig, figures_dir / "group_missingness_summary.png", result)


def plot_ems_burden(
    dist: pd.DataFrame, result: MissingnessAuditResult, figures_dir: Path
) -> None:
    if dist.empty:
        return
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_manuscript_grid(ax)
        bars = ax.bar(dist["n_ems_vitals_missing"], dist["percent"],
                      color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Number of EMS vitals missing (per patient)")
        ax.set_ylabel("Percent of patients (%)")
        ax.set_xticks(dist["n_ems_vitals_missing"])
        ax.set_title("EMS vital missingness burden distribution")
        for rect, pct in zip(bars, dist["percent"]):
            ax.annotate(f"{pct:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "ems_vital_missingness_burden_distribution.png", result)


# ---------------------------------------------------------------------------
# Analysis 2: missingness pattern analysis
# ---------------------------------------------------------------------------
def _pattern_table(
    df: pd.DataFrame,
    cols: List[str],
    hr_map: Dict[str, str],
    outcome: Optional[pd.Series],
    top_n: int = 20,
) -> pd.DataFrame:
    miss = df[cols].isna()
    labels = [_label(c, hr_map) for c in cols]
    n_total = len(df)

    def pattern_for(row_mask: np.ndarray) -> Tuple[str, str, int]:
        missing_labels = [labels[i] for i, m in enumerate(row_mask) if m]
        n = len(missing_labels)
        if n == 0:
            return "None missing", "", 0
        if n == len(cols):
            return "All missing", " + ".join(missing_labels), n
        return " + ".join(missing_labels), " + ".join(missing_labels), n

    pat_strings = []
    pat_vars = []
    pat_n = []
    miss_values = miss.to_numpy()
    for row_mask in miss_values:
        s, v, n = pattern_for(row_mask)
        pat_strings.append(s)
        pat_vars.append(v)
        pat_n.append(n)

    work = pd.DataFrame({"pattern": pat_strings, "variables_missing": pat_vars, "n_missing_variables": pat_n})
    if outcome is not None:
        work["_y"] = pd.to_numeric(outcome.to_numpy(), errors="coerce")

    grouped = work.groupby(["pattern", "variables_missing", "n_missing_variables"], dropna=False)
    agg = grouped.size().reset_index(name="count")
    if outcome is not None:
        rate = grouped["_y"].mean().reset_index(name="nfti_positive_rate")
        agg = agg.merge(rate, on=["pattern", "variables_missing", "n_missing_variables"], how="left")
    agg["percent"] = 100.0 * agg["count"] / n_total if n_total else np.nan
    agg = agg.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    ordered = ["pattern", "variables_missing", "n_missing_variables", "count", "percent"]
    if "nfti_positive_rate" in agg.columns:
        ordered.append("nfti_positive_rate")
    return agg[ordered]


def compute_missingness_patterns(
    df: pd.DataFrame,
    hr_map: Dict[str, str],
    result: MissingnessAuditResult,
    figures_dir: Path,
    tables_dir: Path,
) -> Optional[pd.DataFrame]:
    ems_cols = _present(EMS_VITAL_COLUMNS, df)
    hosp_cols = _present(HOSPITAL_VITAL_COLUMNS, df)
    outcome = df[OUTCOME_COLUMN] if result.outcome_available else None

    ems_patterns = None
    if ems_cols:
        ems_patterns = _pattern_table(df, ems_cols, hr_map, outcome)
        _save_table(ems_patterns, tables_dir / "ems_vital_missingness_patterns_top.csv", result)
        _plot_pattern_bars(
            ems_patterns,
            "Top EMS vital missingness patterns",
            figures_dir / "ems_vital_missingness_patterns_top.png",
            result,
        )
    else:
        _warn(result, "No EMS vitals; EMS missingness pattern analysis skipped.")

    vital_cols = ems_cols + hosp_cols
    if vital_cols:
        vital_patterns = _pattern_table(df, vital_cols, hr_map, outcome)
        _save_table(vital_patterns, tables_dir / "vital_missingness_patterns_top.csv", result)

    # Heatmaps (sampled rows for legibility; tables use the full dataset).
    if ems_cols:
        _plot_missingness_heatmap(
            df, ems_cols, hr_map,
            "EMS vital missingness (sampled rows)",
            figures_dir / "ems_vital_missingness_heatmap.png", result,
        )
    if vital_cols:
        _plot_missingness_heatmap(
            df, vital_cols, hr_map,
            "EMS + hospital vital missingness (sampled rows)",
            figures_dir / "vital_missingness_heatmap.png", result,
        )

    # Pairwise co-missingness across EMS + hospital vitals.
    if len(vital_cols) >= 2:
        _pairwise_missingness(df, vital_cols, hr_map, result, figures_dir, tables_dir)

    return ems_patterns


def _plot_pattern_bars(
    pattern_df: pd.DataFrame, title: str, path: Path, result: MissingnessAuditResult
) -> None:
    if pattern_df.empty:
        return
    data = pattern_df.sort_values("count", ascending=True)
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(10, max(4.0, 0.4 * len(data))))
        y = np.arange(len(data))
        ax.barh(y, data["percent"], color="#4C72B0", edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(data["pattern"], fontsize=8)
        ax.set_xlabel("Percent of patients (%)")
        ax.set_title(title)
        for yi, (pct, cnt) in enumerate(zip(data["percent"], data["count"])):
            ax.annotate(f"{pct:.1f}% (n={int(cnt):,})", xy=(pct, yi), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, path, result)


def _plot_missingness_heatmap(
    df: pd.DataFrame, cols: List[str], hr_map: Dict[str, str], title: str,
    path: Path, result: MissingnessAuditResult,
) -> None:
    miss = df[cols].isna().astype(int)
    sampled_note = ""
    if len(miss) > HEATMAP_MAX_ROWS:
        miss = miss.sample(n=HEATMAP_MAX_ROWS, random_state=RANDOM_SEED)
        sampled_note = f" (random sample of {HEATMAP_MAX_ROWS:,} rows)"
    # Sort sampled rows by total burden for a readable block structure.
    miss = miss.loc[miss.sum(axis=1).sort_values().index]
    labels = [_label(c, hr_map) for c in cols]
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(cols) + 3), 7))
        im = ax.imshow(miss.to_numpy(), aspect="auto", cmap="Greys",
                       interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel(f"Patients (sorted by burden){sampled_note}")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["present", "missing"])
        fig.tight_layout()
        _save_fig(fig, path, result)


def _pairwise_missingness(
    df: pd.DataFrame, cols: List[str], hr_map: Dict[str, str],
    result: MissingnessAuditResult, figures_dir: Path, tables_dir: Path,
) -> None:
    miss = df[cols].isna().astype(float)
    labels = [_label(c, hr_map) for c in cols]
    corr = miss.corr()  # phi correlation between binary indicators
    corr.index = labels
    corr.columns = labels
    corr_out = corr.reset_index().rename(columns={"index": "variable"})
    _save_table(corr_out, tables_dir / "vital_missingness_pairwise_correlation.csv", result)

    n_total = len(df)
    co_rows: List[Dict] = []
    miss_bool = df[cols].isna()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            both = int((miss_bool.iloc[:, i] & miss_bool.iloc[:, j]).sum())
            co_rows.append(
                {
                    "variable_1": labels[i],
                    "variable_2": labels[j],
                    "n_both_missing": both,
                    "pct_both_missing": (100.0 * both / n_total) if n_total else np.nan,
                    "phi_correlation": float(corr.iloc[i, j]),
                }
            )
    _save_table(pd.DataFrame(co_rows), tables_dir / "vital_missingness_pairwise_comissing.csv", result)

    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(cols) + 2), max(5, 0.7 * len(cols) + 1)))
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Pairwise missingness correlation (phi)")
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "vital_missingness_pairwise_correlation_heatmap.png", result)


# ---------------------------------------------------------------------------
# Clinical-context stratifiers
# ---------------------------------------------------------------------------
def _build_stratifiers(df: pd.DataFrame, ems_n_missing: pd.Series) -> Dict[str, pd.Series]:
    strat: Dict[str, pd.Series] = {}

    if "nfti_positive" in df.columns:
        y = pd.to_numeric(df["nfti_positive"], errors="coerce")
        strat["NFTI status"] = y.map({0: "NFTI negative", 1: "NFTI positive"})

    if "SEX" in df.columns:
        sex = pd.to_numeric(df["SEX"], errors="coerce")
        # Project convention throughout: SEX is 0 = Female, 1 = Male.
        # Code 3 (Non-binary) is reserved for forward-compatibility with raw
        # TQIP data; any other/unexpected code is preserved as "SEX=<code>"
        # rather than being silently mislabeled.
        strat["Sex"] = sex.map(
            {0: "Female", 1: "Male", 3: "Non-binary"}
        ).fillna(sex.apply(lambda v: f"SEX={v:g}" if pd.notna(v) else np.nan))

    if "AGEyears" in df.columns:
        age = pd.to_numeric(df["AGEyears"], errors="coerce")
        strat["Age group"] = pd.cut(
            age, bins=[-np.inf, 18, 40, 65, np.inf], right=False,
            labels=["<18", "18-39", "40-64", "65+"],
        )

    # Race (collapsed from the one-hot race indicators). Multiple flagged races
    # -> "Multiple races"; none flagged -> "Unknown/Not recorded". Used for
    # descriptive missingness stratification only (never as a model predictor).
    race_present = [c for c in RACE_COLUMNS if c in df.columns]
    if race_present:
        rnum = df[race_present].apply(pd.to_numeric, errors="coerce")
        indicator = rnum == 1
        n_flag = indicator.sum(axis=1)
        race = pd.Series("Unknown/Not recorded", index=df.index, dtype=object)
        single = n_flag == 1
        if single.any():
            primary = indicator.idxmax(axis=1).map(RACE_LABELS)
            race.loc[single] = primary.loc[single]
        race.loc[n_flag >= 2] = "Multiple races"
        strat["Race"] = race

    # Ethnicity (1 = Hispanic, 2 = non-Hispanic). Descriptive strata only.
    if "ETHNICITY" in df.columns:
        eth = pd.to_numeric(df["ETHNICITY"], errors="coerce")
        strat["Ethnicity"] = eth.map({1: "Hispanic", 2: "Non-Hispanic"})

    if "MECHANISM" in df.columns:
        mech = df["MECHANISM"]
        strat["Mechanism"] = mech.apply(
            lambda v: f"MECHANISM={v}" if pd.notna(v) and str(v) != "" else np.nan
        )

    if "TRAUMATYPE" in df.columns:
        tt = pd.to_numeric(df["TRAUMATYPE"], errors="coerce")
        # Common NTDB coding: 1=Blunt, 2=Penetrating, 3=Burn, 4=Other.
        strat["Trauma type"] = tt.map(
            {1: "Blunt", 2: "Penetrating", 3: "Burn", 4: "Other"}
        ).fillna(tt.apply(lambda v: f"TRAUMATYPE={v:g}" if pd.notna(v) else np.nan))

    if "TRANSPORTMODE" in df.columns:
        tm = pd.to_numeric(df["TRANSPORTMODE"], errors="coerce")
        strat["Arrival mode"] = tm.map(
            {1: "Ground ambulance", 2: "Helicopter", 3: "Fixed-wing"}
        ).fillna(tm.apply(lambda v: f"TRANSPORTMODE={v:g}" if pd.notna(v) else np.nan))

    if "PREHOSPITALCARDIACARREST" in df.columns:
        pca = pd.to_numeric(df["PREHOSPITALCARDIACARREST"], errors="coerce")
        strat["Prehospital cardiac arrest"] = pca.map({0: "No", 1: "Yes"})

    # Shock index category from EMS HR / EMS SBP (raw vitals).
    if "EMSPULSERATE" in df.columns and "EMSSBP" in df.columns:
        hr = pd.to_numeric(df["EMSPULSERATE"], errors="coerce")
        sbp = pd.to_numeric(df["EMSSBP"], errors="coerce")
        si = hr.where(sbp > 0) / sbp.where(sbp > 0)
        strat["EMS shock index"] = pd.cut(
            si, bins=[-np.inf, 0.7, 1.0, np.inf],
            labels=["<0.7 (normal)", "0.7-1.0 (elevated)", ">=1.0 (high)"],
        )

    # ED hypotension category.
    if "SBP" in df.columns:
        sbp = pd.to_numeric(df["SBP"], errors="coerce")
        strat["ED hypotension (SBP<90)"] = pd.cut(
            sbp, bins=[-np.inf, 90, np.inf], right=False, labels=["SBP<90", "SBP>=90"]
        )

    # GCS category from EMS total GCS.
    if "EMSTOTALGCS" in df.columns:
        gcs = pd.to_numeric(df["EMSTOTALGCS"], errors="coerce")
        strat["EMS GCS category"] = pd.cut(
            gcs, bins=[2, 8, 12, 15], labels=["3-8 (severe)", "9-12 (moderate)", "13-15 (mild)"]
        )

    # EMS missingness burden category (derived).
    if ems_n_missing.notna().any():
        n_ems = len(_present(EMS_VITAL_COLUMNS, df))
        def burden_cat(k):
            if pd.isna(k):
                return np.nan
            k = int(k)
            if k == 0:
                return "Complete EMS vitals"
            if k >= n_ems and n_ems > 0:
                return "All EMS vitals missing"
            if k <= 2:
                return "1-2 EMS vitals missing"
            return ">=3 EMS vitals missing"
        strat["EMS missingness burden"] = ems_n_missing.map(burden_cat)

    return strat


def _ems_targets(df: pd.DataFrame, ems_n_missing: pd.Series) -> Dict[str, pd.Series]:
    """Boolean (or numeric for burden) EMS missingness targets."""
    targets: Dict[str, pd.Series] = {}
    ems_cols = _present(EMS_VITAL_COLUMNS, df)
    for col in ems_cols:
        targets[f"{col}_missing"] = df[col].isna()
    if ems_cols:
        n_ems = len(ems_cols)
        targets["any_ems_vital_missing"] = ems_n_missing > 0
        targets["all_ems_vitals_missing"] = ems_n_missing == n_ems
        targets["complete_ems_vitals"] = ems_n_missing == 0
    return targets


# ---------------------------------------------------------------------------
# Analysis 3: missingness by clinical context
# ---------------------------------------------------------------------------
def compute_missingness_by_context(
    df: pd.DataFrame,
    ems_n_missing: pd.Series,
    result: MissingnessAuditResult,
    figures_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    strat = _build_stratifiers(df, ems_n_missing)
    targets = _ems_targets(df, ems_n_missing)
    if not strat or not targets:
        _warn(result, "No stratifiers or EMS targets available; context analysis skipped.")
        result.skipped_analyses.append("missingness_by_clinical_context")
        return pd.DataFrame()

    outcome = df[OUTCOME_COLUMN] if result.outcome_available else None
    rows: List[Dict] = []
    for var_name, var_series in strat.items():
        values = var_series.dropna().unique()
        for value in values:
            mask = (var_series == value).to_numpy()
            n_records = int(mask.sum())
            small_n = n_records < SMALL_N_THRESHOLD
            nfti_rate = np.nan
            if outcome is not None and n_records:
                grp_y = pd.to_numeric(outcome[mask], errors="coerce")
                nfti_rate = float(grp_y.mean()) if grp_y.notna().any() else np.nan
            # Mean EMS burden in this stratum.
            burden_vals = ems_n_missing[mask]
            for target_name, target_series in targets.items():
                t = target_series[mask]
                if target_name == "complete_ems_vitals":
                    n_meas = int(t.sum())
                    pct = (100.0 * n_meas / n_records) if n_records else np.nan
                    measure = "complete_ems_vitals"
                else:
                    n_meas = int(t.sum())
                    pct = (100.0 * n_meas / n_records) if n_records else np.nan
                    measure = target_name
                rows.append(
                    {
                        "stratum_variable": var_name,
                        "stratum_value": str(value),
                        "n_records": n_records,
                        "target_missingness_measure": measure,
                        "n_missing": n_meas,
                        "pct_missing": pct,
                        "mean_n_ems_vitals_missing": float(burden_vals.mean()) if n_records else np.nan,
                        "nfti_positive_rate": nfti_rate,
                        "small_n": small_n,
                    }
                )
    out = pd.DataFrame(rows)
    _save_table(out, tables_dir / "missingness_by_clinical_context.csv", result)

    # Compact manuscript-friendly version: any-EMS-missing + mean burden.
    compact = out[out["target_missingness_measure"] == "any_ems_vital_missing"].copy()
    compact = compact[
        [
            "stratum_variable", "stratum_value", "n_records",
            "pct_missing", "mean_n_ems_vitals_missing", "nfti_positive_rate", "small_n",
        ]
    ].rename(columns={"pct_missing": "pct_any_ems_vital_missing"})
    _save_table(compact, tables_dir / "ems_missingness_by_clinical_context_compact.csv", result)

    # Capture race / ethnicity dependence of "any EMS vital missing" for the
    # summary report (descriptive; effect size emphasized over p-values).
    for strat_name, key in (("Race", "race"), ("Ethnicity", "ethnicity")):
        sub = out[
            (out["stratum_variable"] == strat_name)
            & (out["target_missingness_measure"] == "any_ems_vital_missing")
        ]
        if not sub.empty:
            pcts = sub["pct_missing"].dropna()
            if len(pcts) >= 2:
                hi = sub.loc[sub["pct_missing"].idxmax()]
                lo = sub.loc[sub["pct_missing"].idxmin()]
                result.headline[f"{key}_spread"] = {
                    "max_value": str(hi["stratum_value"]),
                    "max_pct": float(hi["pct_missing"]),
                    "min_value": str(lo["stratum_value"]),
                    "min_pct": float(lo["pct_missing"]),
                    "spread_pp": float(hi["pct_missing"] - lo["pct_missing"]),
                }

    _context_figures(out, df, ems_n_missing, strat, result, figures_dir)
    return out


def _context_figures(
    context_df: pd.DataFrame,
    df: pd.DataFrame,
    ems_n_missing: pd.Series,
    strat: Dict[str, pd.Series],
    result: MissingnessAuditResult,
    figures_dir: Path,
) -> None:
    ems_cols = _present(EMS_VITAL_COLUMNS, df)

    def _grouped_bar(stratum_var: str, path: Path, title: str) -> None:
        sub = context_df[
            (context_df["stratum_variable"] == stratum_var)
            & (context_df["target_missingness_measure"] == "any_ems_vital_missing")
        ]
        if sub.empty:
            return
        sub = sub.sort_values("stratum_value")
        with plt.rc_context(_PLOT_RC):
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(sub))
            bars = ax.bar(x, sub["pct_missing"], color="#4C72B0", edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["stratum_value"], rotation=30, ha="right")
            ax.set_ylabel("% with any EMS vital missing")
            ax.set_title(title)
            for rect, small in zip(bars, sub["small_n"]):
                if small:
                    ax.annotate("small n", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 2), textcoords="offset points", ha="center",
                                fontsize=7, color="#C44E52")
            fig.tight_layout()
            _save_fig(fig, path, result)

    # EMS vital missingness by NFTI status (per-vital grouped bars).
    if "NFTI status" in strat and ems_cols and result.outcome_available:
        _ems_by_nfti_figure(df, ems_cols, result, figures_dir)
    if "Mechanism" in strat:
        _grouped_bar("Mechanism", figures_dir / "ems_missingness_burden_by_mechanism.png",
                     "EMS vital missingness by mechanism")
    if "Arrival mode" in strat:
        _grouped_bar("Arrival mode", figures_dir / "ems_missingness_burden_by_transfer_status.png",
                     "EMS vital missingness by arrival mode")
    if "Age group" in strat:
        _grouped_bar("Age group", figures_dir / "ems_missingness_burden_by_age_group.png",
                     "EMS vital missingness by age group")
    if "Sex" in strat:
        _grouped_bar("Sex", figures_dir / "ems_missingness_burden_by_sex.png",
                     "EMS vital missingness by sex")
    if "Race" in strat:
        _grouped_bar("Race", figures_dir / "ems_missingness_burden_by_race.png",
                     "EMS vital missingness by race")
    if "Ethnicity" in strat:
        _grouped_bar("Ethnicity", figures_dir / "ems_missingness_burden_by_ethnicity.png",
                     "EMS vital missingness by ethnicity")

    # Heatmap: EMS vital percent missing across strata.
    _context_heatmap(df, ems_cols, strat, result, figures_dir)
    # Dedicated per-vital heatmap for race (answers: is missingness race-dependent?).
    if "Race" in strat and ems_cols:
        _strat_vital_heatmap(
            df, ems_cols, strat["Race"], "race",
            figures_dir / "ems_vital_missingness_by_race_heatmap.png", result,
        )


def _ems_by_nfti_figure(
    df: pd.DataFrame, ems_cols: List[str], result: MissingnessAuditResult, figures_dir: Path
) -> None:
    y = pd.to_numeric(df["nfti_positive"], errors="coerce")
    pos = y == 1
    neg = y == 0
    labels = [EMS_SHORT_LABELS.get(c, c) for c in ems_cols]
    pct_pos = [100.0 * df.loc[pos, c].isna().mean() if pos.any() else np.nan for c in ems_cols]
    pct_neg = [100.0 * df.loc[neg, c].isna().mean() if neg.any() else np.nan for c in ems_cols]
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_manuscript_grid(ax)
        x = np.arange(len(ems_cols))
        width = 0.38
        ax.bar(x - width / 2, pct_neg, width, label="NFTI negative", color="#4C72B0")
        ax.bar(x + width / 2, pct_pos, width, label="NFTI positive", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percent missing (%)")
        ax.set_title("EMS vital missingness by NFTI status")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "ems_vital_missingness_by_nfti_status.png", result)


def _strat_vital_heatmap(
    df: pd.DataFrame, ems_cols: List[str], var: pd.Series, key: str,
    path: Path, result: MissingnessAuditResult,
) -> None:
    """Heatmap of EMS vital % missing across the categories of one stratifier."""
    cats = [c for c in pd.Series(var.dropna().unique())]
    if not cats or not ems_cols:
        return
    matrix = np.full((len(cats), len(ems_cols)), np.nan)
    for i, cat in enumerate(cats):
        mask = (var == cat).to_numpy()
        for j, col in enumerate(ems_cols):
            if mask.sum():
                matrix[i, j] = 100.0 * df.loc[mask, col].isna().mean()
    labels = [EMS_SHORT_LABELS.get(c, c) for c in ems_cols]
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(ems_cols) + 3), max(4, 0.5 * len(cats) + 2)))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(ems_cols)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([str(c) for c in cats], fontsize=8)
        ax.set_title(f"EMS vital % missing by {key}")
        for i in range(len(cats)):
            for j in range(len(ems_cols)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                            fontsize=7, color="white")
        fig.colorbar(im, ax=ax, label="% missing", fraction=0.046, pad=0.04)
        fig.tight_layout()
        _save_fig(fig, path, result)


def _context_heatmap(
    df: pd.DataFrame, ems_cols: List[str], strat: Dict[str, pd.Series],
    result: MissingnessAuditResult, figures_dir: Path,
) -> None:
    if not ems_cols:
        return
    # Use the burden stratifier if present, else mechanism/age.
    key = None
    for candidate in ("EMS missingness burden", "Age group", "Mechanism", "NFTI status"):
        if candidate in strat:
            key = candidate
            break
    if key is None:
        return
    _strat_vital_heatmap(
        df, ems_cols, strat[key], key,
        figures_dir / "ems_vital_missingness_by_context_heatmap.png", result,
    )


# ---------------------------------------------------------------------------
# Analysis I: statistical comparisons
# ---------------------------------------------------------------------------
def _odds_ratio_ci(a: int, b: int, c: int, d: int) -> Tuple[float, float, float]:
    """OR with 95% CI for a 2x2 table [[a,b],[c,d]] (Haldane correction)."""
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a_ * d_) / (b_ * c_)
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    lo = float(np.exp(np.log(or_) - 1.96 * se))
    hi = float(np.exp(np.log(or_) + 1.96 * se))
    return float(or_), lo, hi


def compute_statistical_comparisons(
    df: pd.DataFrame,
    ems_n_missing: pd.Series,
    result: MissingnessAuditResult,
    tables_dir: Path,
) -> pd.DataFrame:
    strat = _build_stratifiers(df, ems_n_missing)
    targets = _ems_targets(df, ems_n_missing)
    # Focus on the most informative binary targets.
    target_names = [t for t in ("any_ems_vital_missing", "all_ems_vitals_missing") if t in targets]
    for extra in ("EMSSBP_missing", "EMSGCS_missing"):
        col = extra.replace("_missing", "")
        if f"{col}_missing" in targets:
            target_names.append(f"{col}_missing")

    rows: List[Dict] = []
    for target_name in target_names:
        target = targets[target_name].astype(bool)
        for var_name, var_series in strat.items():
            if var_name == "EMS missingness burden":
                continue  # derived from the target itself
            valid = var_series.notna()
            cats = list(pd.Series(var_series[valid].unique()))
            if len(cats) < 2:
                continue

            # Build contingency table (categories x [missing, not missing]).
            table = []
            for cat in cats:
                m = (var_series == cat).to_numpy() & valid.to_numpy()
                n_miss = int(target[m].sum())
                n_present = int((~target[m]).sum())
                table.append([n_miss, n_present])
            table_arr = np.array(table)

            p_value = np.nan
            if _SCIPY_AVAILABLE and table_arr.shape[0] >= 2 and table_arr.sum() > 0:
                try:
                    if table_arr.shape == (2, 2) and table_arr.min() < 5:
                        _, p_value = fisher_exact(table_arr)
                    else:
                        _, p_value, _, _ = chi2_contingency(table_arr)
                except Exception:
                    p_value = np.nan

            if len(cats) == 2:
                (a, b), (c, d) = table_arr  # a=miss in g1, b=present g1, ...
                n1, n2 = a + b, c + d
                pct1 = 100.0 * a / n1 if n1 else np.nan
                pct2 = 100.0 * c / n2 if n2 else np.nan
                or_, lo, hi = _odds_ratio_ci(int(a), int(b), int(c), int(d))
                rows.append(
                    {
                        "missingness_target": target_name,
                        "stratum_variable": var_name,
                        "comparison": f"{cats[0]} vs {cats[1]}",
                        "n_group_1": int(n1),
                        "pct_missing_group_1": pct1,
                        "n_group_2": int(n2),
                        "pct_missing_group_2": pct2,
                        "absolute_percentage_point_difference": abs(pct1 - pct2),
                        "odds_ratio": or_,
                        "ci_lower": lo,
                        "ci_upper": hi,
                        "p_value": p_value,
                        "note": "Fisher" if (_SCIPY_AVAILABLE and table_arr.min() < 5) else "chi-square",
                    }
                )
            else:
                # Multi-category: report overall chi-square + spread of rates.
                pcts = [100.0 * r[0] / (r[0] + r[1]) if (r[0] + r[1]) else np.nan for r in table_arr]
                rows.append(
                    {
                        "missingness_target": target_name,
                        "stratum_variable": var_name,
                        "comparison": f"chi-square across {len(cats)} categories",
                        "n_group_1": int(table_arr.sum()),
                        "pct_missing_group_1": float(np.nanmin(pcts)),
                        "n_group_2": np.nan,
                        "pct_missing_group_2": float(np.nanmax(pcts)),
                        "absolute_percentage_point_difference": float(np.nanmax(pcts) - np.nanmin(pcts)),
                        "odds_ratio": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": p_value,
                        "note": "min/max rate across categories; effect size > p-value (large n)",
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        _warn(result, "No statistical comparisons could be computed.")
    _save_table(out, tables_dir / "missingness_statistical_comparisons.csv", result)
    return out


# ---------------------------------------------------------------------------
# Analysis 4: missingness as a prediction target
# ---------------------------------------------------------------------------
def fit_missingness_prediction_models(
    df: pd.DataFrame,
    ems_n_missing: pd.Series,
    hr_map: Dict[str, str],
    result: MissingnessAuditResult,
    figures_dir: Path,
    tables_dir: Path,
    metrics_dir: Path,
) -> pd.DataFrame:
    targets = _ems_targets(df, ems_n_missing)
    target_names = [
        t for t in (
            [f"{c}_missing" for c in EMS_VITAL_COLUMNS]
            + ["any_ems_vital_missing", "all_ems_vitals_missing"]
        )
        if t in targets
    ]
    if not target_names:
        _warn(result, "No EMS targets; missingness-prediction models skipped.")
        result.skipped_analyses.append("missingness_prediction")
        return pd.DataFrame()

    cont_cols = _present(PREDICTOR_CONTINUOUS, df)
    cat_cols = _present(PREDICTOR_CATEGORICAL, df)
    predictor_cols = cont_cols + cat_cols
    if not predictor_cols:
        _warn(result, "No baseline predictors available; missingness-prediction models skipped.")
        result.skipped_analyses.append("missingness_prediction")
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_SEED)
    X_all = df[predictor_cols].copy()
    for c in cont_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
    for c in cat_cols:
        X_all[c] = X_all[c].astype("object")

    summary_rows: List[Dict] = []
    coef_rows: List[Dict] = []

    for target_name in target_names:
        y = targets[target_name].astype(int).to_numpy()
        n_total = len(y)
        prevalence = float(y.mean()) if n_total else np.nan
        if len(np.unique(y)) < 2:
            summary_rows.append(
                {
                    "target": target_name, "n_total": n_total, "prevalence_missing": prevalence,
                    "train_n": np.nan, "test_n": np.nan, "auroc": np.nan, "auprc": np.nan,
                    "note": "only one class present; model not fit",
                }
            )
            continue

        # Optional seeded subsample for tractable fitting on huge data.
        idx = np.arange(n_total)
        sampled_note = ""
        if n_total > MODEL_FIT_MAX_N:
            idx = rng.choice(n_total, size=MODEL_FIT_MAX_N, replace=False)
            sampled_note = f"fit on seeded subsample of {MODEL_FIT_MAX_N:,} (prevalence preserved approximately)"
        X = X_all.iloc[idx]
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, yy, test_size=0.25, random_state=RANDOM_SEED, stratify=yy
        )

        pre = ColumnTransformer(
            transformers=[
                ("cont", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), cont_cols),
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                ]), cat_cols),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs")
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        try:
            pipe.fit(X_train, y_train)
            prob = pipe.predict_proba(X_test)[:, 1]
        except Exception as exc:  # pragma: no cover - defensive
            _warn(result, f"Prediction model for {target_name} failed: {exc}")
            continue

        metrics = evaluate_binary_classifier(y_test, prob)
        summary_rows.append(
            {
                "target": target_name,
                "n_total": n_total,
                "prevalence_missing": prevalence,
                "train_n": int(len(y_train)),
                "test_n": int(len(y_test)),
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "note": sampled_note,
            }
        )

        # Extract coefficients with readable feature names.
        try:
            feat_names = pipe.named_steps["pre"].get_feature_names_out()
        except Exception:
            feat_names = np.array([f"f{i}" for i in range(pipe.named_steps["clf"].coef_.shape[1])])
        coefs = pipe.named_steps["clf"].coef_.ravel()
        order = np.argsort(np.abs(coefs))[::-1]
        for rank, i in enumerate(order, start=1):
            coef_rows.append(
                {
                    "target": target_name,
                    "predictor": _prettify_feature(str(feat_names[i]), hr_map),
                    "coefficient": float(coefs[i]),
                    "odds_ratio": float(np.exp(coefs[i])),
                    "abs_coefficient_rank": rank,
                    "direction": "increases missingness" if coefs[i] > 0 else "decreases missingness",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    _save_metric(summary_df, metrics_dir / "missingness_prediction_model_summary.csv", result)
    coef_df = pd.DataFrame(coef_rows)
    _save_table(coef_df, tables_dir / "missingness_prediction_coefficients.csv", result)

    _plot_prediction_performance(summary_df, result, figures_dir)
    if not coef_df.empty:
        if "any_ems_vital_missing" in coef_df["target"].values:
            _plot_top_predictors(
                coef_df, "any_ems_vital_missing",
                "Top predictors of ANY EMS vital missing",
                figures_dir / "missingness_prediction_top_predictors_any_ems_missing.png", result,
            )
        if "EMSSBP_missing" in coef_df["target"].values:
            _plot_top_predictors(
                coef_df, "EMSSBP_missing",
                "Top predictors of EMS SBP missing",
                figures_dir / "missingness_prediction_top_predictors_emssbp_missing.png", result,
            )
    return summary_df


def _prettify_feature(name: str, hr_map: Dict[str, str]) -> str:
    # ColumnTransformer prefixes: "cont__AGEyears", "cat__SEX_2.0".
    for prefix in ("cont__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for header in sorted(hr_map.keys(), key=len, reverse=True):
        if name == header:
            return _label(header, hr_map)
        if name.startswith(header + "_"):
            level = name[len(header) + 1:]
            return f"{_label(header, hr_map)}={level}"
    return name


def _plot_prediction_performance(
    summary_df: pd.DataFrame, result: MissingnessAuditResult, figures_dir: Path
) -> None:
    data = summary_df.dropna(subset=["auroc"])
    if data.empty:
        return
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(data))
        width = 0.38
        ax.bar(x - width / 2, data["auroc"], width, label="AUROC", color="#4C72B0")
        ax.bar(x + width / 2, data["auprc"], width, label="AUPRC", color="#DD8452")
        ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0, label="AUROC = 0.5 (random)")
        ax.set_xticks(x)
        ax.set_xticklabels(data["target"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Predicting EMS vital missingness (test set)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "missingness_prediction_performance.png", result)


def _plot_top_predictors(
    coef_df: pd.DataFrame, target: str, title: str, path: Path,
    result: MissingnessAuditResult, top_n: int = 15,
) -> None:
    sub = coef_df[coef_df["target"] == target].nsmallest(top_n, "abs_coefficient_rank")
    if sub.empty:
        return
    sub = sub.sort_values("coefficient")
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(sub))))
        colors = ["#C44E52" if c > 0 else "#4C72B0" for c in sub["coefficient"]]
        ax.barh(np.arange(len(sub)), sub["coefficient"], color=colors, edgecolor="white")
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["predictor"], fontsize=8)
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_xlabel("Logistic regression coefficient (log-odds)")
        ax.set_title(title)
        fig.tight_layout()
        _save_fig(fig, path, result)


# ---------------------------------------------------------------------------
# Analysis 5: model performance stratified by EMS missingness burden
# ---------------------------------------------------------------------------
def _burden_group(k: float, n_ems: int) -> Optional[str]:
    if pd.isna(k):
        return None
    k = int(k)
    if k == 0:
        return "Complete EMS vitals"
    if n_ems > 0 and k >= n_ems:
        return "All EMS vitals missing"
    if k <= 2:
        return "Low (1-2 missing)"
    return "High (>=3 missing)"


def _metrics_for_group(y_true, y_prob, threshold: float, min_n: int = 50) -> Dict:
    n = len(y_true)
    small_n = n < min_n
    if n == 0 or len(np.unique(y_true)) < 2:
        base = evaluate_binary_classifier(y_true, y_prob, threshold=threshold) if n else {}
        return {
            "n": n,
            "nfti_prevalence": float(np.mean(y_true)) if n else np.nan,
            "auroc": np.nan, "auprc": np.nan, "brier": base.get("brier", np.nan) if n else np.nan,
            "threshold": threshold,
            "sensitivity": base.get("sensitivity", np.nan) if n else np.nan,
            "specificity": base.get("specificity", np.nan) if n else np.nan,
            "ppv": base.get("precision_ppv", np.nan) if n else np.nan,
            "npv": base.get("npv", np.nan) if n else np.nan,
            "f1": base.get("f1", np.nan) if n else np.nan,
            "accuracy": base.get("accuracy", np.nan) if n else np.nan,
            "tp": base.get("tp", np.nan) if n else np.nan, "tn": base.get("tn", np.nan) if n else np.nan,
            "fp": base.get("fp", np.nan) if n else np.nan, "fn": base.get("fn", np.nan) if n else np.nan,
            "small_n": True,
        }
    m = evaluate_binary_classifier(y_true, y_prob, threshold=threshold)
    return {
        "n": n,
        "nfti_prevalence": m["prevalence"],
        "auroc": m["auroc"], "auprc": m["auprc"], "brier": m["brier"],
        "threshold": threshold,
        "sensitivity": m["sensitivity"], "specificity": m["specificity"],
        "ppv": m["precision_ppv"], "npv": m["npv"], "f1": m["f1"], "accuracy": m["accuracy"],
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "small_n": small_n,
    }


def compute_model_performance_by_missingness_burden(
    df: pd.DataFrame,
    ems_n_missing: pd.Series,
    result: MissingnessAuditResult,
    figures_dir: Path,
    metrics_dir: Path,
) -> Optional[pd.DataFrame]:
    pred_path = NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH
    if not pred_path.exists():
        msg = (
            "Holdout predictions not found at "
            f"{pred_path}. Analysis 5 (model performance by missingness burden) skipped."
        )
        _warn(result, msg)
        result.skipped_analyses.append("model_performance_by_missingness_burden")
        (metrics_dir / "model_performance_by_missingness_WARNING.txt").write_text(msg + "\n", encoding="utf-8")
        return None

    preds = pd.read_csv(pred_path)
    burden = pd.DataFrame({"record_id": df["record_id"].values, "ems_n_missing": ems_n_missing.values})
    merged = preds.merge(burden, on="record_id", how="inner")
    if merged.empty:
        msg = (
            "No overlap between holdout prediction record IDs and the loaded "
            "cohort's record IDs (likely a subsampled or mismatched input). "
            "Analysis 5 skipped. Run on the full raw cohort CSV used for training."
        )
        _warn(result, msg)
        result.skipped_analyses.append("model_performance_by_missingness_burden")
        (metrics_dir / "model_performance_by_missingness_WARNING.txt").write_text(msg + "\n", encoding="utf-8")
        return None

    n_ems = len(_present(EMS_VITAL_COLUMNS, df))
    merged["burden_group"] = merged["ems_n_missing"].apply(lambda k: _burden_group(k, n_ems))
    merged = merged.dropna(subset=["burden_group", "y_true", "y_pred_prob"])

    # Secondary (validation-locked) threshold, if available.
    secondary_threshold = None
    if NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH.exists():
        try:
            sel = pd.read_csv(NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH)
            if "selected_threshold" in sel.columns and len(sel):
                secondary_threshold = float(sel["selected_threshold"].iloc[0])
        except Exception:
            secondary_threshold = None

    group_order = ["Complete EMS vitals", "Low (1-2 missing)", "High (>=3 missing)", "All EMS vitals missing"]
    rows: List[Dict] = []
    for group in group_order:
        g = merged[merged["burden_group"] == group]
        if g.empty:
            continue
        row = {"burden_group": group, "threshold_type": "primary_0.5"}
        row.update(_metrics_for_group(g["y_true"].to_numpy(), g["y_pred_prob"].to_numpy(), PRIMARY_THRESHOLD))
        rows.append(row)
        if secondary_threshold is not None:
            row2 = {"burden_group": group, "threshold_type": f"validation_selected_{secondary_threshold:.2f}"}
            row2.update(_metrics_for_group(g["y_true"].to_numpy(), g["y_pred_prob"].to_numpy(), secondary_threshold))
            rows.append(row2)

    perf = pd.DataFrame(rows)
    _save_metric(perf, metrics_dir / "model_performance_by_ems_missingness_burden.csv", result)

    # Compact comparison: complete vs any-missing, complete vs all-missing.
    merged["any_missing"] = merged["ems_n_missing"] > 0
    merged["all_missing"] = (merged["ems_n_missing"] >= n_ems) & (n_ems > 0)
    comp_rows: List[Dict] = []
    complete = merged[merged["ems_n_missing"] == 0]
    any_miss = merged[merged["any_missing"]]
    all_miss = merged[merged["all_missing"]]
    for label, grp in [
        ("complete_ems_vitals", complete),
        ("any_ems_vital_missing", any_miss),
        ("all_ems_vitals_missing", all_miss),
    ]:
        if grp.empty:
            continue
        r = {"comparison_group": label, "threshold_type": "primary_0.5"}
        r.update(_metrics_for_group(grp["y_true"].to_numpy(), grp["y_pred_prob"].to_numpy(), PRIMARY_THRESHOLD))
        comp_rows.append(r)
    _save_metric(
        pd.DataFrame(comp_rows),
        metrics_dir / "model_performance_complete_vs_missing_ems_vitals.csv",
        result,
    )

    _plot_performance_by_burden(perf, result, figures_dir)
    return perf


def _plot_performance_by_burden(
    perf: pd.DataFrame, result: MissingnessAuditResult, figures_dir: Path
) -> None:
    data = perf[perf["threshold_type"] == "primary_0.5"].copy()
    if data.empty:
        return
    order = ["Complete EMS vitals", "Low (1-2 missing)", "High (>=3 missing)", "All EMS vitals missing"]
    data["__order"] = data["burden_group"].apply(lambda g: order.index(g) if g in order else 99)
    data = data.sort_values("__order")

    # Discrimination: AUROC / AUPRC / Brier.
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(data))
        width = 0.27
        ax.bar(x - width, data["auroc"], width, label="AUROC", color="#4C72B0")
        ax.bar(x, data["auprc"], width, label="AUPRC", color="#DD8452")
        ax.bar(x + width, data["brier"], width, label="Brier", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(data["burden_group"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title("Model discrimination by EMS missingness burden")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "model_discrimination_by_ems_missingness_burden.png", result)

    # Threshold metrics.
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(data))
        width = 0.2
        for i, (metric, color) in enumerate(
            [("sensitivity", "#4C72B0"), ("specificity", "#DD8452"),
             ("ppv", "#55A868"), ("npv", "#C44E52")]
        ):
            ax.bar(x + (i - 1.5) * width, data[metric], width, label=metric.upper(), color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(data["burden_group"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Metric value (threshold 0.5)")
        ax.set_ylim(0, 1)
        ax.set_title("Sensitivity / specificity / PPV / NPV by EMS missingness burden")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "model_threshold_metrics_by_ems_missingness_burden.png", result)

    # NFTI prevalence.
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(data))
        bars = ax.bar(x, 100.0 * data["nfti_prevalence"], color="#4C72B0", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(data["burden_group"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("NFTI prevalence (%)")
        ax.set_title("NFTI prevalence by EMS missingness burden")
        for rect, val, small in zip(bars, data["nfti_prevalence"], data["small_n"]):
            txt = f"{100 * val:.1f}%" + ("\nsmall n" if small else "")
            ax.annotate(txt, xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
        fig.tight_layout()
        _save_fig(fig, figures_dir / "nfti_prevalence_by_ems_missingness_burden.png", result)


# ---------------------------------------------------------------------------
# Combined manuscript figure
# ---------------------------------------------------------------------------
def build_combined_figure(
    df: pd.DataFrame,
    ems_patterns: Optional[pd.DataFrame],
    perf: Optional[pd.DataFrame],
    result: MissingnessAuditResult,
    figures_dir: Path,
) -> None:
    ems_cols = _present(EMS_VITAL_COLUMNS, df)
    if not ems_cols:
        return
    try:
        with plt.rc_context(_PLOT_RC):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # A. Percent missing by EMS vital.
            ax = axes[0, 0]
            labels = [EMS_SHORT_LABELS.get(c, c) for c in ems_cols]
            pct = [100.0 * df[c].isna().mean() for c in ems_cols]
            ax.bar(np.arange(len(ems_cols)), pct, color="#4C72B0", edgecolor="white")
            ax.set_xticks(np.arange(len(ems_cols)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("% missing")
            ax.set_title("A. Percent missing by EMS vital", loc="left")

            # B. Top EMS missingness patterns.
            ax = axes[0, 1]
            if ems_patterns is not None and not ems_patterns.empty:
                top = ems_patterns.head(8).sort_values("percent")
                ax.barh(np.arange(len(top)), top["percent"], color="#DD8452", edgecolor="white")
                ax.set_yticks(np.arange(len(top)))
                ax.set_yticklabels(top["pattern"], fontsize=7)
                ax.set_xlabel("% of patients")
                ax.set_title("B. Top EMS missingness patterns", loc="left")
            else:
                ax.axis("off")

            # C. EMS missingness by NFTI status.
            ax = axes[1, 0]
            if result.outcome_available and "nfti_positive" in df.columns:
                y = pd.to_numeric(df["nfti_positive"], errors="coerce")
                pos, neg = y == 1, y == 0
                pct_pos = [100.0 * df.loc[pos, c].isna().mean() if pos.any() else np.nan for c in ems_cols]
                pct_neg = [100.0 * df.loc[neg, c].isna().mean() if neg.any() else np.nan for c in ems_cols]
                x = np.arange(len(ems_cols))
                ax.bar(x - 0.19, pct_neg, 0.38, label="NFTI-", color="#4C72B0")
                ax.bar(x + 0.19, pct_pos, 0.38, label="NFTI+", color="#C44E52")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel("% missing")
                ax.set_title("C. EMS missingness by NFTI status", loc="left")
                ax.legend(fontsize=8)
            else:
                ax.axis("off")

            # D. Model AUROC / AUPRC by burden.
            ax = axes[1, 1]
            if perf is not None and not perf.empty:
                data = perf[perf["threshold_type"] == "primary_0.5"]
                order = ["Complete EMS vitals", "Low (1-2 missing)", "High (>=3 missing)", "All EMS vitals missing"]
                data = data.set_index("burden_group").reindex([g for g in order if g in set(data["burden_group"])]).reset_index()
                x = np.arange(len(data))
                ax.bar(x - 0.2, data["auroc"], 0.4, label="AUROC", color="#4C72B0")
                ax.bar(x + 0.2, data["auprc"], 0.4, label="AUPRC", color="#DD8452")
                ax.set_xticks(x)
                ax.set_xticklabels(data["burden_group"], rotation=20, ha="right", fontsize=7)
                ax.set_ylim(0, 1)
                ax.set_title("D. Model AUROC/AUPRC by EMS burden", loc="left")
                ax.legend(fontsize=8)
            else:
                ax.axis("off")

            fig.suptitle("Missing-data audit (EMS prehospital vitals)", fontsize=16, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            _save_fig(fig, figures_dir / "missingness_audit_combined.png", result)
    except Exception as exc:  # pragma: no cover - combined figure is optional
        _warn(result, f"Combined figure skipped: {exc}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def write_summary_report(
    result: MissingnessAuditResult,
    resolved: Dict[str, List[str]],
    variable_df: pd.DataFrame,
    ems_patterns: Optional[pd.DataFrame],
    prediction_summary: pd.DataFrame,
    perf: Optional[pd.DataFrame],
    metrics_dir: Path,
) -> Path:
    lines: List[str] = []
    lines.append("# Missing-Data Audit Summary\n")
    lines.append(f"- Dataset: `{result.source_path}` (source type: {result.data_source})")
    lines.append(f"- Records analyzed: {result.n_records:,}")
    if result.processed_data_warning:
        lines.append(
            "- **WARNING: processed/imputed data detected.** Missingness may be "
            "understated; rerun on raw pre-imputation data for accurate counts."
        )
    else:
        lines.append("- Data are raw / pre-imputation (missingness computed before imputation/scaling/one-hot).")
    lines.append(f"- Outcome (`{OUTCOME_COLUMN}`) available: {result.outcome_available}")
    lines.append("")

    lines.append(f"- EMS vitals found: {', '.join(result.ems_vitals_found) or '(none)'}")
    lines.append(f"- Hospital vitals found: {', '.join(result.hospital_vitals_found) or '(none)'}")
    lines.append("")

    lines.append("## Feature groups analyzed")
    for group, cols in resolved.items():
        lines.append(f"- {group}: {len(cols)} variable(s) resolved")
    lines.append("")

    if not variable_df.empty:
        top5 = variable_df.sort_values("pct_missing", ascending=False).head(5)
        lines.append("## Top 5 most-missing variables (selected groups)")
        for _, r in top5.iterrows():
            lines.append(f"- {r['variable_label']} ({r['variable']}): {r['pct_missing']:.1f}% missing")
        lines.append("")

    h = result.headline
    if "pct_any_ems_missing" in h:
        lines.append("## EMS vital missingness")
        lines.append(f"- % with ANY EMS vital missing: {h['pct_any_ems_missing']:.1f}%")
        lines.append(f"- % with ALL EMS vitals missing: {h['pct_all_ems_missing']:.1f}%")
        if ems_patterns is not None and not ems_patterns.empty:
            top = ems_patterns.iloc[0]
            lines.append(f"- Most common EMS pattern: \"{top['pattern']}\" ({top['percent']:.1f}% of patients)")
        lines.append("")

    if "race_spread" in h or "ethnicity_spread" in h:
        lines.append("## Is missingness dependent on race / ethnicity?")
        if "race_spread" in h:
            r = h["race_spread"]
            lines.append(
                f"- Any EMS vital missing ranges from {r['min_pct']:.1f}% "
                f"({r['min_value']}) to {r['max_pct']:.1f}% ({r['max_value']}) "
                f"across race groups (spread {r['spread_pp']:.1f} pp)."
            )
        if "ethnicity_spread" in h:
            e = h["ethnicity_spread"]
            lines.append(
                f"- Any EMS vital missing: {e['max_pct']:.1f}% ({e['max_value']}) "
                f"vs {e['min_pct']:.1f}% ({e['min_value']}) by ethnicity "
                f"(spread {e['spread_pp']:.1f} pp)."
            )
        lines.append(
            "- See `missingness_by_clinical_context.csv` (Race / Ethnicity strata) "
            "and `missingness_statistical_comparisons.csv` for full detail. "
            "Race and ethnicity are analyzed descriptively only and are NEVER used "
            "as predictors in any model."
        )
        lines.append("")

    if not prediction_summary.empty:
        lines.append("## Is missingness systematic / predictable?")
        fitted = prediction_summary.dropna(subset=["auroc"])
        if not fitted.empty:
            max_auroc = float(fitted["auroc"].max())
            best = fitted.loc[fitted["auroc"].idxmax(), "target"]
            if max_auroc >= 0.6:
                verdict = (
                    f"Yes — missingness is likely SYSTEMATIC/predictable (best AUROC "
                    f"{max_auroc:.3f} for {best}, well above 0.5). Missingness is not "
                    "completely random. (No causal claims are made.)"
                )
            elif max_auroc > 0.55:
                verdict = (
                    f"Partially — modest predictability (best AUROC {max_auroc:.3f}); "
                    "missingness is somewhat systematic."
                )
            else:
                verdict = (
                    f"Largely no — missingness is only weakly predictable from baseline "
                    f"covariates (best AUROC {max_auroc:.3f}, near 0.5)."
                )
            lines.append(f"- {verdict}")
        lines.append(
            "- Note: race and ethnicity are intentionally excluded from these "
            "missingness-prediction models to avoid encoding racial/ethnic bias."
        )
        lines.append("")

    if perf is not None and not perf.empty:
        lines.append("## Does model performance change with EMS missingness burden?")
        data = perf[perf["threshold_type"] == "primary_0.5"].dropna(subset=["auroc"])
        if len(data) >= 2:
            spread = float(data["auroc"].max() - data["auroc"].min())
            for _, r in data.iterrows():
                lines.append(
                    f"  - {r['burden_group']}: n={int(r['n']):,}, "
                    f"AUROC={r['auroc']:.3f}, AUPRC={r['auprc']:.3f}, "
                    f"prevalence={100 * r['nfti_prevalence']:.1f}%"
                    + ("  [small n]" if r["small_n"] else "")
                )
            lines.append(
                f"- AUROC spread across burden groups: {spread:.3f} "
                + ("(notable — performance differs with missingness)" if spread >= 0.03
                   else "(small — performance is relatively stable)")
            )
        lines.append("")
    elif "model_performance_by_missingness_burden" in result.skipped_analyses:
        lines.append("## Model performance by EMS missingness burden")
        lines.append("- Skipped (holdout predictions unavailable or record IDs did not align). See WARNING file.")
        lines.append("")

    if result.missing_expected_columns:
        lines.append("## Missing expected columns (warned + skipped)")
        lines.append(f"- {', '.join(sorted(set(result.missing_expected_columns)))}")
        lines.append("")

    if result.skipped_analyses:
        lines.append("## Skipped analyses")
        for s in result.skipped_analyses:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## Generated tables")
    for p in result.tables_saved:
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("## Generated metrics")
    for p in result.metrics_saved:
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("## Generated figures")
    for p in result.figures_saved:
        lines.append(f"- {p.name}")
    lines.append("")

    if result.warnings:
        lines.append("## Warnings / notes")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    path = metrics_dir / "missingness_audit_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    result.metrics_saved.append(path)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_missingness_audit(
    input_path: Optional[Path] = None,
    *,
    figures_dir: Path = MISSINGNESS_FIGURES_DIR,
    tables_dir: Path = MISSINGNESS_TABLES_DIR,
    metrics_dir: Path = MISSINGNESS_METRICS_DIR,
    max_records: Optional[int] = None,
) -> MissingnessAuditResult:
    """Run the full missing-data audit and write all tables, metrics, figures."""
    ensure_dirs()
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    metrics_dir = Path(metrics_dir)
    for d in (figures_dir, tables_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path, max_records=max_records)
    df = data.df
    hr_map = load_human_readable_map()

    result = MissingnessAuditResult(
        data_source=data.source,
        source_path=data.source_path,
        n_records=len(df),
        processed_data_warning=data.processed_warning,
        outcome_available=data.outcome_available,
    )
    result.warnings.extend(data.notes)
    result.ems_vitals_found = _present(EMS_VITAL_COLUMNS, df)
    result.hospital_vitals_found = _present(HOSPITAL_VITAL_COLUMNS, df)

    if df.empty:
        _warn(result, "Loaded dataset is empty; nothing to audit.")
        write_summary_report(result, {}, pd.DataFrame(), None, pd.DataFrame(), None, metrics_dir)
        _print_summary(result)
        return result

    # B. Resolve feature groups.
    resolved = resolve_feature_groups(data.available_columns, result, tables_dir)

    # C. Analysis 1: variable + group missingness.
    variable_df = compute_variable_missingness(df, resolved, hr_map, result, tables_dir)
    group_df = compute_group_missingness(df, resolved, result, tables_dir)
    ems_n_missing, ems_dist = compute_ems_burden(df, result, tables_dir)
    plot_variable_missingness(variable_df, result, figures_dir)
    plot_group_missingness(group_df, result, figures_dir)
    plot_ems_burden(ems_dist, result, figures_dir)

    ems_cols = _present(EMS_VITAL_COLUMNS, df)
    if ems_cols:
        any_missing = float((ems_n_missing > 0).mean() * 100.0)
        all_missing = float((ems_n_missing == len(ems_cols)).mean() * 100.0)
        result.headline["pct_any_ems_missing"] = any_missing
        result.headline["pct_all_ems_missing"] = all_missing

    # D. Analysis 2: patterns + pairwise.
    ems_patterns = compute_missingness_patterns(df, hr_map, result, figures_dir, tables_dir)

    # E. Analysis 3: missingness by clinical context.
    compute_missingness_by_context(df, ems_n_missing, result, figures_dir, tables_dir)

    # I. Statistical comparisons.
    compute_statistical_comparisons(df, ems_n_missing, result, tables_dir)

    # F. Analysis 4: missingness as a prediction target.
    prediction_summary = fit_missingness_prediction_models(
        df, ems_n_missing, hr_map, result, figures_dir, tables_dir, metrics_dir
    )

    # G. Analysis 5: model performance by missingness burden.
    perf = compute_model_performance_by_missingness_burden(
        df, ems_n_missing, result, figures_dir, metrics_dir
    )

    # H. Combined figure + summary report.
    build_combined_figure(df, ems_patterns, perf, result, figures_dir)
    write_summary_report(
        result, resolved, variable_df, ems_patterns, prediction_summary, perf, metrics_dir
    )

    _print_summary(result)
    return result


def _print_summary(result: MissingnessAuditResult) -> None:
    print("\n=== Missing-Data Audit ===")
    print(f"Data source:          {result.data_source} ({result.source_path})")
    print(f"Records analyzed:     {result.n_records:,}")
    if result.processed_data_warning:
        print("WARNING: processed/imputed data detected; missingness may be understated.")
    print(f"EMS vitals found:     {', '.join(result.ems_vitals_found) or '(none)'}")
    print(f"Hospital vitals found:{' ' + ', '.join(result.hospital_vitals_found) if result.hospital_vitals_found else ' (none)'}")
    if "pct_any_ems_missing" in result.headline:
        print(f"Any EMS vital missing: {result.headline['pct_any_ems_missing']:.1f}%")
        print(f"All EMS vitals missing:{result.headline['pct_all_ems_missing']:.1f}%")
    print(f"Tables saved:         {len(result.tables_saved)}")
    print(f"Metrics saved:        {len(result.metrics_saved)}")
    print(f"Figures saved:        {len(result.figures_saved)}")
    if result.missing_expected_columns:
        print(f"Expected-but-absent:  {', '.join(sorted(set(result.missing_expected_columns)))}")
    if result.skipped_analyses:
        print(f"Skipped analyses:     {', '.join(result.skipped_analyses)}")
    print("Missing-data audit complete.\n")
