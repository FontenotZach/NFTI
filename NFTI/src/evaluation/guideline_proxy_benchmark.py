"""Exploratory 2021 Field Triage Guideline "available-variable proxy" benchmark.

This module implements a compact, manuscript/supplement-ready benchmark that
approximates the 2021 National Guideline for the Field Triage of Injured
Patients using ONLY available prehospital TQIP variables, and compares that
proxy against the primary XGBoost NFTI model on the exact same holdout cohort.

IMPORTANT SCOPE / CONSTRAINTS (intentional and enforced):

* This is an *available-variable proxy*, NOT a full reconstruction of the 2021
  guideline. Several criteria cannot be represented in TQIP and are explicitly
  marked ``unmapped`` rather than silently ignored.
* The guideline was not designed to predict NFTI; this is an EXPLORATORY
  benchmark only.
* ``VPOEMSJUDGE`` (opaque EMS-provider-judgment flag) is intentionally NOT used.
* No hospital-arrival / post-arrival variables are used to define proxy
  positivity. Every source variable is a prehospital (scene, Timing=1) value.
* The benchmark outcome remains NFTI-positive status, evaluated on the SAME
  filtered prehospital EMS cohort and the SAME holdout split used for primary
  model evaluation. Holdout alignment is achieved by reloading the raw cohort
  CSV, re-applying the prehospital EMS cohort filter, and inner-joining the
  saved holdout predictions on ``record_id`` (mirroring the missingness audit).

This module is intentionally separate from the model training pipeline: it
never retrains the model, never mutates ``record.data``, and only writes under
``artifacts/{tables,figures}/guideline_proxy`` and one report under
``artifacts/reports``.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: write figures without requiring a display
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.metrics import roc_auc_score

from src.evaluation.binary_metrics import calculate_binary_classification_metrics
from src.plotting import NAVY, RED, apply_manuscript_grid
from src.paths import (
    GUIDELINE_PROXY_BENCHMARK_SUMMARY_PATH,
    GUIDELINE_PROXY_CRITERION_MAPPING_PATH,
    GUIDELINE_PROXY_FIGURES_DIR,
    GUIDELINE_PROXY_RULE_METRICS_PATH,
    GUIDELINE_PROXY_TABLES_DIR,
    GUIDELINE_PROXY_TIER_FIGURE_PATH,
    GUIDELINE_PROXY_TIER_TABLE_PATH,
    GUIDELINE_PROXY_VS_MODEL_METRICS_PATH,
    NFTI_POSITIVE_LR_HOLDOUT_PREDICTIONS_PATH,
    NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH,
    NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
    ensure_dirs,
)

OUTCOME_COLUMN = "nfti_positive"

# ---------------------------------------------------------------------------
# Source columns.
#
# Every column below is a prehospital (scene, Timing=1) value per
# data/schemas/header_definitions.csv. We assert this allowlist contains no
# post-arrival columns and never references VPOEMSJUDGE (see _quality_checks).
# ---------------------------------------------------------------------------
# EMS-documented anatomic / physiologic field-triage criteria flags (0/1).
TCC_FLAG_COLUMNS: List[str] = [
    "TCCGCSLE13",
    "TCCSBPLT30",
    "TCC10RR29",
    "TCCPEN",
    "TCCCHEST",
    "TCCLONGBONE",
    "TCCCRUSHED",
    "TCCAMPUTATION",
    "TCCPELVIC",
    "TCCSKULLFRACTURE",
    "TCCPARALYSIS",
]
# EMS-documented mechanism / special-consideration field-triage criteria flags.
VPO_FLAG_COLUMNS: List[str] = [
    "VPOFALLADULT",
    "VPOFALLCHILD",
    "VPOCRASHINTRUSION",
    "VPOCRASHEJECT",
    "VPOCRASHDEATH",
    "VPOCRASHTELEMETRY",
    "VPOAUTOPEDIMPACT",
    "VPOMOTORCYCLECRASH",
    "VPO65SBP110",
    "VPOANTICOAGULANT",
    "VPOPREGNANCY20WKS",
    "VPOBURNS",
    "VPOTRAUMABURNS",
]
# Raw prehospital continuous vitals + demographics used for numeric thresholds.
EMS_VITAL_COLUMNS: List[str] = [
    "EMSSBP",
    "EMSPULSERATE",
    "EMSRESPIRATORYRATE",
    "EMSPULSEOXIMETRY",
    "EMSGCSMOTOR",
    "EMSTOTALGCS",
    "AGEyears",
]
# Cohort-filter columns (must be read so the filter reproduces the cohort).
COHORT_COLUMNS: List[str] = ["TRANSPORTMODE", "INTERFACILITYTRANSFER"]

# Columns that MUST NEVER be used to define proxy positivity (post-arrival or
# the opaque EMS-judgment flag). Used only as a guard in _quality_checks.
FORBIDDEN_COLUMNS: set = {
    "VPOEMSJUDGE",
    # Post-arrival respiratory context (intentionally excluded).
    "RESPIRATORYASSISTANCE",
    "SUPPLEMENTALOXYGEN",
    # Any ED/hospital-arrival vitals.
    "SBP",
    "PULSERATE",
    "RESPIRATORYRATE",
    "PULSEOXIMETRY",
    "TOTALGCS",
    "GCSMOTOR",
    "TEMPERATURE",
    "LOWESTSBP",
}

# Physiologically impossible "0" readings on low-threshold vitals are treated as
# not-a-valid-measurement (do NOT trigger a RED criterion) to avoid documenting
# artifacts as positives. Documented in the criterion mapping notes.
_LOW_VITAL_FLOOR = 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GuidelineProxyResult:
    source_path: str
    n_cohort: int = 0
    n_holdout: int = 0
    holdout_prevalence: float = float("nan")
    tables_saved: List[Path] = field(default_factory=list)
    figures_saved: List[Path] = field(default_factory=list)
    reports_saved: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_checks: List[Dict[str, object]] = field(default_factory=list)
    headline: Dict[str, object] = field(default_factory=dict)


def _warn(result: GuidelineProxyResult, message: str) -> None:
    result.warnings.append(message)
    warnings.warn(message, UserWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of a column; all-NaN if the column is absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean view of a 0/1 flag (value == 1). NaN/unknown -> False."""
    return _num(df, col) == 1


def _wilson_ci(k: int, n: int) -> tuple:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


# ---------------------------------------------------------------------------
# Criterion registry.
#
# Each entry documents one 2021 guideline criterion. ``builder`` returns a
# boolean Series (per-record present/absent) for mapped/partially_mapped
# criteria, or is None for unmapped criteria. NaN inputs -> criterion absent.
# ---------------------------------------------------------------------------
@dataclass
class Criterion:
    section: str  # "red" | "yellow"
    domain: str  # "injury pattern" | "mental status vitals" | "mechanism" | "risk factor"
    text: str
    mapped_status: str  # "mapped" | "partially_mapped" | "unmapped"
    source: str
    notes: str
    rollup: Optional[str] = None  # subcategory key that this criterion feeds
    builder: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def _build_criteria() -> List[Criterion]:
    def b(fn: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: fn(df).fillna(False).astype(bool)

    # ---- RED injury pattern (documented anatomic flags) ----------------
    red_injury = "red_injury_pattern_proxy"
    red_vitals = "red_mental_status_vitals_proxy"
    yellow_mech = "yellow_mechanism_proxy"
    yellow_risk = "yellow_risk_factor_proxy"

    criteria: List[Criterion] = [
        Criterion(
            "red", "injury pattern",
            "Penetrating injuries to head, neck, torso, or proximal extremities",
            "mapped", "TCCPEN == 1", "EMS-documented penetrating-injury triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCPEN")),
        ),
        Criterion(
            "red", "injury pattern",
            "Skull deformity / suspected skull fracture",
            "mapped", "TCCSKULLFRACTURE == 1", "EMS-documented skull-fracture triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCSKULLFRACTURE")),
        ),
        Criterion(
            "red", "injury pattern",
            "Suspected spinal injury with new motor or sensory loss / paralysis",
            "partially_mapped", "TCCPARALYSIS == 1",
            "Paralysis flag available; new sensory-only loss not separately coded.",
            red_injury, b(lambda df: _flag(df, "TCCPARALYSIS")),
        ),
        Criterion(
            "red", "injury pattern",
            "Chest wall instability, deformity, or suspected flail chest",
            "mapped", "TCCCHEST == 1", "EMS-documented chest-wall triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCCHEST")),
        ),
        Criterion(
            "red", "injury pattern",
            "Suspected pelvic fracture",
            "mapped", "TCCPELVIC == 1", "EMS-documented pelvic-fracture triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCPELVIC")),
        ),
        Criterion(
            "red", "injury pattern",
            "Suspected fracture of two or more proximal long bones",
            "mapped", "TCCLONGBONE == 1", "EMS-documented long-bone triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCLONGBONE")),
        ),
        Criterion(
            "red", "injury pattern",
            "Crushed, degloved, mangled, or pulseless extremity",
            "mapped", "TCCCRUSHED == 1", "EMS-documented crushed-extremity triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCCRUSHED")),
        ),
        Criterion(
            "red", "injury pattern",
            "Amputation proximal to wrist or ankle",
            "mapped", "TCCAMPUTATION == 1", "EMS-documented amputation triage flag.",
            red_injury, b(lambda df: _flag(df, "TCCAMPUTATION")),
        ),
        Criterion(
            "red", "injury pattern",
            "Active bleeding requiring tourniquet or wound packing",
            "unmapped", "",
            "No TQIP variable captures tourniquet/wound-packing use; excluded.",
            None, None,
        ),
        # ---- RED mental status / vital signs (raw EMS vitals) ----------
        Criterion(
            "red", "mental status vitals",
            "Unable to follow commands (motor GCS < 6)",
            "mapped", "EMSGCSMOTOR < 6",
            "Computed from raw scene motor GCS (NaN -> absent).",
            red_vitals, b(lambda df: _num(df, "EMSGCSMOTOR") < 6),
        ),
        Criterion(
            "red", "mental status vitals",
            "Depressed mental status (total GCS <= 13) [imperfect proxy]",
            "partially_mapped", "EMSTOTALGCS <= 13 (corroborated by TCCGCSLE13 == 1)",
            "Total-GCS proxy for the mental-status criterion; total GCS is an "
            "imperfect substitute for motor-only assessment.",
            red_vitals,
            b(lambda df: (_num(df, "EMSTOTALGCS") <= 13) | (_flag(df, "TCCGCSLE13"))),
        ),
        Criterion(
            "red", "mental status vitals",
            "Respiratory rate < 10 or > 29 breaths/min",
            "mapped",
            "EMSRESPIRATORYRATE > 29 OR (0 < EMSRESPIRATORYRATE < 10)",
            "Computed from raw scene respiratory rate; a recorded 0 is treated as "
            "not-a-valid-measurement for the low-rate arm (NaN -> absent).",
            red_vitals,
            b(lambda df: (_num(df, "EMSRESPIRATORYRATE") > 29)
              | ((_num(df, "EMSRESPIRATORYRATE") < 10) & (_num(df, "EMSRESPIRATORYRATE") > _LOW_VITAL_FLOOR))),
        ),
        Criterion(
            "red", "mental status vitals",
            "Respiratory distress or need for respiratory support",
            "unmapped", "",
            "Only post-arrival RESPIRATORYASSISTANCE/SUPPLEMENTALOXYGEN exist; "
            "post-arrival variables are excluded by design.",
            None, None,
        ),
        Criterion(
            "red", "mental status vitals",
            "Room-air pulse oximetry < 90%",
            "partially_mapped", "0 < EMSPULSEOXIMETRY < 90",
            "Room-air status is unknown in TQIP; a recorded 0 is treated as "
            "not-a-valid-measurement (NaN -> absent).",
            red_vitals,
            b(lambda df: (_num(df, "EMSPULSEOXIMETRY") < 90) & (_num(df, "EMSPULSEOXIMETRY") > _LOW_VITAL_FLOOR)),
        ),
        Criterion(
            "red", "mental status vitals",
            "Systolic BP < 90 mmHg for age 10-64",
            "mapped", "10 <= AGEyears < 65 AND 0 < EMSSBP < 90",
            "Computed from raw scene SBP and age (NaN -> absent).",
            red_vitals,
            b(lambda df: (_num(df, "AGEyears") >= 10) & (_num(df, "AGEyears") < 65)
              & (_num(df, "EMSSBP") < 90) & (_num(df, "EMSSBP") > _LOW_VITAL_FLOOR)),
        ),
        Criterion(
            "red", "mental status vitals",
            "Systolic BP < 110 mmHg for age >= 65",
            "mapped", "AGEyears >= 65 AND 0 < EMSSBP < 110 (corroborated by VPO65SBP110)",
            "Computed from raw scene SBP and age (NaN -> absent).",
            red_vitals,
            b(lambda df: (_num(df, "AGEyears") >= 65)
              & (_num(df, "EMSSBP") < 110) & (_num(df, "EMSSBP") > _LOW_VITAL_FLOOR)),
        ),
        Criterion(
            "red", "mental status vitals",
            "Pediatric hypotension: SBP < 70 + 2*age for age < 10",
            "mapped", "AGEyears < 10 AND 0 < EMSSBP < (70 + 2*AGEyears)",
            "Pediatric age-based hypotension threshold from raw scene SBP and age.",
            red_vitals,
            b(lambda df: (_num(df, "AGEyears") < 10)
              & (_num(df, "EMSSBP") < (70 + 2 * _num(df, "AGEyears")))
              & (_num(df, "EMSSBP") > _LOW_VITAL_FLOOR)),
        ),
        Criterion(
            "red", "mental status vitals",
            "Shock index > 1 (HR > SBP) for age >= 10",
            "mapped", "AGEyears >= 10 AND EMSPULSERATE > 0 AND EMSSBP > 0 AND (EMSPULSERATE / EMSSBP) > 1",
            "Computed from raw scene HR and SBP (NaN/zero -> absent).",
            red_vitals,
            b(lambda df: (_num(df, "AGEyears") >= 10)
              & (_num(df, "EMSPULSERATE") > 0) & (_num(df, "EMSSBP") > 0)
              & ((_num(df, "EMSPULSERATE") / _num(df, "EMSSBP").replace(0, np.nan)) > 1)),
        ),
        # ---- YELLOW mechanism (documented mechanism flags) ------------
        Criterion(
            "yellow", "mechanism",
            "High-risk auto crash: partial or complete ejection",
            "mapped", "VPOCRASHEJECT == 1", "EMS-documented ejection triage flag.",
            yellow_mech, b(lambda df: _flag(df, "VPOCRASHEJECT")),
        ),
        Criterion(
            "yellow", "mechanism",
            "High-risk auto crash: significant intrusion",
            "mapped", "VPOCRASHINTRUSION == 1", "EMS-documented intrusion triage flag.",
            yellow_mech, b(lambda df: _flag(df, "VPOCRASHINTRUSION")),
        ),
        Criterion(
            "yellow", "mechanism",
            "High-risk auto crash: death in passenger compartment",
            "mapped", "VPOCRASHDEATH == 1", "EMS-documented same-compartment-death flag.",
            yellow_mech, b(lambda df: _flag(df, "VPOCRASHDEATH")),
        ),
        Criterion(
            "yellow", "mechanism",
            "High-risk auto crash: vehicle telemetry consistent with severe injury",
            "mapped", "VPOCRASHTELEMETRY == 1", "EMS-documented vehicle-telemetry flag.",
            yellow_mech, b(lambda df: _flag(df, "VPOCRASHTELEMETRY")),
        ),
        Criterion(
            "yellow", "mechanism",
            "High-risk auto crash: need for extrication",
            "unmapped", "", "No TQIP variable captures extrication; excluded.",
            None, None,
        ),
        Criterion(
            "yellow", "mechanism",
            "High-risk auto crash: child age 0-9 unrestrained/unsecured",
            "unmapped", "",
            "No clean prehospital restraint-by-age variable; excluded.",
            None, None,
        ),
        Criterion(
            "yellow", "mechanism",
            "Rider separated from transport vehicle with significant impact",
            "partially_mapped", "VPOMOTORCYCLECRASH == 1",
            "Motorcycle-crash flag available; ATV/horse/other riders not separately coded.",
            yellow_mech, b(lambda df: _flag(df, "VPOMOTORCYCLECRASH")),
        ),
        Criterion(
            "yellow", "mechanism",
            "Pedestrian/bicyclist thrown, run over, or with significant impact",
            "mapped", "VPOAUTOPEDIMPACT == 1", "EMS-documented auto-vs-pedestrian/cyclist flag.",
            yellow_mech, b(lambda df: _flag(df, "VPOAUTOPEDIMPACT")),
        ),
        Criterion(
            "yellow", "mechanism",
            "Fall from height > 10 feet",
            "mapped", "VPOFALLADULT == 1 OR VPOFALLCHILD == 1",
            "EMS-documented adult/child high-fall triage flags.",
            yellow_mech, b(lambda df: _flag(df, "VPOFALLADULT") | _flag(df, "VPOFALLCHILD")),
        ),
        # ---- YELLOW risk factors --------------------------------------
        Criterion(
            "yellow", "risk factor",
            "Low-level falls in young children or older adults with significant head impact",
            "partially_mapped", "VPOFALLADULT == 1 OR VPOFALLCHILD == 1",
            "High-fall flags partially capture falls; low-level-fall + head-impact "
            "distinction is not separately coded.",
            yellow_risk, b(lambda df: _flag(df, "VPOFALLADULT") | _flag(df, "VPOFALLCHILD")),
        ),
        Criterion(
            "yellow", "risk factor",
            "Anticoagulant use",
            "mapped", "VPOANTICOAGULANT == 1", "EMS-documented anticoagulant triage flag.",
            yellow_risk, b(lambda df: _flag(df, "VPOANTICOAGULANT")),
        ),
        Criterion(
            "yellow", "risk factor",
            "Suspicion of child abuse",
            "unmapped", "", "No TQIP variable captures suspected abuse; excluded.",
            None, None,
        ),
        Criterion(
            "yellow", "risk factor",
            "Special, high-resource healthcare needs",
            "unmapped", "", "No TQIP variable captures special healthcare needs; excluded.",
            None, None,
        ),
        Criterion(
            "yellow", "risk factor",
            "Pregnancy > 20 weeks",
            "mapped", "VPOPREGNANCY20WKS == 1", "EMS-documented pregnancy>20wk triage flag.",
            yellow_risk, b(lambda df: _flag(df, "VPOPREGNANCY20WKS")),
        ),
        Criterion(
            "yellow", "risk factor",
            "Burns in conjunction with trauma",
            "mapped", "VPOTRAUMABURNS == 1 OR VPOBURNS == 1",
            "Trauma-with-burns flag available (VPOBURNS captures burns generally).",
            yellow_risk, b(lambda df: _flag(df, "VPOTRAUMABURNS") | _flag(df, "VPOBURNS")),
        ),
        Criterion(
            "yellow", "risk factor",
            "Pediatric-capable destination preference",
            "unmapped", "",
            "Depends on destination/system context, not patient state; excluded.",
            None, None,
        ),
    ]
    return criteria


# Subcategory rollup keys -> RED/YELLOW rollups.
RED_SUBCATEGORIES = ["red_injury_pattern_proxy", "red_mental_status_vitals_proxy"]
YELLOW_SUBCATEGORIES = ["yellow_mechanism_proxy", "yellow_risk_factor_proxy"]


def build_proxy_indicators(df: pd.DataFrame, criteria: List[Criterion]) -> pd.DataFrame:
    """Add all per-record proxy indicator columns to a copy of ``df``."""
    out = df.copy()

    subcat_keys = RED_SUBCATEGORIES + YELLOW_SUBCATEGORIES
    for key in subcat_keys:
        out[key] = False

    for crit in criteria:
        if crit.builder is None or crit.rollup is None:
            continue
        present = crit.builder(out)
        out[crit.rollup] = out[crit.rollup] | present.to_numpy()

    out["guideline_red_proxy_positive"] = (
        out[RED_SUBCATEGORIES[0]] | out[RED_SUBCATEGORIES[1]]
    )
    out["guideline_yellow_proxy_positive"] = (
        out[YELLOW_SUBCATEGORIES[0]] | out[YELLOW_SUBCATEGORIES[1]]
    )
    # Ordinal tier: 2 = red present, 1 = yellow only, 0 = none.
    tier = np.where(
        out["guideline_red_proxy_positive"],
        2,
        np.where(out["guideline_yellow_proxy_positive"], 1, 0),
    )
    out["guideline_tier_proxy"] = tier.astype(int)

    # Cast booleans to int for clean downstream metrics / CSV output.
    bool_cols = subcat_keys + [
        "guideline_red_proxy_positive",
        "guideline_yellow_proxy_positive",
    ]
    for col in bool_cols:
        out[col] = out[col].astype(int)
    return out


# ---------------------------------------------------------------------------
# Data loading (mirrors src/evaluation/missingness_audit.py _load_from_csv)
# ---------------------------------------------------------------------------
def _resolve_default_input() -> Path:
    primary = RAW_DATA_DIR / "dat5.csv"
    if primary.exists():
        return primary
    return SAMPLES_DATA_DIR / "dat5_limited.csv"


def _needed_columns() -> List[str]:
    needed = (
        list(TCC_FLAG_COLUMNS)
        + list(VPO_FLAG_COLUMNS)
        + list(EMS_VITAL_COLUMNS)
        + list(COHORT_COLUMNS)
        + [OUTCOME_COLUMN]
    )
    return list(dict.fromkeys(needed))


def _load_cohort_dataframe(
    input_path: Optional[Path], max_records: Optional[int], result: GuidelineProxyResult
) -> pd.DataFrame:
    path = Path(input_path) if input_path is not None else _resolve_default_input()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    result.source_path = str(path)

    header_only = pd.read_csv(path, nrows=0)
    available_all = list(header_only.columns)
    needed = [c for c in _needed_columns() if c in available_all]
    for col in COHORT_COLUMNS:
        if col in available_all and col not in needed:
            needed.append(col)

    df = pd.read_csv(path, usecols=needed, low_memory=False)
    if max_records is not None:
        df = df.head(max_records).copy()
        _warn(
            result,
            f"max_records={max_records} applied (head before cohort filter); record "
            "IDs will NOT align with full-cohort holdout predictions.",
        )

    from src.preprocessing.cohort_filter import apply_prehospital_ems_cohort_filter

    eligible_df, stats = apply_prehospital_ems_cohort_filter(df)
    eligible_df = eligible_df.reset_index(drop=True)
    eligible_df.insert(0, "record_id", [f"row_{i}" for i in range(len(eligible_df))])
    result.n_cohort = len(eligible_df)
    result.warnings.append(
        "Prehospital EMS cohort filter applied "
        f"({stats.records_before:,} -> {stats.records_eligible:,} records)."
    )
    return eligible_df


def _load_holdout(proxy_df: pd.DataFrame, result: GuidelineProxyResult) -> Optional[pd.DataFrame]:
    """Inner-join proxy indicators with saved XGBoost holdout predictions."""
    pred_path = NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH
    if not pred_path.exists():
        _warn(
            result,
            f"Holdout predictions not found at {pred_path}. Run primary training "
            "(menu option 4) before the guideline proxy benchmark.",
        )
        return None

    preds = pd.read_csv(pred_path)  # record_id, y_true, y_pred_prob
    merged = preds.merge(proxy_df, on="record_id", how="inner", suffixes=("", "_cohort"))
    if merged.empty:
        _warn(
            result,
            "No overlap between holdout prediction record IDs and cohort record IDs "
            "(likely a subsampled/mismatched input). Run on the full raw cohort CSV.",
        )
        return None

    # Optional logistic-regression holdout probabilities for the comparison table.
    lr_path = NFTI_POSITIVE_LR_HOLDOUT_PREDICTIONS_PATH
    if lr_path.exists():
        try:
            lr = pd.read_csv(lr_path)[["record_id", "y_pred_prob"]].rename(
                columns={"y_pred_prob": "lr_pred_prob"}
            )
            merged = merged.merge(lr, on="record_id", how="left")
        except Exception as exc:  # pragma: no cover - defensive
            _warn(result, f"Could not load LR holdout predictions: {exc}")
    return merged


# ---------------------------------------------------------------------------
# Mapping audit table
# ---------------------------------------------------------------------------
def write_criterion_mapping(
    criteria: List[Criterion],
    holdout: Optional[pd.DataFrame],
    result: GuidelineProxyResult,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for crit in criteria:
        n_positive: object = ""
        if crit.builder is not None and holdout is not None:
            try:
                n_positive = int(crit.builder(holdout).sum())
            except Exception:
                n_positive = ""
        rows.append(
            {
                "guideline_section": crit.section,
                "guideline_domain": crit.domain,
                "criterion_text": crit.text,
                "mapped_status": crit.mapped_status,
                "source_variable_or_logic": crit.source,
                "notes": crit.notes,
                "n_positive_in_holdout": n_positive,
            }
        )
    out = pd.DataFrame(rows, columns=[
        "guideline_section", "guideline_domain", "criterion_text", "mapped_status",
        "source_variable_or_logic", "notes", "n_positive_in_holdout",
    ])
    GUIDELINE_PROXY_CRITERION_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GUIDELINE_PROXY_CRITERION_MAPPING_PATH, index=False)
    result.tables_saved.append(GUIDELINE_PROXY_CRITERION_MAPPING_PATH)
    return out


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------
def _rule_metrics_row(rule_name: str, threshold_or_rule: str, y_true, y_score) -> Dict[str, object]:
    m = calculate_binary_classification_metrics(y_true, y_score, threshold=0.5)
    return {
        "rule_name": rule_name,
        "threshold_or_rule": threshold_or_rule,
        "n": m["n"],
        "nfti_prevalence": m["prevalence"],
        "tp": m["TP"], "fp": m["FP"], "tn": m["TN"], "fn": m["FN"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "ppv": m["precision"],
        "npv": m["NPV"],
        "accuracy": m["accuracy"],
        "f1": m["F1"],
    }


def evaluate_rules(holdout: pd.DataFrame, result: GuidelineProxyResult) -> pd.DataFrame:
    y_true = pd.to_numeric(holdout["y_true"], errors="coerce").to_numpy(dtype=float)
    rules = [
        ("guideline_red_proxy_positive", "RED proxy positive",
         holdout["guideline_red_proxy_positive"]),
        ("guideline_yellow_only", "YELLOW-only proxy positive",
         (holdout["guideline_tier_proxy"] == 1).astype(int)),
        ("guideline_yellow_or_red", "YELLOW-or-RED proxy positive",
         ((holdout["guideline_red_proxy_positive"] == 1)
          | (holdout["guideline_yellow_proxy_positive"] == 1)).astype(int)),
    ]
    rows = [
        _rule_metrics_row(name, label, y_true,
                          pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
        for name, label, series in rules
    ]
    out = pd.DataFrame(rows)
    GUIDELINE_PROXY_RULE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GUIDELINE_PROXY_RULE_METRICS_PATH, index=False)
    result.tables_saved.append(GUIDELINE_PROXY_RULE_METRICS_PATH)
    return out


def evaluate_tiers(holdout: pd.DataFrame, result: GuidelineProxyResult) -> pd.DataFrame:
    y_true = pd.to_numeric(holdout["y_true"], errors="coerce")
    tier = holdout["guideline_tier_proxy"].astype(int)
    tier_labels = {0: "No mapped criteria", 1: "Yellow only", 2: "Red present"}
    rows: List[Dict[str, object]] = []
    for t in (0, 1, 2):
        mask = tier == t
        n = int(mask.sum())
        yt = y_true[mask]
        k = int((yt == 1).sum())
        rate = float(yt.mean()) if n else float("nan")
        lo, hi = _wilson_ci(k, n)
        rows.append(
            {
                "guideline_tier_proxy": t,
                "tier_label": tier_labels[t],
                "n": n,
                "nfti_positive_count": k,
                "observed_nfti_rate": rate,
                "ci95_lower": lo,
                "ci95_upper": hi,
            }
        )
    out = pd.DataFrame(rows)

    # Optional ordinal-proxy AUROC using tier 0/1/2 as an ordinal score.
    ordinal_auroc = float("nan")
    yt_all = y_true.to_numpy(dtype=float)
    valid = ~np.isnan(yt_all)
    if valid.sum() and len(np.unique(yt_all[valid])) == 2:
        try:
            ordinal_auroc = float(roc_auc_score(yt_all[valid], tier.to_numpy(dtype=float)[valid]))
        except ValueError:
            ordinal_auroc = float("nan")
    result.headline["ordinal_proxy_auroc"] = ordinal_auroc

    GUIDELINE_PROXY_TIER_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GUIDELINE_PROXY_TIER_TABLE_PATH, index=False)
    result.tables_saved.append(GUIDELINE_PROXY_TIER_TABLE_PATH)
    return out


# ---------------------------------------------------------------------------
# Model-vs-guideline comparison
# ---------------------------------------------------------------------------
def _read_validation_locked_threshold(result: GuidelineProxyResult) -> Optional[float]:
    path = NFTI_POSITIVE_XGB_SELECTED_80_SENSITIVITY_PATH
    if not path.exists():
        _warn(result, f"Validation-locked threshold file not found at {path}.")
        return None
    try:
        sel = pd.read_csv(path)
        if "selected_threshold" in sel.columns and len(sel):
            return float(sel["selected_threshold"].iloc[0])
    except Exception as exc:  # pragma: no cover - defensive
        _warn(result, f"Could not read validation-locked threshold: {exc}")
    return None


def _comparison_row(
    label: str, threshold_or_rule: str, y_true, y_score, threshold: float,
    *, ranking: bool,
) -> Dict[str, object]:
    m = calculate_binary_classification_metrics(y_true, y_score, threshold=threshold)
    row: Dict[str, object] = {
        "rule_or_model": label,
        "threshold_or_rule": threshold_or_rule,
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "ppv": m["precision"],
        "npv": m["NPV"],
        "f1": m["F1"],
        "auroc": m["AUROC"] if ranking else "",
        "auprc": m["AUPRC"] if ranking else "",
        "brier": m["Brier"] if ranking else "",
        "tp": m["TP"], "fp": m["FP"], "tn": m["TN"], "fn": m["FN"],
    }
    return row


def compare_against_model(holdout: pd.DataFrame, result: GuidelineProxyResult) -> pd.DataFrame:
    y_true = pd.to_numeric(holdout["y_true"], errors="coerce").to_numpy(dtype=float)
    xgb_prob = pd.to_numeric(holdout["y_pred_prob"], errors="coerce").to_numpy(dtype=float)
    red = pd.to_numeric(holdout["guideline_red_proxy_positive"], errors="coerce").to_numpy(dtype=float)
    yellow_or_red = (
        (holdout["guideline_red_proxy_positive"] == 1)
        | (holdout["guideline_yellow_proxy_positive"] == 1)
    ).astype(float).to_numpy()

    rows: List[Dict[str, object]] = []
    # Binary guideline rules (no AUROC/AUPRC/Brier; threshold 0.5 == the rule).
    rows.append(_comparison_row(
        "Guideline RED proxy", "RED criteria present", y_true, red, 0.5, ranking=False))
    rows.append(_comparison_row(
        "Guideline YELLOW-or-RED proxy", "any mapped criterion present",
        y_true, yellow_or_red, 0.5, ranking=False))

    # XGBoost fixed 0.5.
    rows.append(_comparison_row(
        "XGBoost", "probability >= 0.50", y_true, xgb_prob, 0.5, ranking=True))

    # XGBoost validation-locked high-sensitivity threshold.
    locked = _read_validation_locked_threshold(result)
    if locked is not None:
        rows.append(_comparison_row(
            "XGBoost", f"validation-locked >= {locked:.3f}",
            y_true, xgb_prob, locked, ranking=True))

    # Optional logistic regression fixed 0.5.
    if "lr_pred_prob" in holdout.columns and holdout["lr_pred_prob"].notna().any():
        lr_prob = pd.to_numeric(holdout["lr_pred_prob"], errors="coerce").to_numpy(dtype=float)
        rows.append(_comparison_row(
            "Logistic Regression", "probability >= 0.50", y_true, lr_prob, 0.5, ranking=True))

    out = pd.DataFrame(rows, columns=[
        "rule_or_model", "threshold_or_rule", "sensitivity", "specificity", "ppv",
        "npv", "f1", "auroc", "auprc", "brier", "tp", "fp", "tn", "fn",
    ])
    GUIDELINE_PROXY_VS_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GUIDELINE_PROXY_VS_MODEL_METRICS_PATH, index=False)
    result.tables_saved.append(GUIDELINE_PROXY_VS_MODEL_METRICS_PATH)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_nfti_rate_by_tier(tier_df: pd.DataFrame, result: GuidelineProxyResult) -> None:
    if tier_df.empty:
        return
    data = tier_df.sort_values("guideline_tier_proxy").reset_index(drop=True)
    rate = data["observed_nfti_rate"].to_numpy(dtype=float)
    lo = data["ci95_lower"].to_numpy(dtype=float)
    hi = data["ci95_upper"].to_numpy(dtype=float)
    yerr = np.vstack([rate - lo, hi - rate])
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(8, 5))
    apply_manuscript_grid(ax)
    ax.bar(
        x, rate, color=NAVY, alpha=0.85,
        yerr=yerr, ecolor=RED, capsize=4,
        error_kw={"elinewidth": 1.2, "alpha": 0.8},
    )
    for xi, (r, n) in enumerate(zip(rate, data["n"])):
        ax.annotate(
            f"{r * 100:.1f}%\n(n={int(n):,})",
            xy=(xi, r), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(data["tier_label"])
    ax.set_xlabel("Guideline proxy tier")
    ax.set_ylabel("Observed NFTI-positive rate")
    ax.set_title("Observed NFTI Rate by 2021 Guideline Proxy Tier (holdout)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    GUIDELINE_PROXY_TIER_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GUIDELINE_PROXY_TIER_FIGURE_PATH, dpi=300)
    plt.close(fig)
    result.figures_saved.append(GUIDELINE_PROXY_TIER_FIGURE_PATH)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def _quality_checks(
    criteria: List[Criterion],
    holdout: Optional[pd.DataFrame],
    mapping_df: pd.DataFrame,
    proxy_cohort: pd.DataFrame,
    result: GuidelineProxyResult,
) -> None:
    def check(name: str, passed: bool, detail: str = "") -> None:
        result.quality_checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            _warn(result, f"QUALITY CHECK FAILED: {name} -- {detail}")

    # Exact identifier tokens referenced across all source-logic strings (so
    # e.g. "SBP" is not matched inside "EMSSBP").
    referenced_tokens: set = set()
    for crit in criteria:
        if crit.source:
            referenced_tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", crit.source))

    # 1. VPOEMSJUDGE never referenced as a source.
    check(
        "vpoemsjudge_not_used",
        "VPOEMSJUDGE" not in referenced_tokens,
        "VPOEMSJUDGE must not be used to define proxy positivity.",
    )

    # 2. No forbidden (post-arrival) columns referenced in any source logic.
    referenced_forbidden = sorted(FORBIDDEN_COLUMNS & referenced_tokens)
    check(
        "no_post_arrival_variables",
        not referenced_forbidden,
        f"forbidden columns referenced: {referenced_forbidden}" if referenced_forbidden else "ok",
    )

    # 3. Every mapped/partially_mapped criterion has documented source logic.
    undocumented = [
        c.text for c in criteria
        if c.mapped_status in ("mapped", "partially_mapped") and not c.source.strip()
    ]
    check(
        "mapped_criteria_documented",
        not undocumented,
        f"missing source logic: {undocumented}" if undocumented else "ok",
    )

    # 4. Holdout alignment: merge size matches the saved holdout predictions.
    if holdout is not None:
        pred_n = len(pd.read_csv(NFTI_POSITIVE_XGB_HOLDOUT_PREDICTIONS_PATH))
        check(
            "holdout_alignment",
            len(holdout) == pred_n,
            f"merged holdout n={len(holdout)} vs predictions n={pred_n}.",
        )
        # 5. Nonzero, plausible positive counts.
        red_n = int((holdout["guideline_red_proxy_positive"] == 1).sum())
        yellow_n = int((holdout["guideline_yellow_proxy_positive"] == 1).sum())
        tier_counts = holdout["guideline_tier_proxy"].value_counts().to_dict()
        check("red_positive_nonzero", red_n > 0, f"red positive n={red_n}.")
        check("yellow_positive_nonzero", yellow_n > 0, f"yellow positive n={yellow_n}.")
        check(
            "all_tiers_present",
            all(t in tier_counts for t in (0, 1, 2)),
            f"tier counts: {tier_counts}.",
        )
    else:
        check("holdout_alignment", False, "holdout predictions unavailable.")

    # 6. Missing variables -> unmapped, never silently ignored.
    n_unmapped = int((mapping_df["mapped_status"] == "unmapped").sum())
    check(
        "unmapped_documented",
        n_unmapped > 0,
        f"{n_unmapped} criteria explicitly marked unmapped.",
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def write_summary_report(
    criteria: List[Criterion],
    mapping_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    tier_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    holdout: Optional[pd.DataFrame],
    result: GuidelineProxyResult,
) -> None:
    n_mapped_red = int(
        ((mapping_df["guideline_section"] == "red")
         & (mapping_df["mapped_status"].isin(["mapped", "partially_mapped"]))).sum()
    )
    n_mapped_yellow = int(
        ((mapping_df["guideline_section"] == "yellow")
         & (mapping_df["mapped_status"].isin(["mapped", "partially_mapped"]))).sum()
    )
    unmapped_examples = mapping_df.loc[
        mapping_df["mapped_status"] == "unmapped", "criterion_text"
    ].tolist()
    examples_unmapped = "; ".join(unmapped_examples[:3]) if unmapped_examples else "(none)"

    # Headline metrics for the yellow-or-red rule.
    redyellow = rule_df[rule_df["rule_name"] == "guideline_yellow_or_red"]
    sens = spec = ppv = npv = float("nan")
    if not redyellow.empty:
        r = redyellow.iloc[0]
        sens, spec, ppv, npv = r["sensitivity"], r["specificity"], r["ppv"], r["npv"]

    rate_by_tier = {int(row["guideline_tier_proxy"]): row["observed_nfti_rate"]
                    for _, row in tier_df.iterrows()}
    rate_none = rate_by_tier.get(0, float("nan"))
    rate_yellow = rate_by_tier.get(1, float("nan"))
    rate_red = rate_by_tier.get(2, float("nan"))

    lines: List[str] = []
    lines.append("# 2021 Field Triage Guideline Proxy Benchmark (Exploratory)\n")
    lines.append(f"- Data source: `{result.source_path}`")
    lines.append(f"- Prehospital EMS cohort size: {result.n_cohort:,}")
    if holdout is not None:
        lines.append(f"- Holdout records evaluated: {result.n_holdout:,}")
        lines.append(f"- Holdout NFTI-positive prevalence: {_fmt_pct(result.holdout_prevalence)}")
    lines.append("")

    lines.append("## Scope and important caveats")
    lines.append(
        "- This is an **available-variable proxy**, not a full implementation of the "
        "2021 National Guideline for the Field Triage of Injured Patients."
    )
    lines.append(
        "- Some guideline criteria **cannot be reconstructed from TQIP** and are "
        "explicitly excluded from the proxy definition (marked `unmapped` in the "
        "criterion mapping table) rather than silently ignored."
    )
    lines.append(
        "- The opaque EMS-provider-judgment flag (**VPOEMSJUDGE**) was intentionally "
        "**not used**."
    )
    lines.append(
        "- No hospital-arrival / post-arrival variables were used to define proxy "
        "positivity; every source variable is a prehospital (scene) value."
    )
    lines.append(
        "- The guideline was **not designed to predict NFTI**, so this is an "
        "**exploratory benchmark only**. The primary XGBoost model also has access "
        "to these field-triage criteria as features, so this comparison illustrates "
        "incremental value rather than an independent head-to-head."
    )
    lines.append("")

    lines.append("## Criterion mapping summary")
    for status in ("mapped", "partially_mapped", "unmapped"):
        n = int((mapping_df["mapped_status"] == status).sum())
        lines.append(f"- {status}: {n} criteria")
    lines.append(f"- See `{GUIDELINE_PROXY_CRITERION_MAPPING_PATH.name}` for the full mapping audit.")
    lines.append("")

    if not rule_df.empty:
        lines.append("## Guideline proxy rule performance (holdout)")
        for _, r in rule_df.iterrows():
            lines.append(
                f"- {r['threshold_or_rule']}: n={int(r['n']):,}, "
                f"sensitivity={_fmt_pct(r['sensitivity'])}, "
                f"specificity={_fmt_pct(r['specificity'])}, "
                f"PPV={_fmt_pct(r['ppv'])}, NPV={_fmt_pct(r['npv'])}, "
                f"F1={r['f1']:.3f} "
                f"(TP={int(r['tp'])}, FP={int(r['fp'])}, TN={int(r['tn'])}, FN={int(r['fn'])})"
            )
        lines.append("")

    if not tier_df.empty:
        lines.append("## Observed NFTI rate by guideline proxy tier (holdout)")
        for _, r in tier_df.iterrows():
            lines.append(
                f"- {r['tier_label']}: n={int(r['n']):,}, "
                f"NFTI rate={_fmt_pct(r['observed_nfti_rate'])} "
                f"(95% CI {_fmt_pct(r['ci95_lower'])}-{_fmt_pct(r['ci95_upper'])})"
            )
        ord_auroc = result.headline.get("ordinal_proxy_auroc", float("nan"))
        if isinstance(ord_auroc, float) and not np.isnan(ord_auroc):
            lines.append(f"- Ordinal proxy AUROC (tier 0/1/2 as score): {ord_auroc:.3f}")
        lines.append("")

    if not comparison_df.empty:
        lines.append("## Guideline proxy vs model operating points (holdout)")
        lines.append("| rule/model | threshold_or_rule | sens | spec | PPV | NPV | F1 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in comparison_df.iterrows():
            lines.append(
                f"| {r['rule_or_model']} | {r['threshold_or_rule']} | "
                f"{_fmt_pct(r['sensitivity'])} | {_fmt_pct(r['specificity'])} | "
                f"{_fmt_pct(r['ppv'])} | {_fmt_pct(r['npv'])} | {r['f1']:.3f} |"
            )
        lines.append("")
        lines.append(f"- See `{GUIDELINE_PROXY_VS_MODEL_METRICS_PATH.name}` for AUROC/AUPRC/Brier and confusion counts.")
        lines.append("")

    if result.quality_checks:
        lines.append("## Quality checks")
        for qc in result.quality_checks:
            status = "PASS" if qc["passed"] else "FAIL"
            lines.append(f"- [{status}] {qc['check']}: {qc['detail']}")
        lines.append("")

    lines.append("## Manuscript text snippets")
    lines.append(
        f"Using available TQIP variables, {n_mapped_red} red criteria and "
        f"{n_mapped_yellow} yellow criteria from the 2021 Field Triage Guidelines "
        "were partially reconstructed as an exploratory benchmark. Several criteria, "
        f"including {examples_unmapped}, could not be represented in TQIP and were "
        "excluded from the proxy definition."
    )
    lines.append("")
    lines.append(
        f"The guideline proxy (any mapped red or yellow criterion present) "
        f"demonstrated {_fmt_pct(sens)} sensitivity, {_fmt_pct(spec)} specificity, "
        f"{_fmt_pct(ppv)} PPV, and {_fmt_pct(npv)} NPV for NFTI-positive status on the "
        f"holdout cohort. Observed NFTI rates increased from {_fmt_pct(rate_none)} "
        f"among patients with no mapped criteria to {_fmt_pct(rate_yellow)} among "
        f"yellow-only patients and {_fmt_pct(rate_red)} among red-criteria patients."
    )
    lines.append("")
    lines.append(
        "This comparison is exploratory: the 2021 guideline was not designed to "
        "predict NFTI, and the proxy is an incomplete, available-variable "
        "approximation of the guideline rather than a validated implementation."
    )
    lines.append("")

    GUIDELINE_PROXY_BENCHMARK_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDELINE_PROXY_BENCHMARK_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    result.reports_saved.append(GUIDELINE_PROXY_BENCHMARK_SUMMARY_PATH)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_guideline_proxy_benchmark(
    input_path: Optional[Path] = None,
    *,
    max_records: Optional[int] = None,
) -> GuidelineProxyResult:
    """Run the exploratory 2021 guideline proxy benchmark end to end."""
    ensure_dirs()
    GUIDELINE_PROXY_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    GUIDELINE_PROXY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    result = GuidelineProxyResult(source_path="")
    criteria = _build_criteria()

    cohort_df = _load_cohort_dataframe(input_path, max_records, result)
    proxy_cohort = build_proxy_indicators(cohort_df, criteria)
    holdout = _load_holdout(proxy_cohort, result)

    if holdout is not None:
        result.n_holdout = len(holdout)
        yt = pd.to_numeric(holdout["y_true"], errors="coerce")
        result.holdout_prevalence = float((yt == 1).mean()) if len(yt) else float("nan")

    mapping_df = write_criterion_mapping(criteria, holdout, result)

    if holdout is not None:
        rule_df = evaluate_rules(holdout, result)
        tier_df = evaluate_tiers(holdout, result)
        comparison_df = compare_against_model(holdout, result)
        plot_nfti_rate_by_tier(tier_df, result)
    else:
        rule_df = pd.DataFrame()
        tier_df = pd.DataFrame()
        comparison_df = pd.DataFrame()

    _quality_checks(criteria, holdout, mapping_df, proxy_cohort, result)
    write_summary_report(
        criteria, mapping_df, rule_df, tier_df, comparison_df, holdout, result
    )

    _print_summary(result)
    return result


def _print_summary(result: GuidelineProxyResult) -> None:
    print("\n=== 2021 Field Triage Guideline Proxy Benchmark (Exploratory) ===")
    print(f"Data source:        {result.source_path}")
    print(f"Cohort size:        {result.n_cohort:,}")
    print(f"Holdout evaluated:  {result.n_holdout:,}")
    if not np.isnan(result.holdout_prevalence):
        print(f"Holdout prevalence: {result.holdout_prevalence * 100:.1f}%")
    print(f"Tables saved:       {len(result.tables_saved)}")
    print(f"Figures saved:      {len(result.figures_saved)}")
    print(f"Reports saved:      {len(result.reports_saved)}")
    failed = [qc["check"] for qc in result.quality_checks if not qc["passed"]]
    if failed:
        print(f"Quality checks FAILED: {', '.join(failed)}")
    else:
        print("Quality checks:     all passed")
    print("Guideline proxy benchmark complete.\n")
