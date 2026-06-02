"""
Derived clinical scores and flags from scene (EMS) vitals.

Weighted Revised Trauma Score (RTS) uses Champion et al. category weights applied to
coded GCS, SBP, and RR (scene values). Hypotension/tachycardia use fixed adult thresholds.
Pediatric-adjusted thresholds are not applied here.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Set

import numpy as np
import pandas as pd

# Champion et al. RTS weights on 0–4 category codes.
RTS_WEIGHT_GCS = 0.9368
RTS_WEIGHT_SBP = 0.7326
RTS_WEIGHT_RR = 0.2908

# Adult-oriented EMS thresholds (document in header_definitions).
EMS_HYPOTENSION_SBP_LT = 90.0  # mmHg
EMS_TACHYCARDIA_HR_GT = 120.0  # beats/min

CLINICAL_DERIVED_HEADER_NAMES = (
    "rts_ems_weighted",
    "ems_hypotension",
    "ems_tachycardia",
)


def register_clinical_derived_headers(trauma_dataset, header_info: Mapping[str, Mapping[str, Any]]) -> None:
    existing = {h.name for h in trauma_dataset.headers}
    for name in CLINICAL_DERIVED_HEADER_NAMES:
        if name in existing:
            continue
        meta = header_info.get(name, {})
        dtype = str(meta.get("data_type", "")).strip()
        if not dtype:
            continue
        trauma_dataset.add_header(
            name,
            ntds_page=str(meta.get("ntds_page", "")),
            definition=str(meta.get("definition", "")),
            timing=str(meta.get("timing", "")),
            data_type=dtype,
            usage=str(meta.get("usage", "")),
            one_hot_grouping=str(meta.get("one_hot_grouping", "")),
            y=str(meta.get("y", "")),
        )


def _field(row: Any, key: str) -> float:
    if isinstance(row, pd.Series):
        if key not in row.index:
            return np.nan
        v = row[key]
    else:
        v = row.get(key) if hasattr(row, "get") else None
    if v is None:
        return np.nan
    try:
        if pd.isna(v):
            return np.nan
    except TypeError:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _biu_invalid(row: Any, key: str) -> bool:
    """True if companion *_BIU indicates blank/invalid (typically 1)."""
    v = _field(row, key)
    if np.isnan(v):
        return False
    return float(v) != 0.0


def _code_gcs_total(gcs: float) -> float:
    if np.isnan(gcs):
        return np.nan
    if gcs > 12:
        return 4.0
    if gcs > 8:
        return 3.0
    if gcs > 5:
        return 2.0
    if gcs > 3:
        return 1.0
    return 0.0


def _code_sbp(sbp: float) -> float:
    if np.isnan(sbp):
        return np.nan
    if sbp > 89:
        return 4.0
    if sbp > 75:
        return 3.0
    if sbp > 49:
        return 2.0
    if sbp > 0:
        return 1.0
    return 0.0


def _code_rr(rr: float) -> float:
    if np.isnan(rr):
        return np.nan
    if 10 <= rr <= 29:
        return 4.0
    if rr > 29:
        return 3.0
    if 6 <= rr <= 9:
        return 2.0
    if rr > 0:
        return 1.0
    return 0.0


def compute_clinical_derived(row: Any) -> Dict[str, float]:
    gcs = _field(row, "EMSTOTALGCS")
    sbp = _field(row, "EMSSBP")
    rr = _field(row, "EMSRESPIRATORYRATE")
    hr = _field(row, "EMSPULSERATE")

    if _biu_invalid(row, "EMSTOTALGCS_BIU"):
        gcs = np.nan
    if _biu_invalid(row, "EMSSBP_BIU"):
        sbp = np.nan
    if _biu_invalid(row, "EMSRESPIRATORYRATE_BIU"):
        rr = np.nan
    if _biu_invalid(row, "EMSPULSERATE_BIU"):
        hr = np.nan

    cg = _code_gcs_total(gcs)
    cs = _code_sbp(sbp)
    cr = _code_rr(rr)

    out: Dict[str, float] = {}

    if np.isnan(cg) or np.isnan(cs) or np.isnan(cr):
        out["rts_ems_weighted"] = np.nan
    else:
        out["rts_ems_weighted"] = RTS_WEIGHT_GCS * cg + RTS_WEIGHT_SBP * cs + RTS_WEIGHT_RR * cr

    if np.isnan(sbp):
        out["ems_hypotension"] = np.nan
    else:
        out["ems_hypotension"] = 1.0 if sbp < EMS_HYPOTENSION_SBP_LT else 0.0

    if np.isnan(hr):
        out["ems_tachycardia"] = np.nan
    else:
        out["ems_tachycardia"] = 1.0 if hr > EMS_TACHYCARDIA_HR_GT else 0.0

    return out


def merge_clinical_derived_into_record(
    filtered_record: MutableMapping[str, Any],
    data_row: Any,
    *,
    valid_names: Set[str],
) -> None:
    derived = compute_clinical_derived(data_row)
    for k, v in derived.items():
        if k in valid_names:
            filtered_record[k] = v
