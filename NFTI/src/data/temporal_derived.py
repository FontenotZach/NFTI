"""
Derived temporal features from NTDS-style *DAYS + *HRS pairs and optional calendar dates.

All elapsed-minute features assume every referenced event uses the **same day anchor**
(typically offset from injury incident date), matching ACS NTDS flat-file conventions.
If anchors differ in your export, these intervals should be revisited.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np
import pandas as pd

# Headers declared in header_definitions.csv; registered at runtime if missing from the CSV columns.
TEMPORAL_DERIVED_HEADER_NAMES = (
    "ems_notify_to_scene_arrival_min",
    "ems_scene_arrival_to_leave_scene_min",
    "ems_notify_to_leave_scene_min",
    "ems_leave_scene_to_ed_arrival_min",
    "ems_notify_to_ed_arrival_min",
    "encounter_month",
)


def register_temporal_derived_headers(trauma_dataset, header_info: Mapping[str, Mapping[str, Any]]) -> None:
    """Add schema rows for computed temporal columns so they participate in usage/timing filters."""
    existing = {h.name for h in trauma_dataset.headers}
    for name in TEMPORAL_DERIVED_HEADER_NAMES:
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


def _elapsed_minutes(day_a: float, hrs_a: float, day_b: float, hrs_b: float) -> float:
    """Minutes from event A to event B; HRS are decimal hours on that offset day."""
    if any(np.isnan(x) for x in (day_a, hrs_a, day_b, hrs_b)):
        return np.nan
    ta = day_a * 1440.0 + hrs_a * 60.0
    tb = day_b * 1440.0 + hrs_b * 60.0
    return float(tb - ta)


def _parse_month_from_date_string(raw: Any) -> float:
    """Return calendar month 1.0–12.0, or NaN if unparseable."""
    if raw is None:
        return np.nan
    try:
        if pd.isna(raw):
            return np.nan
    except TypeError:
        pass
    s = str(raw).strip().strip('"').strip("'")
    if not s or s.upper() in ("NA", "NAN", "NONE"):
        return np.nan
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return float(datetime.strptime(s, fmt).month)
        except ValueError:
            continue
    return np.nan


def _first_calendar_month(row: Any) -> float:
    """Prefer injury incident date, then hospital arrival calendar date."""
    for key in ("INJURYINCIDENTDATE", "HOSPITALARRIVALDATE"):
        if isinstance(row, pd.Series):
            if key not in row.index:
                continue
            raw = row[key]
        else:
            raw = row.get(key) if hasattr(row, "get") else None
        m = _parse_month_from_date_string(raw)
        if not np.isnan(m):
            return m
    return np.nan


def compute_temporal_derived(row: Any) -> Dict[str, float]:
    """
    Compute elapsed intervals (minutes) and encounter month.

    Date priority for ``encounter_month``: INJURYINCIDENTDATE, then HOSPITALARRIVALDATE.
    """
    out: Dict[str, float] = {}

    ems_notify_d = _field(row, "EMSNOTIFYDAYS")
    ems_notify_h = _field(row, "EMSNOTIFYHRS")
    ems_arr_d = _field(row, "EMSARRIVALDAYS")
    ems_arr_h = _field(row, "EMSARRIVALHRS")
    ems_left_d = _field(row, "EMSLEFTDAYS")
    ems_left_h = _field(row, "EMSLEFTHRS")
    hosp_d = _field(row, "HOSPITALARRIVALDAYS")
    hosp_h = _field(row, "HOSPITALARRIVALHRS")

    out["ems_notify_to_scene_arrival_min"] = _elapsed_minutes(
        ems_notify_d, ems_notify_h, ems_arr_d, ems_arr_h
    )
    out["ems_scene_arrival_to_leave_scene_min"] = _elapsed_minutes(
        ems_arr_d, ems_arr_h, ems_left_d, ems_left_h
    )
    out["ems_notify_to_leave_scene_min"] = _elapsed_minutes(
        ems_notify_d, ems_notify_h, ems_left_d, ems_left_h
    )
    out["ems_leave_scene_to_ed_arrival_min"] = _elapsed_minutes(
        ems_left_d, ems_left_h, hosp_d, hosp_h
    )
    out["ems_notify_to_ed_arrival_min"] = _elapsed_minutes(
        ems_notify_d, ems_notify_h, hosp_d, hosp_h
    )

    out["encounter_month"] = _first_calendar_month(row)

    return out


def merge_temporal_derived_into_record(
    filtered_record: MutableMapping[str, Any],
    data_row: Any,
    *,
    valid_names: set,
) -> None:
    """Overwrite derived header slots in filtered_record when those headers are active."""
    derived = compute_temporal_derived(data_row)
    for k, v in derived.items():
        if k in valid_names:
            filtered_record[k] = v
