from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from src.paths import REPORTS_DIR, ensure_dirs

# Primary prehospital EMS cohort transport modes (ground / helicopter / fixed-wing).
ELIGIBLE_TRANSPORT_MODES = {1, 2, 3}
NON_EMS_TRANSPORT_MODES = {4, 5, 6}

TRANSPORTMODE_COLUMN = "TRANSPORTMODE"
INTERFACILITYTRANSFER_COLUMN = "INTERFACILITYTRANSFER"


@dataclass(frozen=True)
class PrehospitalCohortStats:
    records_before: int
    excluded_non_ems_transport: int
    excluded_interfacility_transfer: int
    excluded_overlap: int
    excluded_total: int
    records_eligible: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "records_before": self.records_before,
            "excluded_non_ems_transport": self.excluded_non_ems_transport,
            "excluded_interfacility_transfer": self.excluded_interfacility_transfer,
            "excluded_overlap": self.excluded_overlap,
            "excluded_total": self.excluded_total,
            "records_eligible": self.records_eligible,
        }


def _transport_series(df: pd.DataFrame) -> pd.Series:
    if TRANSPORTMODE_COLUMN not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)

    transport = pd.to_numeric(df[TRANSPORTMODE_COLUMN], errors="coerce")
    return ~transport.isin(ELIGIBLE_TRANSPORT_MODES)


def _interfacility_series(df: pd.DataFrame) -> pd.Series:
    if INTERFACILITYTRANSFER_COLUMN not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)

    ift = pd.to_numeric(df[INTERFACILITYTRANSFER_COLUMN], errors="coerce")
    return ift != 0

def apply_prehospital_ems_cohort_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, PrehospitalCohortStats]:
    """
    Build the primary prehospital EMS cohort for modeling.

    Keeps ambulance-transported patients (TRANSPORTMODE 1/2/3) and excludes
    walk-in/police/other transport (4/5/6) and interfacility transfers (IFT=1).
    Missing EMS vitals or BIU codes do not exclude otherwise eligible records.
    """
    if df.empty:
        stats = PrehospitalCohortStats(0, 0, 0, 0, 0, 0)
        return df.copy(), stats

    exclude_transport = _transport_series(df)
    exclude_ift = _interfacility_series(df)
    exclude_any = exclude_transport | exclude_ift

    stats = PrehospitalCohortStats(
        records_before=len(df),
        excluded_non_ems_transport=int(exclude_transport.sum()),
        excluded_interfacility_transfer=int(exclude_ift.sum()),
        excluded_overlap=int((exclude_transport & exclude_ift).sum()),
        excluded_total=int(exclude_any.sum()),
        records_eligible=int((~exclude_any).sum()),
    )

    eligible_df = df.loc[~exclude_any].copy()
    return eligible_df, stats


def format_prehospital_cohort_log(stats: PrehospitalCohortStats) -> str:
    lines = [
        "=== Prehospital EMS Cohort Filter ===",
        f"Records before filtering: {stats.records_before}",
        f"Excluded (non-EMS TRANSPORTMODE 4/5/6): {stats.excluded_non_ems_transport}",
        f"Excluded (INTERFACILITYTRANSFER == 1): {stats.excluded_interfacility_transfer}",
        f"Excluded overlap (both reasons): {stats.excluded_overlap}",
        f"Total excluded (unique): {stats.excluded_total}",
        f"Eligible cohort size: {stats.records_eligible}",
    ]
    return "\n".join(lines)


def log_prehospital_cohort_stats(stats: PrehospitalCohortStats, *, write_report: bool = True) -> None:
    message = format_prehospital_cohort_log(stats)
    print(message)
    if write_report:
        ensure_dirs()
        report_path = REPORTS_DIR / "prehospital_cohort_filter.txt"
        report_path.write_text(message + "\n", encoding="utf-8")
        print(f"Cohort filter summary saved to {report_path}")


def apply_prehospital_ems_cohort_filter_to_dataset(trauma_dataset, *, write_report: bool = True):
    """
    Apply the prehospital EMS cohort filter to TraumaDataset records in place.
    """
    if not trauma_dataset.get_records():
        stats = PrehospitalCohortStats(0, 0, 0, 0, 0, 0)
        trauma_dataset.cohort_state = stats.to_dict()
        log_prehospital_cohort_stats(stats, write_report=write_report)
        return trauma_dataset

    df = pd.DataFrame([record.data for record in trauma_dataset.get_records()])
    eligible_df, stats = apply_prehospital_ems_cohort_filter(df)

    eligible_indices = eligible_df.index.tolist()
    trauma_dataset.records = [trauma_dataset.records[i] for i in eligible_indices]
    trauma_dataset.cohort_state = stats.to_dict()
    log_prehospital_cohort_stats(stats, write_report=write_report)
    return trauma_dataset
