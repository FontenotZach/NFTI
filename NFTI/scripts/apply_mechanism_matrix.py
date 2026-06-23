#!/usr/bin/env python3
"""
One-off utility: relabel MECHANISM and TRAUMATYPE from ACS external-cause ICD codes.

Usage:
    python scripts/apply_mechanism_matrix.py path/to/data_file.csv
    python scripts/apply_mechanism_matrix.py path/to/cohort.parquet
    python scripts/apply_mechanism_matrix.py   # prompts for input path

Outputs (alongside the input file):
    {stem}_matrix_processed.{ext}        processed dataset
    {stem}_matrix_processed_report.txt   human-readable label/ICD breakdown
    {stem}_matrix_processed_report.csv   tabular label/ICD breakdown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = APP_ROOT / "data" / "schemas" / "mechanism_matrix.csv"

# ---------------------------------------------------------------------------
# Manual override: set column names here if automatic detection is imperfect.
# Leave empty to rely on known TQIP columns and heuristics below.
# Example:
# MANUAL_EXTERNAL_CAUSE_COLUMNS = ["PRIMARYECODEICD10", "ADDITIONALECODE1"]
# ---------------------------------------------------------------------------
MANUAL_EXTERNAL_CAUSE_COLUMNS: List[str] = []

KNOWN_TQIP_EXTERNAL_CAUSE_COLUMNS: Tuple[str, ...] = (
    "PRIMARYECODEICD10",
    "ADDITIONALECODE1",
    "ADDITIONALECODE2",
)

MISSING_LABEL = "__MISSING__"
UNMAPPED_LABEL = "__UNMAPPED__"

MISSING_ICD_TOKENS = frozenset({"", "NAN", "NONE", "NULL", "<NA>", "NAT"})

MATRIX_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "icd_code": ("ICDDIAGNOSISCODE", "ICD_DIAGNOSIS_CODE", "ICDCODE", "ICD_CODE"),
    "mechanism": ("MECHANISM",),
    "intent": ("INTENT",),
    "trauma_type": ("TRAUMATYPE", "TRAUMA_TYPE"),
    "hierarchy": ("HIERARCHY", "HIERARCHY_RANK", "PRIORITY"),
}


def normalize_icd_code(code: Any) -> Optional[str]:
    """Normalize an ICD-10 code for lookup; return None when missing/invalid."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    text = str(code).strip().upper()
    if text in MISSING_ICD_TOKENS:
        return None
    text = text.replace(".", "")
    if text in MISSING_ICD_TOKENS:
        return None
    return text or None


def _normalize_column_key(name: str) -> str:
    return name.strip().upper().replace(" ", "").replace("_", "")


def _resolve_matrix_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {_normalize_column_key(col): col for col in columns}
    for alias in aliases:
        match = lookup.get(_normalize_column_key(alias))
        if match is not None:
            return match
    return None


def build_output_path(input_path: Path) -> Path:
    """Append `_matrix_processed` before the file extension."""
    return input_path.with_name(f"{input_path.stem}_matrix_processed{input_path.suffix}")


def build_output_report_path(output_path: Path) -> Path:
    """Report path alongside the processed dataset output."""
    return output_path.with_name(f"{output_path.stem}_report.txt")


def build_output_report_csv_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_report.csv")


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires pyarrow or fastparquet. "
                "Install one of those packages, or provide a CSV file."
            ) from exc
    raise ValueError(f"Unsupported dataset format '{suffix}'. Use CSV or Parquet.")


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires pyarrow or fastparquet. "
                "Install one of those packages, or use a CSV input file."
            ) from exc
        return
    raise ValueError(f"Unsupported output format '{suffix}'. Use CSV or Parquet.")


def detect_external_cause_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify dataset columns that hold external-cause ICD-10 codes.

    Preference order:
    1. MANUAL_EXTERNAL_CAUSE_COLUMNS (if any exist in the dataset)
    2. Known TQIP external-cause columns
    3. Heuristic name matching (ECODE / external cause), excluding BIU and place-of-injury fields
    """
    if MANUAL_EXTERNAL_CAUSE_COLUMNS:
        manual = [col for col in MANUAL_EXTERNAL_CAUSE_COLUMNS if col in df.columns]
        if manual:
            return manual
        print(
            "Warning: MANUAL_EXTERNAL_CAUSE_COLUMNS is set but none of those columns "
            f"exist in the dataset. Falling back to automatic detection."
        )

    known = [col for col in KNOWN_TQIP_EXTERNAL_CAUSE_COLUMNS if col in df.columns]
    if known:
        return known

    heuristic: List[str] = []
    for col in df.columns:
        key = _normalize_column_key(col)
        if key.endswith("BIU"):
            continue
        if "PLACEOFINJURY" in key:
            continue
        if "ECODE" in key or "EXTERNALCAUSE" in key:
            heuristic.append(col)

    # Preserve a sensible primary-first ordering when possible.
    def _sort_key(name: str) -> Tuple[int, str]:
        upper = name.upper()
        if "PRIMARY" in upper:
            return (0, upper)
        if "ADDITIONAL" in upper:
            return (1, upper)
        return (2, upper)

    return sorted(heuristic, key=_sort_key)


def load_mechanism_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mechanism matrix not found: {path}")

    matrix_df = pd.read_csv(path, low_memory=False)
    resolved: Dict[str, str] = {}
    for logical_name, aliases in MATRIX_COLUMN_ALIASES.items():
        column = _resolve_matrix_column(matrix_df.columns, aliases)
        if column is None:
            raise ValueError(
                f"Mechanism matrix is missing required column for '{logical_name}'. "
                f"Expected one of: {', '.join(aliases)}"
            )
        resolved[logical_name] = column

    normalized = matrix_df.rename(
        columns={
            resolved["icd_code"]: "ICD_CODE",
            resolved["mechanism"]: "MECHANISM",
            resolved["intent"]: "INTENT",
            resolved["trauma_type"]: "TRAUMATYPE",
            resolved["hierarchy"]: "HIERARCHY",
        }
    )

    normalized["ICD_CODE_NORM"] = normalized["ICD_CODE"].map(normalize_icd_code)
    normalized = normalized[normalized["ICD_CODE_NORM"].notna()].copy()

    # ACS HIERARCHY: lower numeric values indicate higher priority (more specific).
    # When HIERARCHY is blank/non-numeric, treat as lowest priority so explicit ranks win.
    hierarchy_numeric = pd.to_numeric(normalized["HIERARCHY"], errors="coerce")
    normalized["HIERARCHY_NUM"] = hierarchy_numeric.fillna(float("inf"))

    for col in ("MECHANISM", "INTENT", "TRAUMATYPE"):
        normalized[col] = (
            normalized[col]
            .where(normalized[col].notna(), other=pd.NA)
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    normalized = normalized[
        normalized["MECHANISM"].notna() & normalized["TRAUMATYPE"].notna()
    ].copy()

    return normalized


def build_matrix_lookup(matrix_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Map normalized ICD code -> best matrix row.

    When the matrix contains duplicate ICD codes, keep the row with the lowest
    HIERARCHY number (highest ACS priority).
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    sorted_rows = matrix_df.sort_values(["ICD_CODE_NORM", "HIERARCHY_NUM"], kind="mergesort")

    for _, row in sorted_rows.iterrows():
        code = row["ICD_CODE_NORM"]
        if code not in lookup:
            lookup[code] = {
                "mechanism": row["MECHANISM"],
                "intent": row["INTENT"],
                "trauma_type": row["TRAUMATYPE"],
                "hierarchy_num": row["HIERARCHY_NUM"],
                "hierarchy_raw": row["HIERARCHY"],
                "source_icd": row["ICD_CODE"],
            }
    return lookup


def derive_mechanism_for_record(
    row: pd.Series,
    external_cause_cols: Sequence[str],
    matrix_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Derive mechanism labels for one record from its external-cause ICD columns.

    Priority when multiple mapped codes are present:
    1. Lowest HIERARCHY number (ACS convention: smaller = higher priority)
    2. Earlier external-cause column (primary before additional codes)
    """
    candidates: List[Tuple[float, int, str, Dict[str, Any]]] = []
    raw_codes_seen: List[str] = []

    for col_index, col in enumerate(external_cause_cols):
        normalized = normalize_icd_code(row.get(col))
        if normalized is None:
            continue
        raw_codes_seen.append(normalized)

        mapping = matrix_lookup.get(normalized)
        if mapping is None:
            continue

        candidates.append(
            (
                float(mapping["hierarchy_num"]),
                col_index,
                normalized,
                mapping,
            )
        )

    if not raw_codes_seen:
        return {
            "MECHANISM": MISSING_LABEL,
            "TRAUMATYPE": MISSING_LABEL,
            "INTENT": MISSING_LABEL,
            "MECHANISM_MAPPING_STATUS": "missing_external_cause",
            "TRAUMATYPE_MAPPING_STATUS": "missing_external_cause",
            "MECHANISM_SOURCE_ICD": "",
            "MECHANISM_SOURCE_HIERARCHY": "",
        }

    if not candidates:
        return {
            "MECHANISM": UNMAPPED_LABEL,
            "TRAUMATYPE": UNMAPPED_LABEL,
            "INTENT": UNMAPPED_LABEL,
            "MECHANISM_MAPPING_STATUS": "unmapped_external_cause",
            "TRAUMATYPE_MAPPING_STATUS": "unmapped_external_cause",
            "MECHANISM_SOURCE_ICD": "",
            "MECHANISM_SOURCE_HIERARCHY": "",
        }

    _hierarchy, _col_index, winning_code, winning_map = min(candidates, key=lambda item: (item[0], item[1]))
    hierarchy_display = winning_map["hierarchy_raw"]
    if pd.isna(hierarchy_display):
        hierarchy_display = ""

    return {
        "MECHANISM": winning_map["mechanism"],
        "TRAUMATYPE": winning_map["trauma_type"],
        "INTENT": winning_map["intent"],
        "MECHANISM_MAPPING_STATUS": "mapped",
        "TRAUMATYPE_MAPPING_STATUS": "mapped",
        "MECHANISM_SOURCE_ICD": winning_code,
        "MECHANISM_SOURCE_HIERARCHY": str(hierarchy_display).strip(),
    }


def apply_mechanism_mapping(
    df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    external_cause_cols: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    matrix_lookup = build_matrix_lookup(matrix_df)
    output_df = df.copy()

    derived_rows = [
        derive_mechanism_for_record(row, external_cause_cols, matrix_lookup)
        for _, row in output_df.iterrows()
    ]
    derived_df = pd.DataFrame(derived_rows, index=output_df.index)

    label_columns = [
        "MECHANISM",
        "TRAUMATYPE",
        "INTENT",
        "MECHANISM_MAPPING_STATUS",
        "TRAUMATYPE_MAPPING_STATUS",
        "MECHANISM_SOURCE_ICD",
        "MECHANISM_SOURCE_HIERARCHY",
    ]
    for col in label_columns:
        output_df[col] = derived_df[col].astype(str)

    status = derived_df["MECHANISM_MAPPING_STATUS"]
    stats = {
        "mapped": int((status == "mapped").sum()),
        "missing_external_cause": int((status == "missing_external_cause").sum()),
        "unmapped_external_cause": int((status == "unmapped_external_cause").sum()),
        "unmapped_icd_counts": _collect_unmapped_icd_counts(output_df, external_cause_cols, matrix_lookup),
    }
    return output_df, stats


def _collect_unmapped_icd_counts(
    df: pd.DataFrame,
    external_cause_cols: Sequence[str],
    matrix_lookup: Dict[str, Dict[str, Any]],
) -> pd.Series:
    unmapped_only = df["MECHANISM_MAPPING_STATUS"] == "unmapped_external_cause"
    if not unmapped_only.any():
        return pd.Series(dtype="int64")

    counts: Dict[str, int] = {}
    subset = df.loc[unmapped_only, external_cause_cols]
    for _, row in subset.iterrows():
        for col in external_cause_cols:
            code = normalize_icd_code(row.get(col))
            if code is None:
                continue
            if code in matrix_lookup:
                continue
            counts[code] = counts.get(code, 0) + 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


def _source_icds_for_record(
    row: pd.Series,
    external_cause_cols: Sequence[str],
) -> List[str]:
    """
    ICD codes to attribute to a record in label breakdowns.

    Mapped records use MECHANISM_SOURCE_ICD (the hierarchy-winning code).
    Missing/unmapped records fall back to all present external-cause codes.
    """
    winning = normalize_icd_code(row.get("MECHANISM_SOURCE_ICD"))
    if winning:
        return [winning]

    codes: List[str] = []
    for col in external_cause_cols:
        code = normalize_icd_code(row.get(col))
        if code and code not in codes:
            codes.append(code)
    return codes


def build_label_icd_breakdown(
    df: pd.DataFrame,
    label_col: str,
    external_cause_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build ICD code counts grouped by a label column (MECHANISM or TRAUMATYPE).

    Returns:
        breakdown_df with columns [label_col, ICD_CODE, count, label_total]
        label_totals indexed by label value
    """
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        label = row[label_col]
        icd_codes = _source_icds_for_record(row, external_cause_cols)
        if not icd_codes:
            rows.append({label_col: label, "ICD_CODE": "(none)"})
            continue
        for code in icd_codes:
            rows.append({label_col: label, "ICD_CODE": code})

    if not rows:
        empty = pd.DataFrame(columns=[label_col, "ICD_CODE", "count", "label_total"])
        return empty, pd.Series(dtype="int64")

    long_df = pd.DataFrame(rows)
    breakdown = (
        long_df.groupby([label_col, "ICD_CODE"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([label_col, "count", "ICD_CODE"], ascending=[True, False, True])
    )
    label_totals = df[label_col].value_counts(dropna=False)
    breakdown["label_total"] = breakdown[label_col].map(label_totals)
    breakdown = breakdown.sort_values(
        ["label_total", label_col, "count", "ICD_CODE"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return breakdown, label_totals


def format_label_breakdown_section(
    title: str,
    breakdown_df: pd.DataFrame,
    label_totals: pd.Series,
) -> str:
    lines = [title, "=" * len(title), ""]
    if breakdown_df.empty:
        lines.append("(no records)")
        return "\n".join(lines)

    for label, total in label_totals.sort_values(ascending=False).items():
        lines.append(f"{label}  (total: {int(total):,})")
        label_rows = breakdown_df[breakdown_df["label_value"] == label]
        for _, icd_row in label_rows.iterrows():
            lines.append(f"  {icd_row['ICD_CODE']}: {int(icd_row['count']):,}")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_mapping_report(
    output_df: pd.DataFrame,
    external_cause_cols: Sequence[str],
) -> Tuple[str, pd.DataFrame]:
    mechanism_breakdown, mechanism_totals = build_label_icd_breakdown(
        output_df, "MECHANISM", external_cause_cols
    )
    trauma_breakdown, trauma_totals = build_label_icd_breakdown(
        output_df, "TRAUMATYPE", external_cause_cols
    )

    mechanism_breakdown = mechanism_breakdown.rename(columns={"MECHANISM": "label_value"})
    trauma_breakdown = trauma_breakdown.rename(columns={"TRAUMATYPE": "label_value"})
    mechanism_breakdown.insert(0, "label_type", "MECHANISM")
    trauma_breakdown.insert(0, "label_type", "TRAUMATYPE")

    report_csv = pd.concat([mechanism_breakdown, trauma_breakdown], ignore_index=True)

    sections = [
        format_label_breakdown_section("MECHANISM breakdown", mechanism_breakdown, mechanism_totals),
        "",
        format_label_breakdown_section("TRAUMATYPE breakdown", trauma_breakdown, trauma_totals),
    ]
    return "\n".join(sections), report_csv


def write_mapping_report(
    *,
    input_path: Path,
    output_path: Path,
    output_df: pd.DataFrame,
    external_cause_cols: Sequence[str],
    stats: Dict[str, Any],
    record_count: int,
) -> Tuple[Path, Path]:
    report_path = build_output_report_path(output_path)
    report_csv_path = build_output_report_csv_path(output_path)

    body, report_csv = build_mapping_report(output_df, external_cause_cols)
    header_lines = [
        "Mechanism Matrix Mapping Report",
        "================================",
        f"Input file:  {input_path}",
        f"Output file: {output_path}",
        f"Records loaded: {record_count:,}",
        f"Mapped successfully: {stats['mapped']:,}",
        f"Missing external cause codes: {stats['missing_external_cause']:,}",
        f"External cause present but unmapped: {stats['unmapped_external_cause']:,}",
        f"External cause columns used: {', '.join(external_cause_cols) or '(none)'}",
        "",
    ]
    report_path.write_text("\n".join(header_lines) + "\n" + body + "\n", encoding="utf-8")
    report_csv.to_csv(report_csv_path, index=False)
    return report_path, report_csv_path


def validate_output(output_path: Path, df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    if not output_path.exists():
        raise RuntimeError(f"Expected output file was not created: {output_path}")

    for col in ("MECHANISM", "TRAUMATYPE"):
        if col not in df.columns:
            raise RuntimeError(f"Output dataset is missing required column: {col}")
        if not pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            print(f"Warning: {col} is not stored as string/object dtype ({df[col].dtype}).")

    total = len(df)
    if total and stats["mapped"] == 0:
        print("Warning: no records were mapped successfully.")
    if total and stats["mapped"] + stats["missing_external_cause"] + stats["unmapped_external_cause"] != total:
        print("Warning: mapping status counts do not sum to total record count.")


def print_report(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    report_csv_path: Path,
    record_count: int,
    stats: Dict[str, Any],
    external_cause_cols: Sequence[str],
    output_df: pd.DataFrame,
) -> None:
    print("\n=== Mechanism Matrix Relabel Report ===")
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Report file: {report_path}")
    print(f"Report CSV:  {report_csv_path}")
    print(f"Records loaded: {record_count:,}")
    print(f"Mapped successfully: {stats['mapped']:,}")
    print(f"Missing external cause codes: {stats['missing_external_cause']:,}")
    print(f"External cause present but unmapped: {stats['unmapped_external_cause']:,}")
    print(f"External cause columns used: {', '.join(external_cause_cols) or '(none)'}")

    print("\nMECHANISM value counts:")
    print(output_df["MECHANISM"].value_counts(dropna=False).to_string())

    print("\nTRAUMATYPE value counts:")
    print(output_df["TRAUMATYPE"].value_counts(dropna=False).to_string())

    unmapped_counts: pd.Series = stats["unmapped_icd_counts"]
    if not unmapped_counts.empty:
        print("\nTop unmapped ICD codes:")
        print(unmapped_counts.head(20).to_string())
    else:
        print("\nTop unmapped ICD codes: (none)")

    print(f"\nDetailed label/ICD breakdown written to:\n  {report_path}\n  {report_csv_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel MECHANISM and TRAUMATYPE using the ACS mechanism matrix."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Path to input CSV or Parquet dataset.",
    )
    parser.add_argument(
        "--matrix-path",
        default=str(DEFAULT_MATRIX_PATH),
        help=f"Path to ACS mechanism matrix CSV (default: {DEFAULT_MATRIX_PATH}).",
    )
    return parser.parse_args(argv)


def resolve_dataset_path(raw_path: Optional[str]) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()

    entered = input("Enter path to dataset file (CSV or Parquet): ").strip().strip('"').strip("'")
    if not entered:
        raise ValueError("No dataset path provided.")
    return Path(entered).expanduser().resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = resolve_dataset_path(args.dataset_path)
    matrix_path = Path(args.matrix_path).expanduser().resolve()
    output_path = build_output_path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    print(f"Loading dataset: {input_path}")
    df = load_dataset(input_path)
    record_count = len(df)
    print(f"Loaded {record_count:,} records.")

    external_cause_cols = detect_external_cause_columns(df)
    if not external_cause_cols:
        print(
            "Warning: no external-cause ICD columns were detected. "
            "All records will be labeled as missing."
        )

    print(f"Loading mechanism matrix: {matrix_path}")
    matrix_df = load_mechanism_matrix(matrix_path)
    print(f"Loaded {len(matrix_df):,} matrix rows with usable ICD codes.")

    output_df, stats = apply_mechanism_mapping(df, matrix_df, external_cause_cols)

    print(f"Writing output: {output_path}")
    save_dataset(output_df, output_path)

    validate_output(output_path, output_df, stats)

    report_path, report_csv_path = write_mapping_report(
        input_path=input_path,
        output_path=output_path,
        output_df=output_df,
        external_cause_cols=external_cause_cols,
        stats=stats,
        record_count=record_count,
    )

    print_report(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        report_csv_path=report_csv_path,
        record_count=record_count,
        stats=stats,
        external_cause_cols=external_cause_cols,
        output_df=output_df,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
