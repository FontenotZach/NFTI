from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.paths import REPORTS_DIR, ensure_dirs


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _expected_input_headers(trauma_dataset) -> List[str]:
    # Headers populated into record.data are governed by the load flag.
    return [
        header.name
        for header in trauma_dataset.get_headers()
        if header.data_type and header.load == "1"
    ]


def _expected_target_headers(trauma_dataset) -> List[str]:
    return [
        header.name
        for header in trauma_dataset.get_headers()
        if header.data_type and header.y == "1"
    ]


def _custom_header_names(trauma_dataset) -> List[str]:
    return [feature["header"] for feature in trauma_dataset.custom_features]


def _header_row(header) -> Dict[str, Any]:
    return {
        "name": header.name,
        "timing": header.timing,
        "data_type": header.data_type,
        "load": header.load,
        "usage": header.usage,
        "y": header.y,
        "is_custom": header.definition.startswith("Custom feature:"),
    }


def audit_trauma_dataset(trauma_dataset, *, sample_size: int = 10) -> Dict[str, Any]:
    """
    Audit a built TraumaDataset for header registration and record population.
    """
    records = trauma_dataset.get_records()
    headers = trauma_dataset.get_headers()
    input_headers = _expected_input_headers(trauma_dataset)
    target_headers = _expected_target_headers(trauma_dataset)
    custom_headers = set(_custom_header_names(trauma_dataset))

    input_set = set(input_headers)
    target_set = set(target_headers)
    record_count = len(records)

    header_stats: List[Dict[str, Any]] = []
    for header in headers:
        in_data = header.name in input_set
        in_y = header.name in target_set
        is_custom = header.name in custom_headers

        present_in_data = 0
        missing_in_data = 0
        zero_in_data = 0
        non_zero_in_data = 0
        nan_in_data = 0

        present_in_y = 0
        missing_in_y = 0

        for record in records:
            if in_data:
                if header.name not in record.data:
                    missing_in_data += 1
                else:
                    present_in_data += 1
                    value = record.data[header.name]
                    if _is_missing(value):
                        nan_in_data += 1
                    elif value == 0 or value == 0.0:
                        zero_in_data += 1
                    else:
                        non_zero_in_data += 1

            if in_y:
                if header.name not in record.y:
                    missing_in_y += 1
                else:
                    present_in_y += 1

        header_stats.append(
            {
                "header": header.name,
                "timing": header.timing,
                "data_type": header.data_type,
                "load": header.load,
                "usage": header.usage,
                "y": header.y,
                "is_custom": is_custom,
                "in_record_data": in_data,
                "in_record_y": in_y,
                "records_present": present_in_data if in_data else present_in_y,
                "records_missing": missing_in_data if in_data else missing_in_y,
                "data_population_rate": present_in_data / record_count if in_data and record_count else None,
                "y_population_rate": present_in_y / record_count if in_y and record_count else None,
                "population_rate": present_in_data / record_count if in_data and record_count else (
                    present_in_y / record_count if in_y and record_count else None
                ),
                "zero_count": zero_in_data if in_data else None,
                "non_zero_count": non_zero_in_data if in_data else None,
                "missing_or_nan_count": nan_in_data if in_data else None,
            }
        )

    records_missing_input_keys: List[int] = []
    records_with_extra_data_keys: List[int] = []
    for index, record in enumerate(records):
        data_keys = set(record.data.keys())
        missing_keys = input_set - data_keys
        extra_keys = data_keys - input_set
        if missing_keys:
            records_missing_input_keys.append(index)
        if extra_keys:
            records_with_extra_data_keys.append(index)

    custom_feature_stats: List[Dict[str, Any]] = []
    for feature_name in sorted(custom_headers):
        calculated = 0
        all_zero = 0
        has_non_zero = 0
        has_nan = 0
        for record in records:
            if feature_name not in record.data:
                continue
            calculated += 1
            value = record.data[feature_name]
            if _is_missing(value):
                has_nan += 1
            elif value == 0 or value == 0.0:
                all_zero += 1
            else:
                has_non_zero += 1

        custom_feature_stats.append(
            {
                "header": feature_name,
                "records_present": calculated,
                "all_zero_count": all_zero,
                "non_zero_count": has_non_zero,
                "nan_count": has_nan,
                "population_rate": calculated / record_count if record_count else None,
            }
        )

    model_ready_headers = [
        header.name
        for header in headers
        if header.usage == "1" and header.timing in ["1"] and header.data_type in ["1", "2", "3"]
    ]

    failing_input_headers = [
        row["header"]
        for row in header_stats
        if row["in_record_data"] and row["data_population_rate"] != 1.0
    ]
    failing_target_headers = [
        row["header"]
        for row in header_stats
        if row["in_record_y"] and row["y_population_rate"] != 1.0
    ]

    sample_records = records[:sample_size]
    sample_input_df = pd.DataFrame([record.data for record in sample_records])
    if input_headers:
        sample_input_df = sample_input_df.reindex(columns=input_headers)
    sample_target_df = pd.DataFrame([record.y for record in sample_records])
    if target_headers:
        sample_target_df = sample_target_df.reindex(columns=target_headers)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "record_count": record_count,
        "header_count": len(headers),
        "input_header_count": len(input_headers),
        "target_header_count": len(target_headers),
        "custom_feature_count": len(custom_headers),
        "model_ready_header_count": len(model_ready_headers),
        "header_stats": header_stats,
        "custom_feature_stats": custom_feature_stats,
        "records_missing_input_keys": records_missing_input_keys,
        "records_with_extra_data_keys": records_with_extra_data_keys,
        "failing_input_headers": failing_input_headers,
        "failing_target_headers": failing_target_headers,
        "model_ready_headers": model_ready_headers,
        "sample_input_df": sample_input_df,
        "sample_target_df": sample_target_df,
    }


def format_audit_summary(audit: Dict[str, Any]) -> str:
    lines = [
        "=== TraumaDataset Audit Summary ===",
        f"Generated: {audit['generated_at']}",
        f"Records: {audit['record_count']}",
        f"Registered headers: {audit['header_count']}",
        f"Input headers (Load=1): {audit['input_header_count']}",
        f"Target headers (Y=1): {audit['target_header_count']}",
        f"Custom features: {audit['custom_feature_count']}",
        f"Model-ready headers (Usage=1, Timing=1): {audit['model_ready_header_count']}",
        "",
    ]

    if audit["failing_input_headers"]:
        lines.append(
            "FAIL: input headers missing from one or more records "
            f"({len(audit['failing_input_headers'])}): "
            + ", ".join(audit["failing_input_headers"][:15])
        )
    else:
        lines.append("PASS: all input headers present in every record.data")

    if audit["failing_target_headers"]:
        lines.append(
            "FAIL: target headers missing from one or more records "
            f"({len(audit['failing_target_headers'])}): "
            + ", ".join(audit["failing_target_headers"][:15])
        )
    else:
        lines.append("PASS: all target headers present in every record.y")

    if audit["records_missing_input_keys"]:
        lines.append(
            "FAIL: records with missing input keys: "
            f"{len(audit['records_missing_input_keys'])}"
        )
    else:
        lines.append("PASS: no records missing expected input keys")

    if audit["custom_feature_stats"]:
        lines.append("")
        lines.append("Custom feature population:")
        for row in audit["custom_feature_stats"]:
            rate = row["population_rate"]
            rate_text = f"{rate * 100:.1f}%" if rate is not None else "n/a"
            lines.append(
                f"  - {row['header']}: present in {row['records_present']}/{audit['record_count']} "
                f"({rate_text}), non-zero={row['non_zero_count']}, zero={row['all_zero_count']}, "
                f"nan={row['nan_count']}"
            )

    custom_failures = [
        row["header"]
        for row in audit["custom_feature_stats"]
        if row["population_rate"] != 1.0
    ]
    if custom_failures:
        lines.append(
            "FAIL: custom features not present in all records: "
            + ", ".join(custom_failures)
        )
    elif audit["custom_feature_stats"]:
        lines.append("PASS: all custom features present in every record.data")

    lines.append("")
    lines.append("Under-populated input headers (>0% missing/NaN):")
    sparse_headers = [
        row
        for row in audit["header_stats"]
        if row["in_record_data"] and (row["missing_or_nan_count"] or 0) > 0
    ]
    if sparse_headers:
        for row in sparse_headers[:20]:
            lines.append(
                f"  - {row['header']}: missing/NaN in {row['missing_or_nan_count']} records"
            )
        if len(sparse_headers) > 20:
            lines.append(f"  ... (+{len(sparse_headers) - 20} more)")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def write_audit_report(
    trauma_dataset,
    *,
    output_dir: Optional[Path] = None,
    sample_size: int = 10,
) -> Dict[str, Path]:
    """
    Run the audit and write summary + CSV artifacts to the reports directory.
    """
    ensure_dirs()
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_trauma_dataset(trauma_dataset, sample_size=sample_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"dataset_audit_summary_{timestamp}.txt"
    headers_path = output_dir / f"dataset_audit_headers_{timestamp}.csv"
    custom_path = output_dir / f"dataset_audit_custom_features_{timestamp}.csv"
    sample_input_path = output_dir / f"dataset_audit_sample_input_{timestamp}.csv"
    sample_target_path = output_dir / f"dataset_audit_sample_targets_{timestamp}.csv"

    summary_path.write_text(format_audit_summary(audit), encoding="utf-8")
    pd.DataFrame(audit["header_stats"]).to_csv(headers_path, index=False)
    pd.DataFrame(audit["custom_feature_stats"]).to_csv(custom_path, index=False)

    if not audit["sample_input_df"].empty:
        audit["sample_input_df"].to_csv(sample_input_path, index=False)
    if not audit["sample_target_df"].empty:
        audit["sample_target_df"].to_csv(sample_target_path, index=False)

    return {
        "summary": summary_path,
        "headers": headers_path,
        "custom_features": custom_path,
        "sample_input": sample_input_path,
        "sample_targets": sample_target_path,
    }


def print_header_detail(trauma_dataset, header_name: str, *, sample_size: int = 5) -> None:
    """Print value distribution for a single header across records."""
    records = trauma_dataset.get_records()
    if not records:
        print("No records loaded.")
        return

    in_data = any(header_name in record.data for record in records)
    in_y = any(header_name in record.y for record in records)
    if not in_data and not in_y:
        print(f"Header '{header_name}' was not found in record.data or record.y.")
        return

    values: List[Any] = []
    for record in records:
        if in_data and header_name in record.data:
            values.append(record.data[header_name])
        elif in_y and header_name in record.y:
            values.append(record.y[header_name])

    series = pd.Series(values)
    print(f"\n--- Header detail: {header_name} ---")
    print(f"Location: {'record.data' if in_data else 'record.y'}")
    print(f"Records with value: {len(values)} / {len(records)}")
    print(f"Non-null count: {series.notna().sum()}")
    print(f"Unique values: {series.nunique(dropna=True)}")
    print("Value counts (top 10):")
    print(series.value_counts(dropna=False).head(10).to_string())
    print(f"\nFirst {sample_size} values:")
    print(series.head(sample_size).to_string(index=False))
