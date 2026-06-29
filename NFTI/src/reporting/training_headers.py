"""Definitive audit of the headers used as model training inputs.

The model feature selection rule lives in
``src.preprocessing.feature_preprocessor._get_feature_column_groups`` and selects
a header as a training input when ``usage == "1"``, ``timing == "1"`` and
``data_type`` is one of binary/categorical/continuous ("1"/"2"/"3"). Categorical
headers are one-hot expanded into the model design matrix at runtime.

This module re-uses that exact rule so the report can never drift from what the
pipeline actually trains on. It can run against a built :class:`TraumaDataset`
(definitive for a specific run) or, with no dataset, directly from the schema
(``header_definitions.csv`` + ``customs.csv``) for a reproducible, version-able
audit list.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.TraumaDataset import TraumaDataset
from src.data.human_readable import load_human_readable_map
from src.paths import (
    SCHEMAS_DIR,
    REPORTS_DIR,
    TRAINING_HEADERS_REPORT_PATH,
    TRAINING_HEADERS_SUMMARY_PATH,
    ensure_dirs,
)
from src.preprocessing.feature_preprocessor import _get_feature_column_groups


_DATA_TYPE_LABELS = {"1": "binary", "2": "categorical", "3": "continuous"}

_REPORT_FIELDNAMES = [
    "order",
    "header",
    "human_readable",
    "role",
    "data_type",
    "timing",
    "load",
    "usage",
    "is_custom",
    "one_hot_expanded",
    "definition",
]


def build_schema_dataset(
    schema_path: Optional[Path] = None,
    customs_path: Optional[Path] = None,
) -> TraumaDataset:
    """Register every schema header (and custom feature) without loading records.

    This mirrors how ``app.py`` populates headers, but for the full schema rather
    than only the columns present in a particular data CSV, giving a complete,
    data-independent view of the intended training configuration.
    """
    schema_path = Path(schema_path) if schema_path else SCHEMAS_DIR / "header_definitions.csv"
    customs_path = Path(customs_path) if customs_path else SCHEMAS_DIR / "customs.csv"

    dataset = TraumaDataset()
    with open(schema_path, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            dataset.add_header(
                row.get("Header", ""),
                ntds_page=row.get("NTDS_Page", ""),
                definition=row.get("Definition", ""),
                timing=row.get("Timing", ""),
                data_type=row.get("Type", ""),
                load=row.get("Load", ""),
                usage=row.get("Usage", ""),
                y=row.get("Y", ""),
            )

    if customs_path.exists():
        dataset.add_custom_features(str(customs_path))

    return dataset


def _is_custom(header) -> bool:
    return bool(header.definition) and header.definition.startswith("Custom feature:")


def collect_training_headers(trauma_dataset: TraumaDataset) -> List[Dict[str, Any]]:
    """Return the definitive, ordered list of headers used as model inputs.

    Order matches the model design-matrix assembly: binary, then categorical
    (one-hot expanded downstream), then continuous.
    """
    binary_cols, categorical_cols, continuous_cols = _get_feature_column_groups(trauma_dataset)
    role_by_name: Dict[str, str] = {}
    for name in binary_cols:
        role_by_name[name] = "binary"
    for name in categorical_cols:
        role_by_name[name] = "categorical"
    for name in continuous_cols:
        role_by_name[name] = "continuous"

    ordered_names = list(binary_cols) + list(categorical_cols) + list(continuous_cols)
    header_by_name = {header.name: header for header in trauma_dataset.get_headers()}
    hr_map = load_human_readable_map()

    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(ordered_names, start=1):
        header = header_by_name.get(name)
        role = role_by_name.get(name, "")
        rows.append(
            {
                "order": index,
                "header": name,
                "human_readable": hr_map.get(name, name),
                "role": role,
                "data_type": header.data_type if header else "",
                "timing": header.timing if header else "",
                "load": header.load if header else "",
                "usage": header.usage if header else "",
                "is_custom": "yes" if header and _is_custom(header) else "no",
                "one_hot_expanded": "yes" if role == "categorical" else "no",
                "definition": header.definition if header else "",
            }
        )
    return rows


def _diagnostic_usage_excluded(trauma_dataset: TraumaDataset, training_names) -> List[Dict[str, str]]:
    """Headers flagged usage=1 that are NOT used in training, with the reason."""
    training_set = set(training_names)
    excluded: List[Dict[str, str]] = []
    for header in trauma_dataset.get_headers():
        if header.usage != "1" or header.name in training_set:
            continue
        if header.data_type not in _DATA_TYPE_LABELS:
            reason = f"data_type '{header.data_type}' is not a model feature type"
        elif header.timing != "1":
            reason = f"timing '{header.timing}' is excluded (model uses timing=1 only)"
        else:
            reason = "not selected"
        excluded.append({"header": header.name, "timing": header.timing,
                         "data_type": header.data_type, "reason": reason})
    return excluded


def _diagnostic_loaded_not_trained(trauma_dataset: TraumaDataset) -> List[Dict[str, str]]:
    """Headers loaded into memory (load=1) but not used as training inputs (usage=0)."""
    loaded_only: List[Dict[str, str]] = []
    for header in trauma_dataset.get_headers():
        if header.load == "1" and header.usage != "1":
            loaded_only.append({"header": header.name, "timing": header.timing,
                               "data_type": header.data_type})
    return loaded_only


def write_training_headers_report(
    trauma_dataset: Optional[TraumaDataset] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write the definitive training-headers audit (CSV + text summary).

    When ``trauma_dataset`` is ``None`` the report is generated from the schema,
    yielding a reproducible, data-independent audit of the intended configuration.
    """
    ensure_dirs()
    if trauma_dataset is None:
        trauma_dataset = build_schema_dataset()
        source = "schema (data/schemas/header_definitions.csv + customs.csv)"
    else:
        source = "built TraumaDataset"

    rows = collect_training_headers(trauma_dataset)
    training_names = [row["header"] for row in rows]

    output_dir = Path(output_dir) if output_dir else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        output_dir / TRAINING_HEADERS_REPORT_PATH.name
        if output_dir != REPORTS_DIR
        else TRAINING_HEADERS_REPORT_PATH
    )
    summary_path = (
        output_dir / TRAINING_HEADERS_SUMMARY_PATH.name
        if output_dir != REPORTS_DIR
        else TRAINING_HEADERS_SUMMARY_PATH
    )

    with open(csv_path, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=_REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    role_counts = {"binary": 0, "categorical": 0, "continuous": 0}
    custom_count = 0
    for row in rows:
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
        if row["is_custom"] == "yes":
            custom_count += 1

    usage_excluded = _diagnostic_usage_excluded(trauma_dataset, training_names)
    loaded_not_trained = _diagnostic_loaded_not_trained(trauma_dataset)

    lines: List[str] = [
        "=== Training Headers Audit ===",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {source}",
        "",
        "Selection rule (model input): usage == 1 AND timing == 1 AND "
        "data_type in {1=binary, 2=categorical, 3=continuous}.",
        "Categorical headers are one-hot expanded into the model design matrix at runtime.",
        "Definition of flags: load=1 -> populated into memory; usage=1 -> eligible model feature.",
        "",
        f"Total headers used in training: {len(rows)}",
        f"  binary:      {role_counts.get('binary', 0)}",
        f"  categorical: {role_counts.get('categorical', 0)} (one-hot expanded downstream)",
        f"  continuous:  {role_counts.get('continuous', 0)}",
        f"  custom features among the above: {custom_count}",
        "",
        f"Full list: {csv_path}",
        "",
        "Headers used in training:",
    ]
    for row in rows:
        custom_tag = " [custom]" if row["is_custom"] == "yes" else ""
        lines.append(f"  {row['order']:>3}. {row['header']} ({row['role']}){custom_tag}")

    lines.append("")
    lines.append(
        "Diagnostics -- headers flagged usage=1 but NOT used in training "
        f"({len(usage_excluded)}):"
    )
    if usage_excluded:
        for item in usage_excluded:
            lines.append(f"  - {item['header']}: {item['reason']}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(
        "Diagnostics -- headers loaded into memory but not used in training "
        f"(load=1, usage=0) ({len(loaded_not_trained)}):"
    )
    if loaded_not_trained:
        for item in loaded_not_trained:
            lines.append(f"  - {item['header']}")
    else:
        lines.append("  (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"csv": csv_path, "summary": summary_path}


def main() -> None:
    paths = write_training_headers_report()
    print("Training headers audit written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
