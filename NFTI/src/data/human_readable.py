from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

from src.paths import SCHEMAS_DIR

HUMAN_READABLE_PATH = SCHEMAS_DIR / "human_readable_headers.csv"
HEADER_COLUMN = "Header"
LABEL_COLUMN = "HumanReadable"


def load_human_readable_map(path: Path | None = None) -> Dict[str, str]:
    """Load the header -> human-readable label map. Missing file -> empty map."""
    path = path or HUMAN_READABLE_PATH
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            header = (row.get(HEADER_COLUMN) or "").strip()
            if header:
                mapping[header] = (row.get(LABEL_COLUMN) or "").strip()
    return mapping


def get_label(header: str, mapping: Dict[str, str] | None = None) -> str:
    """Return the human-readable label, falling back to the raw header name."""
    mapping = mapping if mapping is not None else load_human_readable_map()
    label = mapping.get(header)
    return label if label else header


def append_human_readable_entries(
    entries: Dict[str, str],
    path: Path | None = None,
) -> int:
    """
    Append entries for headers not already present in the file.

    Returns the number of new rows written.
    """
    path = path or HUMAN_READABLE_PATH
    existing = load_human_readable_map(path)
    new_items = {h: lbl for h, lbl in entries.items() if h and h not in existing}
    if not new_items:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    # Ensure the existing file ends with a newline before appending.
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
        if needs_newline:
            with open(path, "a", newline="", encoding="utf-8") as f:
                f.write("\n")

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([HEADER_COLUMN, LABEL_COLUMN])
        for header, label in new_items.items():
            writer.writerow([header, label])
    return len(new_items)


def humanize_level(level: str) -> str:
    """Turn a one-hot level token (e.g. '15.0', '__MISSING__') into readable text."""
    token = level.strip()
    if token.endswith(".0"):
        token = token[:-2]
    cleaned = token.strip("_")
    upper = cleaned.upper()
    if upper in ("MISSING",):
        return "Missing"
    if upper in ("UNMAPPED",):
        return "Unmapped"
    if upper in ("NA",):
        return "Not Applicable"
    if upper in ("UK", "UNKNOWN"):
        return "Unknown"
    return cleaned.replace("_", " ") if cleaned else token


def build_one_hot_label(
    feature_name: str,
    source_column: str,
    mapping: Dict[str, str] | None = None,
) -> str:
    """Best-guess label for a one-hot column: '<source label>: <level>'."""
    mapping = mapping if mapping is not None else load_human_readable_map()
    base = mapping.get(source_column) or source_column
    prefix = f"{source_column}_"
    level = feature_name[len(prefix):] if feature_name.startswith(prefix) else feature_name
    return f"{base}: {humanize_level(level)}"


def generate_one_hot_entries(
    one_hot_names: Iterable[str],
    source_for_name,
    mapping: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """
    Build {one_hot_column: label} for a set of one-hot columns.

    ``source_for_name`` is a callable mapping a one-hot column name to its
    source categorical column name.
    """
    mapping = mapping if mapping is not None else load_human_readable_map()
    entries: Dict[str, str] = {}
    for feature_name in one_hot_names:
        source_column = source_for_name(feature_name)
        entries[feature_name] = build_one_hot_label(feature_name, source_column, mapping)
    return entries


def find_untracked_headers(headers: Iterable[str], path: Path | None = None) -> List[str]:
    """Return headers (order-preserving, de-duplicated) absent from the label file."""
    mapping = load_human_readable_map(path)
    seen = set()
    untracked: List[str] = []
    for header in headers:
        if header in seen or header in mapping:
            continue
        seen.add(header)
        untracked.append(header)
    return untracked


def find_blank_label_headers(path: Path | None = None) -> List[str]:
    """Return headers present in the file whose human-readable label is empty.

    Order matches the file. These are typically placeholder rows (e.g. one-hot
    feature names registered with a blank label) awaiting a name.
    """
    path = path or HUMAN_READABLE_PATH
    if not path.exists():
        return []
    blanks: List[str] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            header = (row.get(HEADER_COLUMN) or "").strip()
            label = (row.get(LABEL_COLUMN) or "").strip()
            if header and not label:
                blanks.append(header)
    return blanks


def set_human_readable_labels(
    entries: Dict[str, str],
    path: Path | None = None,
) -> int:
    """Upsert labels: update existing rows (incl. filling blanks) and append new.

    Unlike :func:`append_human_readable_entries`, this can overwrite an existing
    (possibly blank) label. Returns the number of rows changed or added. Empty
    label values in ``entries`` are ignored so blanks are never re-blanked.
    """
    path = path or HUMAN_READABLE_PATH
    updates = {h: lbl for h, lbl in entries.items() if h and lbl}
    if not updates:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    if path.exists():
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                header = (row.get(HEADER_COLUMN) or "").strip()
                label = (row.get(LABEL_COLUMN) or "").strip()
                rows.append({HEADER_COLUMN: header, LABEL_COLUMN: label})

    existing_headers = {row[HEADER_COLUMN] for row in rows}
    changed = 0
    for row in rows:
        header = row[HEADER_COLUMN]
        if header in updates and row[LABEL_COLUMN] != updates[header]:
            row[LABEL_COLUMN] = updates[header]
            changed += 1
    for header, label in updates.items():
        if header not in existing_headers:
            rows.append({HEADER_COLUMN: header, LABEL_COLUMN: label})
            changed += 1

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([HEADER_COLUMN, LABEL_COLUMN])
        for row in rows:
            writer.writerow([row[HEADER_COLUMN], row[LABEL_COLUMN]])
    return changed
