#!/usr/bin/env python3
"""Missing-data audit (CLI entrypoint).

Expands the project's missingness analysis into a robust missing-data audit for
the trauma-triage (NFTI) ML manuscript. Implements five analyses and writes
manuscript/supplement-ready tables, metrics, and figures:

    1. Missingness by feature group (not just individual headers)
    2. Missingness pattern analysis (which variables go missing together)
    3. Missingness by clinical context
    4. Missingness as a prediction target (is it systematic?)
    5. Model performance stratified by EMS missingness burden

It never retrains the primary model and never overwrites model outputs. By
default it reads the RAW cohort CSV (data/raw/dat5.csv) and re-applies the
prehospital EMS cohort filter so record IDs align with saved holdout
predictions, computing missingness BEFORE imputation/scaling/one-hot encoding.

Usage:
    python scripts/missingness_audit.py
    python scripts/missingness_audit.py --input data/raw/dat5.csv
    python scripts/missingness_audit.py --input artifacts/pickles/datasets/trauma_dataset.pkl
    python scripts/missingness_audit.py --max-records 20000   # quick smoke test

Outputs:
    artifacts/figures/missingness/*.png
    artifacts/tables/missingness/*.csv
    artifacts/metrics/missingness/*
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Make the app root importable so pickled `src.*` classes resolve when run
# from the project root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.evaluation.missingness_audit import run_missingness_audit  # noqa: E402
from src.paths import (  # noqa: E402
    MISSINGNESS_FIGURES_DIR,
    MISSINGNESS_METRICS_DIR,
    MISSINGNESS_TABLES_DIR,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a robust missing-data audit and generate figures/tables/metrics."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to the raw cohort CSV or a pre-transform TraumaDataset pickle. "
            "Default: data/raw/dat5.csv (falls back to data/samples/dat5_limited.csv)."
        ),
    )
    parser.add_argument("--figures-dir", default=str(MISSINGNESS_FIGURES_DIR))
    parser.add_argument("--tables-dir", default=str(MISSINGNESS_TABLES_DIR))
    parser.add_argument("--metrics-dir", default=str(MISSINGNESS_METRICS_DIR))
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            "Limit the number of records loaded (for a quick smoke test). "
            "NOTE: subsampling breaks record-ID alignment, so Analysis 5 "
            "(model performance by missingness burden) will be skipped."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser() if args.input else None
    if input_path is not None and not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    run_missingness_audit(
        input_path,
        figures_dir=Path(args.figures_dir),
        tables_dir=Path(args.tables_dir),
        metrics_dir=Path(args.metrics_dir),
        max_records=args.max_records,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
