#!/usr/bin/env python3
"""Prehospital vital-sign fidelity audit (CLI entrypoint).

Audits EMS / pre-hospital vital-sign quality against matched ED/hospital
arrival vitals and writes publication-quality figures and summary tables.

Run BEFORE imputation / one-hot / normalization so figures reflect documented
(raw) values. By default it audits the pre-transform dataset pickle produced by
``app.py`` option 1 (Pickle the data), which keeps raw documented vitals.

Usage:
    python scripts/fidelity_audit.py
    python scripts/fidelity_audit.py --input artifacts/pickles/datasets/trauma_dataset.pkl
    python scripts/fidelity_audit.py --row-flags

Outputs:
    artifacts/figures/fidelity/*.png
    artifacts/tables/fidelity/*.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional, Sequence

# Make the app root importable so pickled `src.*` classes resolve when run
# from the project root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.evaluation.fidelity_audit import run_prehospital_vital_fidelity_audit  # noqa: E402
from src.paths import (  # noqa: E402
    FIDELITY_FIGURES_DIR,
    FIDELITY_TABLES_DIR,
    PICKLES_DIR,
)

DEFAULT_INPUT = PICKLES_DIR / "datasets" / "trauma_dataset.pkl"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit prehospital EMS vital-sign fidelity and generate figures/tables."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Path to a pickled TraumaDataset. Default: the pre-transform dataset "
            f"({DEFAULT_INPUT})."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        default=str(FIDELITY_FIGURES_DIR),
        help=f"Directory for figures (default: {FIDELITY_FIGURES_DIR}).",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(FIDELITY_TABLES_DIR),
        help=f"Directory for summary tables (default: {FIDELITY_TABLES_DIR}).",
    )
    parser.add_argument(
        "--row-flags",
        action="store_true",
        help="Also write row-level plausibility flags (can be a large file).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()

    if not input_path.exists():
        print(
            f"Error: input pickle not found: {input_path}\n"
            "Run app.py option 1 (Pickle the data) first, or pass --input.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading TraumaDataset from {input_path} ...")
    with open(input_path, "rb") as handle:
        trauma_dataset = pickle.load(handle)

    run_prehospital_vital_fidelity_audit(
        trauma_dataset,
        figures_dir=Path(args.figures_dir),
        tables_dir=Path(args.tables_dir),
        save_row_flags=args.row_flags,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
