"""Render the combined calibration + risk-decile manuscript figure from saved CSVs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.nfti_positive_primary import _plot_calibration_and_deciles_combined
from src.paths import (
    FIGURE4_CALIBRATION_AND_DECILES_PATH,
    NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH,
    NFTI_POSITIVE_XGB_CALIBRATION_BINS_PATH,
    NFTI_POSITIVE_XGB_RISK_DECILES_PATH,
)


def _read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            converted = {}
            for key, value in row.items():
                if value is None or value == "":
                    converted[key] = value
                    continue
                try:
                    if "." in value or "e" in value.lower():
                        converted[key] = float(value)
                    else:
                        converted[key] = int(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def main() -> None:
    calibration_rows = _read_csv(NFTI_POSITIVE_XGB_CALIBRATION_BINS_PATH)
    decile_rows = _read_csv(NFTI_POSITIVE_XGB_RISK_DECILES_PATH)

    _plot_calibration_and_deciles_combined(
        calibration_rows,
        decile_rows,
        NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH,
    )
    _plot_calibration_and_deciles_combined(
        calibration_rows,
        decile_rows,
        FIGURE4_CALIBRATION_AND_DECILES_PATH,
    )

    print(f"Wrote: {NFTI_POSITIVE_XGB_CALIBRATION_AND_DECILES_FIGURE_PATH}")
    print(f"Wrote: {FIGURE4_CALIBRATION_AND_DECILES_PATH}")


if __name__ == "__main__":
    main()
