"""Prehospital vital-sign fidelity audit.

Audits the quality of EMS / pre-hospital vital signs against the matched
ED/hospital arrival vitals and generates publication-quality figures and
summary tables. This module is intentionally separate from the model training
pipeline: it never mutates ``record.data`` and never overwrites model outputs.

Run this audit BEFORE imputation, one-hot encoding, and normalization so the
figures reflect documented (raw) values rather than model-transformed values.

Outputs:
    Figures -> artifacts/figures/fidelity
    Tables  -> artifacts/tables/fidelity
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: write figures without requiring a display
import matplotlib.pyplot as plt

from src.paths import FIDELITY_FIGURES_DIR, FIDELITY_TABLES_DIR, ensure_dirs
from src.plotting import apply_manuscript_grid

# ---------------------------------------------------------------------------
# Vital-sign mapping.
#
# Column names are the documented NTDB/TQIP headers used in this project.
# ``ems`` / ``hospital`` may be absent in a given dataset build; missing
# columns are detected at runtime and skipped gracefully (with reporting).
# ---------------------------------------------------------------------------
VITAL_PAIRS: Dict[str, Dict[str, object]] = {
    "SBP": {
        "ems": "EMSSBP",
        "hospital": "SBP",
        "unit": "mmHg",
        "ref_lines": [120],
        "plot_xlim": (0, 300),
        "clinical_bands": [10, 20, 30],
        "integer_digit": True,
        "default_values": [120],
        "plausible_low": 40,
        "plausible_high": 300,
    },
    "HR": {
        "ems": "EMSPULSERATE",
        "hospital": "PULSERATE",
        "unit": "bpm",
        "ref_lines": [80],
        "plot_xlim": (0, 250),
        "clinical_bands": [10, 20],
        "integer_digit": True,
        "default_values": [80],
        "plausible_low": 20,
        "plausible_high": 250,
    },
    "RR": {
        "ems": "EMSRESPIRATORYRATE",
        "hospital": "RESPIRATORYRATE",
        "unit": "breaths/min",
        "ref_lines": [16, 18, 20],
        "plot_xlim": (0, 80),
        "clinical_bands": [4, 8],
        "integer_digit": True,
        "default_values": [16, 18, 20],
        "plausible_low": 4,
        "plausible_high": 80,
    },
    "SpO2": {
        "ems": "EMSPULSEOXIMETRY",
        "hospital": "PULSEOXIMETRY",
        "unit": "%",
        "ref_lines": [98, 99, 100],
        "plot_xlim": (50, 100),
        "clinical_bands": [2, 5],
        "integer_digit": False,
        "default_values": [98, 99, 100],
        "plausible_low": 50,
        "plausible_high": 100,
    },
    "GCS": {
        "ems": "EMSTOTALGCS",
        "hospital": "TOTALGCS",
        "unit": "points",
        "ref_lines": [15],
        "plot_xlim": (3, 15),
        "clinical_bands": [1, 2],
        "integer_digit": False,
        "default_values": [15],
        "plausible_low": 3,
        "plausible_high": 15,
    },
}

# Percentiles reported for every distribution summary.
SUMMARY_PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

# Above this paired-sample count, scatter / Bland-Altman plots switch to
# hexbin (and Bland-Altman downsamples points) for legibility. Statistics are
# always computed on the full paired sample regardless of this threshold.
LARGE_SAMPLE_THRESHOLD = 5000
SCATTER_DOWNSAMPLE_N = 20000

# Consistent, publication-quality figure defaults.
FIGURE_DPI = 300
RANDOM_SEED = 42

_PLOT_RC = {
    "figure.dpi": 110,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FidelityAuditResult:
    """Summary of a completed audit run."""

    data_source: str
    n_records: int
    ems_vitals_found: Dict[str, str] = field(default_factory=dict)
    hospital_vitals_found: Dict[str, str] = field(default_factory=dict)
    pairs_analyzed: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    figures_saved: List[Path] = field(default_factory=list)
    tables_saved: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    figures_dir: Optional[Path] = None
    tables_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Raw-value extraction
# ---------------------------------------------------------------------------
def _resolve_raw_source(trauma_dataset) -> Tuple[str, List[str]]:
    """Decide which per-record attribute holds the rawest available values.

    ``record.base_data`` keeps pre-scale physiologic values even after a
    z-score / one-hot transform, so it is preferred when a transform was
    applied. Imputation, however, fills ``base_data`` in place, so when an
    imputation state is present we warn that "missing" counts reflect the
    post-imputation frame and fidelity may be affected.
    """
    notes: List[str] = []
    transform_applied = bool(
        getattr(trauma_dataset, "transform_state", None)
        and trauma_dataset.transform_state.get("applied")
    )
    imputation_applied = getattr(trauma_dataset, "imputation_state", None) is not None

    records = trauma_dataset.get_records()
    has_base = any(
        hasattr(r, "base_data") and r.base_data is not None for r in records
    )

    if imputation_applied:
        notes.append(
            "WARNING: dataset has an imputation_state set. Raw missingness cannot "
            "be recovered from imputed values; missing counts and distributions "
            "may not reflect documented (pre-imputation) values."
        )
    if transform_applied:
        notes.append(
            "WARNING: dataset transforms (z-score/one-hot) were applied. Reading "
            "pre-scale values from record.base_data; reload a pre-transform pickle "
            "for a fully raw audit."
        )

    if has_base:
        source = "base_data"
    else:
        source = "data"
    if not has_base and transform_applied:
        notes.append(
            "WARNING: base_data unavailable; falling back to record.data which "
            "holds TRANSFORMED values. Fidelity figures will be misleading."
        )
    return source, notes


def build_raw_vital_frame(
    trauma_dataset, columns: List[str], source: str
) -> pd.DataFrame:
    """Build a numeric DataFrame of the requested columns from raw values.

    Only columns that exist on at least one record are included. Values are
    coerced to numeric; non-numeric / absent entries become NaN.
    """
    records = trauma_dataset.get_records()
    rows = []
    for record in records:
        if source == "base_data" and getattr(record, "base_data", None) is not None:
            store = record.base_data
        else:
            store = record.data
        rows.append({col: store.get(col, np.nan) for col in columns})
    frame = pd.DataFrame(rows, columns=columns)
    return frame.apply(pd.to_numeric, errors="coerce")


def _available_columns(trauma_dataset, source: str) -> set:
    """Set of column names present on any record (data or base_data)."""
    present: set = set()
    for record in trauma_dataset.get_records():
        if source == "base_data" and getattr(record, "base_data", None) is not None:
            present.update(record.base_data.keys())
        else:
            present.update(record.data.keys())
    return present


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _slug(vital_key: str) -> str:
    return vital_key.lower().replace(" ", "_")


def _save_fig(fig, path: Path, result: FidelityAuditResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    result.figures_saved.append(path)
    return path


def _save_table(df: pd.DataFrame, path: Path, result: FidelityAuditResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    result.tables_saved.append(path)
    return path


def _clean(series: pd.Series) -> pd.Series:
    """Non-missing numeric values only."""
    return pd.to_numeric(series, errors="coerce").dropna()


def _distribution_stats(series: pd.Series) -> Dict[str, float]:
    n_total = int(len(series))
    values = _clean(series)
    n_nonmissing = int(len(values))
    n_missing = n_total - n_nonmissing
    stats: Dict[str, float] = {
        "n_total": n_total,
        "n_nonmissing": n_nonmissing,
        "n_missing": n_missing,
        "percent_missing": (100.0 * n_missing / n_total) if n_total else np.nan,
        "mean": float(values.mean()) if n_nonmissing else np.nan,
        "sd": float(values.std(ddof=1)) if n_nonmissing > 1 else np.nan,
        "median": float(values.median()) if n_nonmissing else np.nan,
        "iqr": (
            float(values.quantile(0.75) - values.quantile(0.25))
            if n_nonmissing
            else np.nan
        ),
        "min": float(values.min()) if n_nonmissing else np.nan,
        "max": float(values.max()) if n_nonmissing else np.nan,
    }
    for pct in SUMMARY_PERCENTILES:
        stats[f"p{pct}"] = (
            float(values.quantile(pct / 100.0)) if n_nonmissing else np.nan
        )
    return stats


def _abs_diff_band_percentages(
    abs_diff: pd.Series, bands: List[int]
) -> Dict[str, float]:
    """Percent of paired absolute differences within each clinical band."""
    out: Dict[str, float] = {}
    n = int(len(abs_diff))
    for band in bands:
        key = f"pct_absdiff_le_{band}"
        out[key] = (
            100.0 * float((abs_diff <= band).sum()) / n if n else np.nan
        )
    return out


# ---------------------------------------------------------------------------
# C. Distribution plots for EMS vitals
# ---------------------------------------------------------------------------
def _plot_single_distribution(ax, values: pd.Series, cfg: Dict, vital_key: str):
    lo, hi = cfg["plot_xlim"]
    n_below = int((values < lo).sum())
    n_above = int((values > hi).sum())

    ax.hist(
        values,
        bins=60,
        range=(lo, hi),
        color="#4C72B0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )
    for ref in cfg["ref_lines"]:
        ax.axvline(ref, color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.annotate(
            f"{ref}",
            xy=(ref, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -10),
            textcoords="offset points",
            color="#C44E52",
            fontsize=8,
            rotation=90,
            va="top",
        )
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"EMS {vital_key} ({cfg['unit']})")
    ax.set_ylabel("Count")
    note = f"n = {len(values):,}"
    clipped = n_below + n_above
    if clipped:
        note += (
            f"\noutside axis: {clipped:,} "
            f"(<{lo}: {n_below:,}, >{hi}: {n_above:,})"
        )
    ax.text(
        0.97,
        0.95,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
    )
    return n_below, n_above


def analyze_distributions(
    ems_frame: pd.DataFrame,
    ems_vitals: Dict[str, str],
    figures_dir: Path,
    tables_dir: Path,
    result: FidelityAuditResult,
) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    panels: List[Tuple[str, str]] = []  # (vital_key, column)

    for vital_key, column in ems_vitals.items():
        cfg = VITAL_PAIRS[vital_key]
        series = ems_frame[column]
        values = _clean(series)
        stats = _distribution_stats(series)

        lo, hi = cfg["plot_xlim"]
        row = {"vital": vital_key, "ems_column": column, "unit": cfg["unit"]}
        row.update(stats)
        row["n_below_axis_limit"] = int((values < lo).sum())
        row["n_above_axis_limit"] = int((values > hi).sum())
        summary_rows.append(row)

        if stats["n_nonmissing"] == 0:
            result.warnings.append(
                f"{vital_key}: no non-missing EMS values; distribution skipped."
            )
            continue

        with plt.rc_context(_PLOT_RC):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            _plot_single_distribution(ax, values, cfg, vital_key)
            ax.set_title(f"EMS {vital_key} distribution")
            fig.tight_layout()
            _save_fig(
                fig,
                figures_dir / f"ems_{_slug(vital_key)}_distribution.png",
                result,
            )
        panels.append((vital_key, column))

    # Combined faceted figure.
    if panels:
        n = len(panels)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        with plt.rc_context(_PLOT_RC):
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(6.5 * ncols, 3.8 * nrows), squeeze=False
            )
            flat = axes.ravel()
            for idx, (vital_key, column) in enumerate(panels):
                cfg = VITAL_PAIRS[vital_key]
                values = _clean(ems_frame[column])
                _plot_single_distribution(flat[idx], values, cfg, vital_key)
                flat[idx].set_title(f"EMS {vital_key}")
            for extra in range(len(panels), len(flat)):
                flat[extra].axis("off")
            fig.suptitle(
                "EMS prehospital vital-sign distributions (documented values)",
                fontsize=15,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            _save_fig(
                fig,
                figures_dir / "ems_vital_distributions_combined.png",
                result,
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        _save_table(
            summary_df,
            tables_dir / "ems_vital_distribution_summary.csv",
            result,
        )
    return summary_df


# ---------------------------------------------------------------------------
# D. Digit preference / terminal digit analysis
# ---------------------------------------------------------------------------
def _terminal_digits(values: pd.Series) -> Optional[pd.Series]:
    """Terminal (ones) digit for near-integer numeric values.

    Returns None when the data are not integer-like (to avoid manufacturing a
    digit pattern out of decimal measurements).
    """
    values = _clean(values)
    if values.empty:
        return None
    frac = (values - values.round()).abs()
    near_integer_share = float((frac < 1e-6).mean())
    if near_integer_share < 0.95:
        return None
    rounded = values.round().astype("int64")
    return (rounded.abs() % 10).astype(int)


def analyze_terminal_digits(
    ems_frame: pd.DataFrame,
    ems_vitals: Dict[str, str],
    figures_dir: Path,
    tables_dir: Path,
    result: FidelityAuditResult,
) -> pd.DataFrame:
    rows: List[Dict] = []
    # Integer-like vitals where terminal-digit preference is clinically
    # meaningful (rounding to 0/5 indicates estimation).
    targets = [k for k in ("SBP", "HR") if k in ems_vitals]

    for vital_key in targets:
        column = ems_vitals[vital_key]
        digits = _terminal_digits(ems_frame[column])
        if digits is None or digits.empty:
            result.warnings.append(
                f"{vital_key}: values not integer-like; terminal-digit analysis skipped."
            )
            continue

        n = int(len(digits))
        counts = digits.value_counts().reindex(range(10), fill_value=0).sort_index()
        pct = 100.0 * counts / n
        pct_0 = float(pct.get(0, 0.0))
        pct_5 = float(pct.get(5, 0.0))
        pct_0_or_5 = pct_0 + pct_5

        row = {
            "vital": vital_key,
            "ems_column": column,
            "n_nonmissing": n,
            "prop_ending_0": pct_0 / 100.0,
            "prop_ending_5": pct_5 / 100.0,
            "prop_ending_0_or_5": pct_0_or_5 / 100.0,
        }
        for digit in range(10):
            row[f"count_{digit}"] = int(counts[digit])
            row[f"pct_{digit}"] = float(pct[digit])
        rows.append(row)

        with plt.rc_context(_PLOT_RC):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            apply_manuscript_grid(ax)
            bars = ax.bar(
                range(10), pct.values, color="#4C72B0", edgecolor="white"
            )
            bars[0].set_color("#C44E52")
            bars[5].set_color("#C44E52")
            ax.axhline(10.0, color="0.4", linestyle=":", linewidth=1.0)
            ax.annotate(
                "uniform (10%)",
                xy=(9, 10.0),
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color="0.4",
            )
            ax.set_xticks(range(10))
            ax.set_xlabel("Terminal digit")
            ax.set_ylabel("Percent of non-missing values (%)")
            ax.set_title(f"EMS {vital_key} terminal-digit preference")
            ax.text(
                0.97,
                0.95,
                f"n = {n:,}\nending in 0 or 5: {pct_0_or_5:.1f}%\n"
                f"(0: {pct_0:.1f}%, 5: {pct_5:.1f}%)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
            )
            fig.tight_layout()
            _save_fig(
                fig,
                figures_dir / f"ems_{_slug(vital_key)}_terminal_digit.png",
                result,
            )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        _save_table(
            summary_df,
            tables_dir / "ems_terminal_digit_summary.csv",
            result,
        )
    return summary_df


def analyze_default_value_flags(
    ems_frame: pd.DataFrame,
    ems_vitals: Dict[str, str],
    tables_dir: Path,
    result: FidelityAuditResult,
) -> pd.DataFrame:
    rows: List[Dict] = []

    def _flag(label: str, mask: pd.Series, denom_values: pd.Series):
        denom = int(denom_values.notna().sum())
        count = int(mask.sum())
        rows.append(
            {
                "flag": label,
                "count": count,
                "denominator_nonmissing": denom,
                "percent": (100.0 * count / denom) if denom else np.nan,
            }
        )

    for vital_key, column in ems_vitals.items():
        cfg = VITAL_PAIRS[vital_key]
        values = pd.to_numeric(ems_frame[column], errors="coerce")
        clean = values.dropna()
        defaults = cfg["default_values"]
        if len(defaults) == 1:
            label = f"EMS {vital_key} == {defaults[0]}"
        else:
            label = f"EMS {vital_key} in {{{', '.join(str(d) for d in defaults)}}}"
        _flag(label, clean.isin(defaults), values)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        _save_table(
            summary_df,
            tables_dir / "ems_default_value_flags.csv",
            result,
        )
    return summary_df


# ---------------------------------------------------------------------------
# Paired-data helpers (shared by Bland-Altman and scatter sections)
# ---------------------------------------------------------------------------
def _paired_values(
    ems_frame: pd.DataFrame,
    hosp_frame: pd.DataFrame,
    ems_col: str,
    hosp_col: str,
) -> Tuple[pd.Series, pd.Series]:
    ems = pd.to_numeric(ems_frame[ems_col], errors="coerce")
    hosp = pd.to_numeric(hosp_frame[hosp_col], errors="coerce")
    both = ems.notna() & hosp.notna()
    return ems[both].reset_index(drop=True), hosp[both].reset_index(drop=True)


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    try:
        return float(a.corr(b, method=method))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# E. Bland-Altman plots: EMS vs hospital/ED vitals
# ---------------------------------------------------------------------------
def analyze_bland_altman(
    ems_frame: pd.DataFrame,
    hosp_frame: pd.DataFrame,
    matched_pairs: Dict[str, Tuple[str, str]],
    figures_dir: Path,
    tables_dir: Path,
    result: FidelityAuditResult,
) -> pd.DataFrame:
    rows: List[Dict] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for vital_key, (ems_col, hosp_col) in matched_pairs.items():
        cfg = VITAL_PAIRS[vital_key]
        ems, hosp = _paired_values(ems_frame, hosp_frame, ems_col, hosp_col)
        n = int(len(ems))
        if n == 0:
            result.warnings.append(
                f"{vital_key}: no paired EMS/hospital values; Bland-Altman skipped."
            )
            continue

        mean_vals = (ems + hosp) / 2.0
        diff = ems - hosp  # EMS minus hospital
        abs_diff = diff.abs()
        md = float(diff.mean())
        sd = float(diff.std(ddof=1)) if n > 1 else np.nan
        loa_low = md - 1.96 * sd if n > 1 else np.nan
        loa_high = md + 1.96 * sd if n > 1 else np.nan

        row = {
            "vital": vital_key,
            "ems_column": ems_col,
            "hospital_column": hosp_col,
            "unit": cfg["unit"],
            "n_paired": n,
            "mean_difference": md,
            "sd_difference": sd,
            "lower_limit_agreement": loa_low,
            "upper_limit_agreement": loa_high,
            "median_absolute_difference": float(abs_diff.median()),
            "iqr_absolute_difference": float(
                abs_diff.quantile(0.75) - abs_diff.quantile(0.25)
            ),
        }
        row.update(_abs_diff_band_percentages(abs_diff, cfg["clinical_bands"]))
        rows.append(row)

        downsampled = False
        if n > SCATTER_DOWNSAMPLE_N:
            sel = rng.choice(n, size=SCATTER_DOWNSAMPLE_N, replace=False)
            plot_mean, plot_diff = mean_vals.iloc[sel], diff.iloc[sel]
            downsampled = True
        else:
            plot_mean, plot_diff = mean_vals, diff

        with plt.rc_context(_PLOT_RC):
            fig, ax = plt.subplots(figsize=(7, 5))
            if n > LARGE_SAMPLE_THRESHOLD:
                hb = ax.hexbin(
                    plot_mean, plot_diff, gridsize=45, cmap="Blues", mincnt=1
                )
                fig.colorbar(hb, ax=ax, label="count")
            else:
                ax.scatter(
                    plot_mean, plot_diff, s=12, alpha=0.25, color="#4C72B0",
                    edgecolors="none",
                )
            ax.axhline(md, color="#C44E52", linestyle="-", linewidth=1.4,
                       label=f"mean diff = {md:.2f}")
            ax.axhline(loa_high, color="#55A868", linestyle="--", linewidth=1.2,
                       label=f"+1.96 SD = {loa_high:.2f}")
            ax.axhline(loa_low, color="#55A868", linestyle="--", linewidth=1.2,
                       label=f"-1.96 SD = {loa_low:.2f}")
            ax.axhline(0.0, color="0.6", linestyle=":", linewidth=0.9)
            ax.set_xlabel(f"Mean of EMS and hospital {vital_key} ({cfg['unit']})")
            ax.set_ylabel(f"EMS - hospital {vital_key} ({cfg['unit']})")
            subtitle = f"n = {n:,} paired"
            if downsampled:
                subtitle += f"  (plot shows {SCATTER_DOWNSAMPLE_N:,} sampled points)"
            ax.set_title(f"Bland-Altman: EMS vs hospital {vital_key}\n{subtitle}",
                         fontsize=12)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            _save_fig(
                fig,
                figures_dir / f"bland_altman_{_slug(vital_key)}_ems_vs_hospital.png",
                result,
            )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        _save_table(
            summary_df,
            tables_dir / "ems_hospital_bland_altman_summary.csv",
            result,
        )
    return summary_df


# ---------------------------------------------------------------------------
# F. Scatter plots with identity line
# ---------------------------------------------------------------------------
def analyze_scatter_agreement(
    ems_frame: pd.DataFrame,
    hosp_frame: pd.DataFrame,
    matched_pairs: Dict[str, Tuple[str, str]],
    figures_dir: Path,
    tables_dir: Path,
    result: FidelityAuditResult,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for vital_key, (ems_col, hosp_col) in matched_pairs.items():
        cfg = VITAL_PAIRS[vital_key]
        ems, hosp = _paired_values(ems_frame, hosp_frame, ems_col, hosp_col)
        n = int(len(ems))
        if n == 0:
            result.warnings.append(
                f"{vital_key}: no paired EMS/hospital values; scatter skipped."
            )
            continue

        abs_diff = (ems - hosp).abs()
        pearson = _safe_corr(ems, hosp, "pearson")
        spearman = _safe_corr(ems, hosp, "spearman")

        row = {
            "vital": vital_key,
            "ems_column": ems_col,
            "hospital_column": hosp_col,
            "unit": cfg["unit"],
            "n_paired": n,
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "mean_absolute_difference": float(abs_diff.mean()),
            "median_absolute_difference": float(abs_diff.median()),
            "iqr_absolute_difference": float(
                abs_diff.quantile(0.75) - abs_diff.quantile(0.25)
            ),
        }
        row.update(_abs_diff_band_percentages(abs_diff, cfg["clinical_bands"]))
        rows.append(row)

        lo, hi = cfg["plot_xlim"]
        with plt.rc_context(_PLOT_RC):
            fig, ax = plt.subplots(figsize=(6, 6))
            apply_manuscript_grid(ax)
            if n > LARGE_SAMPLE_THRESHOLD:
                hb = ax.hexbin(
                    ems, hosp, gridsize=45, cmap="Blues", mincnt=1,
                    extent=(lo, hi, lo, hi),
                )
                fig.colorbar(hb, ax=ax, label="count")
            else:
                ax.scatter(
                    ems, hosp, s=12, alpha=0.25, color="#4C72B0",
                    edgecolors="none",
                )
            ax.plot([lo, hi], [lo, hi], color="#C44E52", linestyle="--",
                    linewidth=1.3, label="identity (y = x)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"EMS {vital_key} ({cfg['unit']})")
            ax.set_ylabel(f"Hospital {vital_key} ({cfg['unit']})")
            corr_txt = []
            if not np.isnan(pearson):
                corr_txt.append(f"Pearson r = {pearson:.3f}")
            if not np.isnan(spearman):
                corr_txt.append(f"Spearman rho = {spearman:.3f}")
            note = f"n = {n:,} paired"
            if corr_txt:
                note += "\n" + "\n".join(corr_txt)
            ax.text(
                0.05, 0.95, note, transform=ax.transAxes, ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
            )
            ax.set_title(f"EMS vs hospital {vital_key}")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            _save_fig(
                fig,
                figures_dir / f"scatter_{_slug(vital_key)}_ems_vs_hospital.png",
                result,
            )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        _save_table(
            summary_df,
            tables_dir / "ems_hospital_agreement_summary.csv",
            result,
        )
    return summary_df


# ---------------------------------------------------------------------------
# G. Physiologic plausibility checks
# ---------------------------------------------------------------------------
def analyze_plausibility(
    ems_frame: pd.DataFrame,
    hosp_frame: pd.DataFrame,
    ems_vitals: Dict[str, str],
    hospital_vitals: Dict[str, str],
    tables_dir: Path,
    result: FidelityAuditResult,
    save_row_flags: bool = False,
) -> pd.DataFrame:
    rows: List[Dict] = []
    row_flags = pd.DataFrame(index=range(len(ems_frame)))

    def _add_rule(vital, source, rule_name, mask: pd.Series, evaluated: pd.Series):
        n_eval = int(evaluated.sum())
        n_flag = int((mask & evaluated).sum())
        rows.append(
            {
                "vital": vital,
                "source": source,
                "rule": rule_name,
                "n_evaluated": n_eval,
                "n_flagged": n_flag,
                "percent_flagged": (100.0 * n_flag / n_eval) if n_eval else np.nan,
            }
        )
        if save_row_flags:
            row_flags[f"{source}_{vital}_{rule_name}"] = (mask & evaluated).astype(int)

    sources = [("EMS", ems_frame, ems_vitals), ("hospital", hosp_frame, hospital_vitals)]
    for source, frame, vitals in sources:
        for vital_key, column in vitals.items():
            cfg = VITAL_PAIRS[vital_key]
            values = pd.to_numeric(frame[column], errors="coerce")
            present = values.notna()
            _add_rule(
                vital_key, source,
                f"implausible_low_lt_{cfg['plausible_low']}",
                values < cfg["plausible_low"], present,
            )
            _add_rule(
                vital_key, source,
                f"implausible_high_gt_{cfg['plausible_high']}",
                values > cfg["plausible_high"], present,
            )

    # EMS shock index = EMS HR / EMS SBP.
    if "HR" in ems_vitals and "SBP" in ems_vitals:
        hr = pd.to_numeric(ems_frame[ems_vitals["HR"]], errors="coerce")
        sbp = pd.to_numeric(ems_frame[ems_vitals["SBP"]], errors="coerce")
        both = hr.notna() & sbp.notna()
        invalid_sbp = both & (sbp <= 0)
        valid = both & (sbp > 0)
        shock_index = hr.where(valid) / sbp.where(valid)
        _add_rule("shock_index", "EMS", "sbp_le_0_invalid", invalid_sbp, both)
        _add_rule("shock_index", "EMS", "shock_index_gt_3",
                  shock_index > 3, valid)
    else:
        result.warnings.append(
            "EMS shock index skipped (EMS HR and EMS SBP not both available)."
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        _save_table(
            summary_df,
            tables_dir / "vital_plausibility_flags_summary.csv",
            result,
        )
    if save_row_flags and not row_flags.empty:
        _save_table(
            row_flags,
            tables_dir / "vital_plausibility_row_flags.csv",
            result,
        )
    return summary_df


# ---------------------------------------------------------------------------
# H. Combined manuscript-ready figure
# ---------------------------------------------------------------------------
def build_combined_manuscript_figure(
    ems_frame: pd.DataFrame,
    hosp_frame: pd.DataFrame,
    ems_vitals: Dict[str, str],
    matched_pairs: Dict[str, Tuple[str, str]],
    figures_dir: Path,
    result: FidelityAuditResult,
) -> Optional[Path]:
    # Panel selection prefers SBP + HR but degrades gracefully.
    if "SBP" not in ems_vitals and "HR" not in ems_vitals:
        result.warnings.append(
            "Combined manuscript figure skipped (neither EMS SBP nor EMS HR present)."
        )
        return None

    with plt.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Panel A: EMS SBP distribution
        ax = axes[0, 0]
        if "SBP" in ems_vitals:
            cfg = VITAL_PAIRS["SBP"]
            _plot_single_distribution(ax, _clean(ems_frame[ems_vitals["SBP"]]), cfg, "SBP")
            ax.set_title("A. EMS SBP distribution", loc="left")
        else:
            ax.axis("off")

        # Panel B: EMS HR distribution
        ax = axes[0, 1]
        if "HR" in ems_vitals:
            cfg = VITAL_PAIRS["HR"]
            _plot_single_distribution(ax, _clean(ems_frame[ems_vitals["HR"]]), cfg, "HR")
            ax.set_title("B. EMS HR distribution", loc="left")
        else:
            ax.axis("off")

        # Panel C: EMS SBP terminal-digit preference
        ax = axes[1, 0]
        digits = (
            _terminal_digits(ems_frame[ems_vitals["SBP"]])
            if "SBP" in ems_vitals
            else None
        )
        if digits is not None and not digits.empty:
            n = int(len(digits))
            counts = digits.value_counts().reindex(range(10), fill_value=0).sort_index()
            pct = 100.0 * counts / n
            bars = ax.bar(range(10), pct.values, color="#4C72B0", edgecolor="white")
            bars[0].set_color("#C44E52")
            bars[5].set_color("#C44E52")
            ax.axhline(10.0, color="0.4", linestyle=":", linewidth=1.0)
            ax.set_xticks(range(10))
            ax.set_xlabel("Terminal digit")
            ax.set_ylabel("Percent (%)")
            ax.set_title("C. EMS SBP terminal-digit preference", loc="left")
        else:
            ax.axis("off")

        # Panel D: SBP Bland-Altman EMS vs hospital
        ax = axes[1, 1]
        if "SBP" in matched_pairs:
            ems_col, hosp_col = matched_pairs["SBP"]
            ems, hosp = _paired_values(ems_frame, hosp_frame, ems_col, hosp_col)
            if len(ems) > 0:
                mean_vals = (ems + hosp) / 2.0
                diff = ems - hosp
                md = float(diff.mean())
                sd = float(diff.std(ddof=1)) if len(diff) > 1 else np.nan
                if len(ems) > LARGE_SAMPLE_THRESHOLD:
                    ax.hexbin(mean_vals, diff, gridsize=40, cmap="Blues", mincnt=1)
                else:
                    ax.scatter(mean_vals, diff, s=10, alpha=0.25,
                               color="#4C72B0", edgecolors="none")
                ax.axhline(md, color="#C44E52", linewidth=1.4)
                ax.axhline(md + 1.96 * sd, color="#55A868", linestyle="--", linewidth=1.2)
                ax.axhline(md - 1.96 * sd, color="#55A868", linestyle="--", linewidth=1.2)
                ax.set_xlabel("Mean of EMS and hospital SBP (mmHg)")
                ax.set_ylabel("EMS - hospital SBP (mmHg)")
                ax.set_title(
                    f"D. SBP Bland-Altman (n = {len(ems):,})", loc="left"
                )
            else:
                ax.axis("off")
        else:
            ax.axis("off")

        fig.suptitle(
            "Prehospital EMS vital-sign fidelity audit",
            fontsize=16,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return _save_fig(
            fig,
            figures_dir / "ems_vital_fidelity_audit_combined.png",
            result,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_prehospital_vital_fidelity_audit(
    trauma_dataset,
    *,
    figures_dir: Path = FIDELITY_FIGURES_DIR,
    tables_dir: Path = FIDELITY_TABLES_DIR,
    save_row_flags: bool = False,
) -> FidelityAuditResult:
    """Run the full prehospital vital-sign fidelity audit.

    Reads raw (documented) values from the loaded :class:`TraumaDataset` without
    mutating it, writes publication-quality figures and summary tables, and
    returns a :class:`FidelityAuditResult` describing what was produced.
    """
    if trauma_dataset is None:
        raise ValueError("No trauma_dataset provided. Load or pickle a dataset first.")

    ensure_dirs()
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    source, source_notes = _resolve_raw_source(trauma_dataset)
    result = FidelityAuditResult(
        data_source=source,
        n_records=len(trauma_dataset.get_records()),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    result.warnings.extend(source_notes)

    available = _available_columns(trauma_dataset, source)

    # Resolve which EMS / hospital columns are actually present.
    ems_vitals: Dict[str, str] = {}
    hospital_vitals: Dict[str, str] = {}
    matched_pairs: Dict[str, Tuple[str, str]] = {}
    for vital_key, cfg in VITAL_PAIRS.items():
        ems_col = cfg["ems"]
        hosp_col = cfg["hospital"]
        if ems_col in available:
            ems_vitals[vital_key] = ems_col
        else:
            result.missing_columns.append(ems_col)
        if hosp_col in available:
            hospital_vitals[vital_key] = hosp_col
        else:
            result.missing_columns.append(hosp_col)
        if vital_key in ems_vitals and vital_key in hospital_vitals:
            matched_pairs[vital_key] = (ems_col, hosp_col)

    result.ems_vitals_found = dict(ems_vitals)
    result.hospital_vitals_found = dict(hospital_vitals)
    result.pairs_analyzed = list(matched_pairs.keys())

    if not ems_vitals:
        result.warnings.append(
            "No EMS vital columns found in the dataset; nothing to audit."
        )
        _print_summary(result)
        return result

    all_ems_cols = list(dict.fromkeys(ems_vitals.values()))
    all_hosp_cols = list(dict.fromkeys(hospital_vitals.values()))
    ems_frame = build_raw_vital_frame(trauma_dataset, all_ems_cols, source)
    hosp_frame = (
        build_raw_vital_frame(trauma_dataset, all_hosp_cols, source)
        if all_hosp_cols
        else pd.DataFrame(index=range(result.n_records))
    )

    # C. Distributions
    analyze_distributions(ems_frame, ems_vitals, figures_dir, tables_dir, result)
    # D. Terminal digits + default-value flags
    analyze_terminal_digits(ems_frame, ems_vitals, figures_dir, tables_dir, result)
    analyze_default_value_flags(ems_frame, ems_vitals, tables_dir, result)
    # E. Bland-Altman
    if matched_pairs:
        analyze_bland_altman(
            ems_frame, hosp_frame, matched_pairs, figures_dir, tables_dir, result
        )
        # F. Scatter with identity line
        analyze_scatter_agreement(
            ems_frame, hosp_frame, matched_pairs, figures_dir, tables_dir, result
        )
    else:
        result.warnings.append(
            "No matched EMS/hospital pairs; Bland-Altman and scatter skipped."
        )
    # G. Plausibility
    analyze_plausibility(
        ems_frame, hosp_frame, ems_vitals, hospital_vitals, tables_dir, result,
        save_row_flags=save_row_flags,
    )
    # H. Combined manuscript figure
    build_combined_manuscript_figure(
        ems_frame, hosp_frame, ems_vitals, matched_pairs, figures_dir, result
    )

    _print_summary(result)
    return result


def _print_summary(result: FidelityAuditResult) -> None:
    print("\n=== Prehospital Vital-Sign Fidelity Audit ===")
    print(f"Records audited:      {result.n_records:,}")
    print(f"Raw value source:     record.{result.data_source}")
    print(
        "EMS vitals found:     "
        + (
            ", ".join(f"{k} ({v})" for k, v in result.ems_vitals_found.items())
            or "(none)"
        )
    )
    print(
        "Hospital vitals found:"
        + (
            " " + ", ".join(
                f"{k} ({v})" for k, v in result.hospital_vitals_found.items()
            )
            if result.hospital_vitals_found
            else " (none)"
        )
    )
    print(
        "Matched pairs:        "
        + (", ".join(result.pairs_analyzed) or "(none)")
    )
    if result.missing_columns:
        print(
            "Expected-but-absent:  "
            + ", ".join(sorted(set(result.missing_columns)))
        )
    print(f"Figures saved:        {len(result.figures_saved)}")
    if result.figures_dir:
        print(f"  -> {result.figures_dir}")
    print(f"Summary tables saved: {len(result.tables_saved)}")
    if result.tables_dir:
        print(f"  -> {result.tables_dir}")
    if result.warnings:
        print("\nNotes / warnings:")
        for note in result.warnings:
            print(f"  - {note}")
    print("Fidelity audit complete.\n")
