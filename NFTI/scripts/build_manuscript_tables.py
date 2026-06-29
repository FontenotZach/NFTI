"""Build polished, manuscript-ready tables from existing analysis artifacts.

This script is intentionally self-contained and deterministic. It reads the
already-computed analysis artifacts under ``artifacts/`` and emits publication
formatted tables under ``results/tables/`` in two flavors:

  * ``*.csv``  - clean, machine-readable values (one consistent format).
  * ``*.md``   - formatted Markdown that can be pasted directly into a manuscript.

Formatting conventions (applied consistently across every table):
  * Discrimination / calibration indices (AUROC, AUPRC, Brier): 3 decimals.
  * Operating-point metrics and any rate/proportion (sensitivity, specificity,
    PPV, NPV, accuracy, F1, prevalence, missingness, clustering): percentage
    with 1 decimal place.
  * Pearson r: 2 decimals.
  * Counts: thousands-separated integers.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
TABLES = ART / "tables"
REPORTS = ART / "reports"
METRICS = ART / "metrics"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def pct(x, digits: int = 1) -> str:
    """Format a proportion in [0, 1] as a percentage string."""
    if x is None or x == "":
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def dec(x, digits: int = 3) -> str:
    if x is None or x == "":
        return "—"
    return f"{float(x):.{digits}f}"


def num(x, digits: int = 1) -> str:
    if x is None or x == "":
        return "—"
    return f"{float(x):.{digits}f}"


def cnt(x) -> str:
    if x is None or x == "":
        return "—"
    return f"{int(round(float(x))):,}"


def write_md(path: Path, title: str, header: list[str], rows: list[list[str]], notes: list[str] | None = None) -> None:
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    if notes:
        lines.append("")
        for n in notes:
            lines.append(n)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Table 1 - Baseline cohort characteristics, overall and by NFTI status.
# Categorical counts are reconstructed from the per-stratum cohort size and
# observed NFTI-positive rate (count = round(n * rate)); continuous vitals are
# reported as overall median (IQR) with percent missing.
# ---------------------------------------------------------------------------
def build_table1() -> None:
    ctx = read_csv(TABLES / "missingness" / "ems_missingness_by_clinical_context_compact.csv")
    vit = read_csv(TABLES / "fidelity" / "ems_vital_distribution_summary.csv")

    by_var: dict[str, list[dict]] = {}
    for row in ctx:
        by_var.setdefault(row["stratum_variable"], []).append(row)

    # Overall totals from the NFTI status stratum.
    nfti_rows = {r["stratum_value"]: r for r in by_var["NFTI status"]}
    n_pos = int(round(float(nfti_rows["NFTI positive"]["n_records"])))
    n_neg = int(round(float(nfti_rows["NFTI negative"]["n_records"])))
    n_total = n_pos + n_neg

    header = [
        "Characteristic",
        f"Overall (N = {n_total:,})",
        f"NFTI-positive (n = {n_pos:,})",
        f"NFTI-negative (n = {n_neg:,})",
    ]
    rows: list[list[str]] = []

    def add_blank_section(label: str) -> None:
        rows.append([f"**{label}**", "", "", ""])

    def add_category(name: str, n_records: float, rate: float) -> None:
        n_records = int(round(float(n_records)))
        pos = int(round(n_records * float(rate)))
        neg = n_records - pos
        rows.append([
            name,
            f"{n_records:,} ({n_records / n_total * 100:.1f}%)",
            f"{pos:,} ({pos / n_pos * 100:.1f}%)",
            f"{neg:,} ({neg / n_neg * 100:.1f}%)",
        ])

    # Preferred ordering / labels for categorical blocks.
    section_specs = [
        ("Age group, years", "Age group", ["18-39", "40-64", "65+", "<18"],
         {"18-39": "18–39", "40-64": "40–64", "65+": "≥65", "<18": "<18"}),
        ("Sex", "Sex", ["Male", "Female"], {}),
        ("Race", "Race", ["White", "Black", "Asian", "American Indian",
                             "Pacific Islander", "Other race", "Multiple races",
                             "Unknown/Not recorded"], {}),
        ("Ethnicity", "Ethnicity", ["Hispanic", "Non-Hispanic"], {}),
        ("Arrival mode", "Arrival mode", ["Ground ambulance", "Helicopter", "Fixed-wing"], {}),
        ("Trauma type", "Trauma type", ["Blunt", "Penetrating", "Burn", "Other"], {}),
        ("EMS GCS category", "EMS GCS category",
         ["13-15 (mild)", "9-12 (moderate)", "3-8 (severe)"], {}),
        ("EMS shock index", "EMS shock index",
         ["<0.7 (normal)", "0.7-1.0 (elevated)", ">=1.0 (high)"],
         {">=1.0 (high)": "≥1.0 (high)"}),
        ("Prehospital cardiac arrest", "Prehospital cardiac arrest", ["Yes", "No"], {}),
        ("ED hypotension (SBP < 90 mmHg)", "ED hypotension (SBP<90)",
         ["SBP<90", "SBP>=90"], {"SBP<90": "Present", "SBP>=90": "Absent"}),
    ]

    for section_label, var, order, relabel in section_specs:
        if var not in by_var:
            continue
        add_blank_section(section_label)
        lut = {r["stratum_value"]: r for r in by_var[var]}
        for key in order:
            if key not in lut:
                continue
            disp = relabel.get(key, key)
            add_category(disp, lut[key]["n_records"], lut[key]["nfti_positive_rate"])

    # Continuous EMS vitals (overall median (IQR), percent missing).
    add_blank_section("EMS vital signs, median (IQR) [% missing]")
    vit_labels = {
        "SBP": "Systolic blood pressure, mmHg",
        "HR": "Heart rate, beats/min",
        "RR": "Respiratory rate, breaths/min",
        "SpO2": "Pulse oximetry, %",
        "GCS": "Total Glasgow Coma Scale",
    }
    for v in vit:
        label = vit_labels.get(v["vital"], v["vital"])
        median = float(v["median"])
        iqr = float(v["iqr"])
        miss = float(v["percent_missing"]) / 100.0
        median_s = f"{median:.0f}"
        iqr_s = f"{iqr:.0f}"
        rows.append([label, f"{median_s} ({iqr_s}) [{miss * 100:.1f}%]", "—", "—"])

    notes = [
        f"*NFTI-positive prevalence in the full analytic cohort was {n_pos / n_total * 100:.1f}%.*",
        "*Category counts within NFTI strata are reconstructed from per-stratum cohort "
        "size and observed NFTI-positive rate; columns may not sum exactly to the group "
        "total because of rounding. EMS vital signs are summarized for the overall cohort "
        "(NFTI-stratified vital distributions were not part of the source artifacts).*",
    ]

    write_md(OUT / "table1_baseline_characteristics.md",
             "Table 1. Baseline characteristics of the analytic cohort, overall and by NFTI status",
             header, rows, notes)
    # CSV version (strip markdown bold markers).
    csv_rows = [[c.replace("**", "") for c in r] for r in rows]
    write_csv(OUT / "table1_baseline_characteristics.csv", header, csv_rows)


# ---------------------------------------------------------------------------
# Table 2 - Holdout discrimination, calibration, and operating-point metrics.
# ---------------------------------------------------------------------------
def build_table2() -> None:
    cmp_rows = {r["model"]: r for r in read_csv(REPORTS / "nfti_positive_model_comparison_xgb_vs_lr.csv")}
    xgb_05 = read_csv(REPORTS / "nfti_positive_xgboost_threshold_0_5_metrics.csv")[0]
    xgb_hs = read_csv(REPORTS / "nfti_positive_xgboost_holdout_80_sensitivity_threshold_metrics.csv")[0]
    xgb = cmp_rows["xgboost"]
    lr = cmp_rows["logistic_regression"]

    header = ["Model (operating point)", "Sensitivity", "Specificity", "PPV", "NPV",
              "Accuracy", "F1", "AUROC", "AUPRC", "Brier", "TP", "FP", "TN", "FN"]

    def metric_row(name, r, with_curves=True):
        return [
            name,
            pct(r["sensitivity"]), pct(r["specificity"]), pct(r["precision"]),
            pct(r["NPV"]), pct(r["accuracy"]), dec(r["F1"]),
            dec(r["AUROC"]) if with_curves else "—",
            dec(r["AUPRC"]) if with_curves else "—",
            dec(r["Brier"]) if with_curves else "—",
            cnt(r["TP"]), cnt(r["FP"]), cnt(r["TN"]), cnt(r["FN"]),
        ]

    rows = [
        metric_row("XGBoost (threshold 0.50)", xgb_05),
        metric_row(f"XGBoost (validation-locked threshold {float(xgb_hs['threshold']):.2f})", xgb_hs),
        metric_row("Logistic regression (threshold 0.50)", lr),
    ]

    notes = [
        f"*Holdout cohort: N = {int(float(xgb_05['n'])):,}; NFTI-positive prevalence "
        f"{pct(xgb_05['prevalence'])}.*",
        "*AUROC, AUPRC, and Brier score are threshold-independent and identical across "
        "the two XGBoost operating points. The validation-locked threshold was selected "
        "to achieve ≥80% sensitivity on the validation set and then applied to the holdout "
        "set. Sensitivity, specificity, PPV, NPV, accuracy, and prevalence are reported as "
        "percentages; AUROC, AUPRC, F1, and Brier score as decimals.*",
    ]

    write_md(OUT / "table2_operating_point_metrics.md",
             "Table 2. Holdout discrimination, calibration, and operating-point metrics",
             header, rows, notes)
    write_csv(OUT / "table2_operating_point_metrics.csv", header, rows)


# ---------------------------------------------------------------------------
# Table 3 - EMS vital-sign fidelity vs hospital-arrival values.
# ---------------------------------------------------------------------------
def build_table3() -> None:
    dist = {r["vital"]: r for r in read_csv(TABLES / "fidelity" / "ems_vital_distribution_summary.csv")}
    agree = {r["vital"]: r for r in read_csv(TABLES / "fidelity" / "ems_hospital_agreement_summary.csv")}
    ba = {r["vital"]: r for r in read_csv(TABLES / "fidelity" / "ems_hospital_bland_altman_summary.csv")}
    digit = {r["vital"]: r for r in read_csv(TABLES / "fidelity" / "ems_terminal_digit_summary.csv")}
    default_flags = read_csv(TABLES / "fidelity" / "ems_default_value_flags.csv")
    plaus = read_csv(TABLES / "fidelity" / "vital_plausibility_flags_summary.csv")

    # Default-value clustering keyed to a vital.
    default_lut = {
        "GCS": next(r for r in default_flags if r["flag"] == "EMS GCS == 15"),
        "RR": next(r for r in default_flags if "RR in" in r["flag"]),
        "SpO2": next(r for r in default_flags if "SpO2 in" in r["flag"]),
    }
    default_text = {
        "GCS": "GCS = 15",
        "RR": "RR ∈ {16, 18, 20}",
        "SpO2": "SpO2 ∈ {98–100}",
    }

    # Worst (max) EMS low/high implausibility flag per vital.
    plaus_by_vital: dict[str, float] = {}
    for r in plaus:
        if r["source"] != "EMS":
            continue
        v = r["vital"]
        plaus_by_vital[v] = max(plaus_by_vital.get(v, 0.0), float(r["percent_flagged"]))

    header = ["EMS vital sign", "Missing", "EMS median (IQR)", "Pearson r vs hospital",
              "Mean bias (95% limits of agreement)", "Default-value clustering",
              "Terminal-digit (0 or 5)", "Implausible (max rule)"]

    order = [("SBP", "Systolic blood pressure, mmHg"),
             ("HR", "Heart rate, beats/min"),
             ("RR", "Respiratory rate, breaths/min"),
             ("SpO2", "Pulse oximetry, %"),
             ("GCS", "Total Glasgow Coma Scale")]

    rows = []
    for key, label in order:
        d = dist[key]
        a = agree[key]
        b = ba[key]
        miss = float(d["percent_missing"]) / 100.0
        median_iqr = f"{float(d['median']):.0f} ({float(d['iqr']):.0f})"
        r_val = f"{float(a['pearson_r']):.2f}"
        bias = float(b["mean_difference"])
        lo = float(b["lower_limit_agreement"])
        hi = float(b["upper_limit_agreement"])
        bias_s = f"{bias:+.2f} ({lo:.1f} to {hi:.1f})"
        if key in default_lut:
            df = default_lut[key]
            default_s = f"{default_text[key]}: {float(df['percent']) / 100 * 100:.1f}%"
        else:
            default_s = "—"
        if key in digit:
            td = pct(digit[key]["prop_ending_0_or_5"])
        else:
            td = "—"
        impl = f"{plaus_by_vital.get(key, 0.0):.1f}%" if key in plaus_by_vital else "—"
        rows.append([label, f"{miss * 100:.1f}%", median_iqr, r_val, bias_s, default_s, td, impl])

    notes = [
        "*Percent missing, default-value clustering, terminal-digit preference, and "
        "implausibility are reported as percentages of EMS documentation; Pearson r is "
        "reported to 2 decimals. Bias and limits of agreement are from Bland-Altman "
        "analysis (EMS minus hospital-arrival value). The implausibility column reports "
        "the most frequently triggered predefined EMS rule for each vital sign.*",
    ]

    write_md(OUT / "table3_ems_vital_fidelity.md",
             "Table 3. EMS vital-sign fidelity relative to hospital-arrival values",
             header, rows, notes)
    write_csv(OUT / "table3_ems_vital_fidelity.csv", header, rows)


# ---------------------------------------------------------------------------
# Supplementary Table - 2021 Field Triage Guideline proxy vs model.
# ---------------------------------------------------------------------------
def build_guideline_table() -> None:
    tier = read_csv(TABLES / "guideline_proxy" / "guideline_proxy_tier_nfti_rates.csv")
    cmp = read_csv(TABLES / "guideline_proxy" / "guideline_proxy_vs_model_threshold_metrics.csv")

    # Panel A - observed NFTI rate by guideline proxy tier.
    a_header = ["Guideline proxy tier", "n", "NFTI-positive rate (95% CI)"]
    a_rows = []
    for r in tier:
        ci = f"{pct(r['observed_nfti_rate'])} ({pct(r['ci95_lower'])}–{pct(r['ci95_upper'])})"
        a_rows.append([r["tier_label"], cnt(r["n"]), ci])

    # Panel B - operating points.
    b_header = ["Rule / model", "Operating point", "Sensitivity", "Specificity",
                "PPV", "NPV", "F1", "AUROC", "AUPRC", "Brier"]
    b_rows = []
    for r in cmp:
        b_rows.append([
            r["rule_or_model"], r["threshold_or_rule"],
            pct(r["sensitivity"]), pct(r["specificity"]), pct(r["ppv"]), pct(r["npv"]),
            dec(r["f1"]), dec(r["auroc"]) if r["auroc"] else "—",
            dec(r["auprc"]) if r["auprc"] else "—", dec(r["brier"]) if r["brier"] else "—",
        ])

    lines = [
        "### Supplementary Table. 2021 Field Triage Guideline proxy vs XGBoost NFTI model (holdout)",
        "",
        "**Panel A. Observed NFTI-positive rate by guideline proxy tier**",
        "",
        "| " + " | ".join(a_header) + " |",
        "| " + " | ".join(["---"] * len(a_header)) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in a_rows]
    lines += [
        "",
        "**Panel B. Operating-point comparison**",
        "",
        "| " + " | ".join(b_header) + " |",
        "| " + " | ".join(["---"] * len(b_header)) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in b_rows]
    lines += [
        "",
        "*Exploratory benchmark using only the subset of 2021 Field Triage Guideline "
        "criteria mappable to prehospital TQIP variables. AUROC/AUPRC/Brier are not "
        "defined for the binary guideline rules. The ordinal guideline-tier proxy "
        "(none / yellow only / red present) achieved an AUROC of 0.661.*",
        "",
    ]
    (OUT / "supplementary_guideline_proxy.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(OUT / "supplementary_guideline_proxy_tiers.csv", a_header, a_rows)
    write_csv(OUT / "supplementary_guideline_proxy_operating_points.csv", b_header, b_rows)


def main() -> None:
    build_table1()
    build_table2()
    build_table3()
    build_guideline_table()
    print(f"Wrote polished tables to: {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
