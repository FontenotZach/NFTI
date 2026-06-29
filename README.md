# NFTI: Machine-Learning Prediction of Need for Trauma Intervention from Prehospital Data

Reproducible analysis code for a retrospective national-cohort study that trains
machine-learning models on **prehospital** variables to predict **Need for Trauma
Intervention (NFTI)** status, using the 2020 ACS Trauma Quality Improvement
Program (TQIP) database.

The primary model is an XGBoost classifier benchmarked against logistic
regression, evaluated on an untouched holdout split with discrimination,
calibration, threshold, missingness, vital-sign fidelity, and field-triage
guideline-proxy analyses.

> **Data are not distributed with this repository.** TQIP records are derived
> from real patients and are governed by an ACS data-use agreement. No patient
> data (raw cohorts or samples) are committed on any branch. You must supply your
> own `NFTI/data/raw/dat5.csv` to run the pipeline. Only schema/configuration
> files under `NFTI/data/schemas/` are tracked.

---

## Repository layout

```
.
├─ README.md
├─ .gitignore
└─ NFTI/
   ├─ app.py                  # Interactive pipeline entry point (build → train → analyze)
   ├─ smoke_check.py          # Fast end-to-end sanity check on a small input
   ├─ requirements.txt        # Pinned dependencies (Python 3.12)
   ├─ data/
   │  ├─ schemas/             # TRACKED: header definitions, custom features, mechanism matrix
   │  ├─ raw/                 # IGNORED: place your dat5.csv here (you supply this)
   │  └─ samples/             # IGNORED: optional local sample(s)
   ├─ scripts/                # Standalone CLIs for preprocessing, audits, tables, figures
   ├─ src/                    # Library code
   │  ├─ data/                # Dataset construction, missing-value + human-readable helpers
   │  ├─ preprocessing/       # Cohort filter, feature matrix, transforms, (optional) MICE
   │  ├─ models/              # XGBoost + logistic regression training and SHAP
   │  ├─ evaluation/          # Primary evaluation, fidelity, missingness, guideline proxy
   │  └─ reporting/           # Training-header provenance report
   ├─ artifacts/              # IGNORED: generated models, predictions, logs, intermediate tables
   └─ results/                # IGNORED: manuscript-ready tables and figures
```

`artifacts/` and `results/` are created on demand and are never committed.

---

## Setup

Requires **Python 3.12**.

```bash
cd NFTI
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Run a quick end-to-end sanity check (needs a small CSV at
`data/samples/dat5_limited.csv` or `data/raw/dat5.csv`):

```bash
python smoke_check.py
```

---

## Input data

Place the analytic source CSV at `NFTI/data/raw/dat5.csv`. Columns must match the
TQIP fields described in `data/schemas/header_definitions.csv` (this file
controls which columns are loaded, which are model inputs, and which is the
outcome). Derived features (e.g. shock index) are defined in
`data/schemas/customs.csv` and computed automatically during the build.

**Mechanism / trauma-type labels (one-time preprocessing).** Mechanism and
trauma type are derived from primary ICD-10 external-cause codes using the ACS
"External Cause of Injury Matrix" in `data/schemas/mechanism_matrix.csv`:

```bash
python scripts/apply_mechanism_matrix.py data/raw/dat5.csv
# writes data/raw/dat5_matrix_processed.csv (+ mapping reports)
```

Use the `*_matrix_processed.csv` output as your `dat5.csv` if your source file
does not already contain `MECHANISM` / `TRAUMATYPE` columns.

---

## Reproducible workflow

All steps are deterministic: the train/validation/holdout split uses a fixed
seed (`random_state=42`); the cohort filter and feature construction are
deterministic. The split is ~72% train / ~13% validation / ~15% holdout, applied
**after** the prehospital EMS cohort filter.

Most steps are driven from the interactive menu (`python app.py`); the audits and
figure/table renderers also have standalone CLIs in `scripts/` so the full
analysis can be reproduced non-interactively.

| # | Step | Interactive (`app.py`) | Standalone CLI | Writes |
|---|------|------------------------|----------------|--------|
| 1 | Build analytic cohort (cohort filter + split) and pickle it | `1` | — | `artifacts/pickles/datasets/trauma_dataset.pkl` |
| 2 | Train primary models (XGBoost vs LR) + SHAP | `6` | — | `artifacts/models/`, `artifacts/predictions/`, `artifacts/reports/`, `artifacts/figures/` |
| 3 | EMS vital-sign fidelity audit (run **before** transforms) | `8` | `python scripts/fidelity_audit.py` | `artifacts/figures/fidelity/`, `artifacts/tables/fidelity/` |
| 4 | Missing-data audit (run **after** training) | `9` | `python scripts/missingness_audit.py` | `artifacts/figures/missingness/`, `artifacts/tables/missingness/`, `artifacts/metrics/missingness/` |
| 5 | 2021 field-triage guideline proxy benchmark (run **after** training) | `10` | — | `artifacts/tables/guideline_proxy/`, `artifacts/figures/guideline_proxy/` |
| 6 | Assemble manuscript tables | — | `python scripts/build_manuscript_tables.py` | `results/tables/*.md`, `results/tables/*.csv` |
| 7 | Render figures | — | see below | `results/figures/`, `results/tables/png/` |

### Typical end-to-end run

```bash
cd NFTI

# 0. (one-time) derive mechanism / trauma-type labels from ICD codes
python scripts/apply_mechanism_matrix.py data/raw/dat5.csv

# 1-2. build the cohort + train, via the interactive menu
python app.py        # choose 1 (build), then 6 (train)

# 3-5. audits (fidelity before transforms; missingness + guideline after training)
python scripts/fidelity_audit.py
python scripts/missingness_audit.py
python app.py        # choose 10 (guideline proxy benchmark)

# 6-7. manuscript tables and figures
python scripts/build_manuscript_tables.py
python scripts/render_figure1_flowchart.py
python scripts/render_calibration_decile_figure.py
python scripts/render_tables_png.py
```

### Outputs

- **Models / predictions / reports:** `NFTI/artifacts/` (XGBoost JSON + LR pickle,
  holdout/validation predictions, threshold sweeps, calibration bins, risk
  deciles, model-comparison summaries).
- **Manuscript figures and tables:** `NFTI/results/figures/` and
  `NFTI/results/tables/` (e.g. study-flow diagram, discrimination curves,
  calibration/risk-decile figure, SHAP beeswarm, baseline/operating-point/fidelity
  tables, guideline-proxy supplement).

All generated outputs are git-ignored; regenerate them by re-running the
workflow above.

---

## Menu reference (`app.py`)

```
Data
  1. Build dataset from raw CSV (cohort filter + split, then pickle)
  2. Load pickled dataset
  3. Audit / verify dataset
  4. Set custom-features CSV (optional; default: data/schemas/customs.csv)
  5. Toggle testing mode (loads a 1,000-row subset for fast runs)
Modeling
  6. Train primary NFTI models (XGBoost vs LR) + SHAP
  7. Load saved XGBoost models
Analyses
  8. EMS vital-sign fidelity audit (run before transforms)
  9. Missing-data audit (run after training)
  10. 2021 field-triage guideline proxy benchmark (run after training)
Labels & maintenance
  11. Register model one-hot feature names
  12. Manage human-readable header labels
  13. Reset output artifacts (keeps dataset pickle)
  0. Exit
```

> MICE imputation and the standalone normalize/one-hot transform are available in
> `src/preprocessing/` but are **not** part of the primary training/evaluation
> pipeline (one-hot encoding and scaling are applied on the fly inside model
> training). They are retained for transparency and exploratory use only.

---

## Notes

- **Determinism:** seed `42` throughout; XGBoost hyperparameters are selected via
  3-fold stratified `GridSearchCV` on the validation set optimizing AUROC.
- **No leakage:** features that directly satisfy NFTI (e.g. prehospital blood
  transfusion, intubation) are excluded; race and ethnicity are excluded from
  model training.
- **Privacy:** do not commit anything under `data/raw/` or `data/samples/`. The
  `.gitignore` enforces this; only `data/schemas/` is tracked.
