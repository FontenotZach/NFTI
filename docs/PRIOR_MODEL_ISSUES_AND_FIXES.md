# Prior NFTI model training: issues and fixes

This document summarizes problems in the **previous** modeling approach (roughly pre–pipeline refactor) and how they were addressed in the current codebase. It is meant for methods sections, reviews, and onboarding—not as a running changelog.

---

## 1. Missing data handled as zeros

**What was wrong**

- Binary and continuous features used `fillna(0)` after `pd.to_numeric(..., errors="coerce")`.
- In clinical/prehospital data, **0 often means “measured zero,” not “missing.”** Collapsing missingness into 0 hides informative patterns (including BIU-style missing indicators) and can distort splits for tree models.
- Row construction in `TraumaDataset.add_record` defaulted missing keys to **0** instead of a missing value.

**What we changed**

- Missing input cells are stored as **`np.nan`** where appropriate.
- Training uses a **sklearn `ColumnTransformer` pipeline**: `SimpleImputer` (+ optional **missingness indicators** on numeric columns) and categorical imputation + `OneHotEncoder`, all **fit only on training data** (or inside CV folds).
- XGBoost can still use missing-aware splits on **genuine** `NaN` where imputation is not applied first; the pipeline is the single place that defines the exact treatment.

---

## 2. Data leakage in categorical encoding

**What was wrong**

- `OneHotEncoder` was **fit on all rows** (train + holdout) to get a “stable” vocabulary, then applied to train and test subsets. That lets **holdout category frequencies and rare categories** influence the encoder and inflates reported performance.

**What we changed**

- Encoding lives inside the **fitted pipeline**. Categories are learned **only from training folds** (GridSearchCV / `cross_val_predict`), never from the held-out test set.
- Legacy array-based `preprocess_data_for_criterion` was updated so the encoder is **fit only on training rows** (`~for_testing`), then applied to the requested subset.

---

## 3. Inconsistent train / test definition (double splitting)

**What was wrong**

- Holdout was assigned randomly per record (`random.random() < 0.15`) with an incorrect comment (“5%” vs 15%), **not reproducible** and not aligned across outcomes.
- **`train_test_split`** inside XGBoost training created a **second** random split on top of `for_testing`, so evaluation rows could disagree across criteria and did not match a single fixed holdout.

**What we changed**

- **Single source of truth**: `TraumaRecord.for_testing` is set once by **`assign_for_testing`** (`StratifiedShuffleSplit`, seeded from config), stratifying on a configurable label (default `nfti_positive`).
- Training/evaluation use **`train_df = df[~for_testing]`, `test_df = df[for_testing]`** only—no extra `train_test_split` for the primary holdout.
- Optional **record ID column** in config: assert **no overlap** between train and test IDs when present.

---

## 4. SMOTE and resampling leakage

**What was wrong**

- SMOTE (when enabled) was applied to the **full** feature matrix **before** splitting, so synthetic examples could **bleed into validation/test** folds and exaggerate metrics.

**What we changed**

- SMOTE is optional and **off by default** in config.
- When enabled, it is inserted in an **`imblearn.pipeline.Pipeline`** **before** the classifier so resampling runs **only during training / inside CV folds**, not on held-out scoring data.

---

## 5. Hyperparameter search without a unified pipeline

**What was wrong**

- Grid search targeted the bare `XGBClassifier` while preprocessing lived outside, so **CV did not refit preprocessing per fold** consistently with modern sklearn practice.

**What we changed**

- **`GridSearchCV`** fits a **full pipeline** (preprocess → optional SMOTE → `XGBClassifier`) on **training rows only**, with **stratified** inner CV and the same **minority-class-aware** fold count as elsewhere.

---

## 6. Stratification gaps

**What was wrong**

- `train_test_split` was not stratified on the label; imbalanced NFTI outcomes made single splits unstable.

**What we changed**

- Holdout assignment is **stratified** (`StratifiedShuffleSplit`).
- Inner CV uses **`StratifiedKFold`** with a fold count capped by **minority class count** so stratification remains valid.

---

## 7. Threshold selection on the holdout (evaluation leakage)

**What was wrong**

- After training, **Youden’s J** (and similar) was computed from **`roc_curve(y_holdout, scores_holdout)`**, so the **threshold was tuned using holdout labels**. That is not permissible for a clean “final evaluation only” test set.

**What we changed**

- Threshold is chosen from **training data only**:
  - Prefer **`cross_val_predict`** out-of-fold probabilities on **training rows** with the **same leakage-safe pipeline** (clone per fold), then **Youden J on `(y_train, oof_prob)`**.
  - If stratified K-fold is impossible (too few minority samples), a **stratified train-internal validation split** is used—still **no holdout labels**.
- Holdout reporting uses **fixed** train-derived threshold for discrete metrics; **AUROC, average precision, Brier** remain threshold-free on the holdout.

Metrics JSON includes **`threshold_selection`**: `threshold`, `threshold_policy`, **`threshold_selected_on`** (e.g. `train_cv_oof_predictions` or internal validation fallback).

---

## 8. Debug / ModelGen path still tuning thresholds on holdout

**What was wrong**

- `test_all_testing_records_xgboost` computed **F1-optimal**, **Youden**, and **middle** thresholds from **holdout** predictions. Even “debug only,” that risks reporting holdout-optimized threshold metrics by mistake.

**What we changed**

- That path **never** selects a threshold from holdout labels.
- It loads the **train-saved threshold** from the latest **`xgb_metrics_<criterion>_*.json`**; if missing, uses **0.5** with provenance **`fixed_default_0_5`**, with a **warning**.
- **ROC curve plotting** stays **threshold-free** (rank-based curve only).

---

## 9. Metrics not suited to rare outcomes

**What was wrong**

- Emphasis on accuracy and ad hoc reporting; weak calibration for **imbalanced** clinical prediction.

**What we changed**

- Reports include **AUROC**, **average precision (AUPRC)**, **Brier score**, and at the **frozen train-derived threshold**: balanced accuracy, **MCC**, sensitivity, specificity, **F1**, confusion counts.
- Config drives primary **GridSearchCV** scoring (e.g. average precision).

---

## 10. Configuration tied to the interactive menu

**What was wrong**

- Imputation and training behavior were coupled to menu-driven flows, making runs hard to reproduce.

**What we changed**

- Central **`TrainingConfig`** (`src/config.py`) for split size, seeds, imputation, SMOTE, CV, grids, optional record ID for overlap checks.
- Pickling data can call **`assign_for_testing`** automatically when enabled.

---

## 11. Artifacts and reproducibility

**What was wrong**

- Saving only the raw XGBoost JSON model dropped preprocessing state; inference could not reliably reproduce training transforms.

**What we changed**

- Primary artifact: **`joblib`** dump of the **full fitted pipeline** (preprocess + optional SMOTE + classifier).
- Optional continued export of the booster JSON for interoperability.

---

## Manuscript-ready leakage sentence

The training code documents this sentence (also in config / logs where relevant):

> All preprocessing steps, including imputation, missingness indicators, encoding, resampling, and model fitting, were performed within training folds only to prevent test-set leakage.

Threshold selection is additionally constrained so **holdout labels are not used to pick the operating point** (only for scoring at a **train-derived** or documented default threshold).

---

## Key files touched (for navigation)

| Area | Location |
|------|----------|
| Training config | `src/config.py` |
| Holdout assignment | `src/splitting.py` |
| Feature matrix + legacy preprocessing | `src/preprocessing/feature_preprocessor.py` |
| Pipeline construction | `src/preprocessing/pipeline_factory.py` |
| XGBoost training + metrics JSON | `src/models/xgboost_model.py` |
| Classification metrics / Youden helpers | `src/evaluation/metrics.py` |
| Debug evaluation after training | `src/ModelGen.py` |
| Records / missing ingestion | `src/TraumaDataset.py`, `src/TraumaRecord.py` |
| Ensemble using pipeline inputs | `src/Ensemble.py`, `app.py` (where applicable) |

---

*Last updated to reflect the pipeline refactor, train-only threshold selection, and ModelGen debug alignment.*
