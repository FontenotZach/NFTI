# TODO / backlog

## Clinical derived features (`src/data/clinical_derived.py`)

- **`rts_ems_weighted`:** Champion-style RTS on scene vitals (`EMSTOTALGCS`, `EMSSBP`, `EMSRESPIRATORYRATE`). Components treated as missing when the corresponding `*_BIU` flag is non-zero. Pediatric GCS/SBP/RR coding differs; **do not use pediatric interpretations without adjustment.**

- **`ems_hypotension` / `ems_tachycardia`:** Adult defaults **SBP < 90** and **HR > 120** (constants in `clinical_derived.py`). Consider age-stratified thresholds for peds or alternate definitions (e.g. shock index only).

## Temporal derived features (`src/data/temporal_derived.py`)

- **Anchor assumption:** Elapsed-minute features (`ems_*_min`, `encounter_month`-adjacent logic) assume all referenced `*DAYS` / `*HRS` fields share the **same calendar anchor** (typically offsets from Injury Incident Date per NTDS-style flat exports). **Validate this against your trauma registry vendor / NTDS submission spec.** If EMS and ED timestamps use different anchors, intervals will be wrong until the formula or normalization is adjusted.

- **`encounter_month`:** Derived only when optional text columns exist and parse successfully:
  - Prefer **`INJURYINCIDENTDATE`**, else **`HOSPITALARRIVALDATE`** (see `header_definitions.csv`). Add these columns to extracts if month seasonality should be modeled; otherwise values stay NaN.

- **Registry support:** `register_temporal_derived_headers` runs in `app.py` / `smoke_check` after loading CSV columns. Paths that build `TraumaDataset` without that step (e.g. JSON-only flows in `TraumaDataset.load_from_json`) do **not** register derived headers unless extended.

## BIU / datetime pairs

- **`review_and_adjust_for_biu`:** Sister-field logic strips `_BIU` to find the primary column (e.g. `EMSNOTIFYDH_BIU` → `EMSNOTIFYDH`). Flat files expose **`EMSNOTIFYHRS` / `EMSNOTIFYDAYS`** instead of a single `EMSNOTIFYDH` column, so BIU adjustment for `*DH_BIU` rows may not apply as intended. Consider mapping BIU to the paired HRS/DAYS fields or introducing composite columns if the registry supports them.

## Feature timing scope

- **Default model pipeline** (`feature_preprocessor`, parts of `app.py`, `ModelGen.py`, `test.py`) filters **`timing in ["1"]`** only (pre-hospital–oriented phase). There are inline `#TODO` markers where stricter timing filters are noted. **Decide** whether ED/hospital timing (`2+`) or derived columns should be included in specific trainers/evaluators and update filters consistently.

## Schema / data hygiene

- **`header_definitions_legacy.csv` / copies:** Keep in sync with `data/schemas/header_definitions.csv` when schema changes, or document as archived snapshots only.

- **Sample data:** `data/samples/dat5_limited.csv` does not include optional `INJURYINCIDENTDATE` / `HOSPITALARRIVALDATE` / `EMSPCRUUID` columns; smoke tests will show NaN for fields that depend on them until samples are updated.

## NTDS / pre-hospital

- **EMS Patient Care Report UUID:** Schema includes `EMSPCRUUID` / `EMSPCRUUID_BIU` per current NTDS; legacy Appendix 4 EMS elements remain optional for local extracts.

## Code cleanup (low priority)

- Consolidate duplicated **`timing in ['1']`** selection logic across `app.py`, `ModelGen.py`, and `test.py` into one helper to avoid drift.
