# Training and testing trial commands

Use this as a checklist for local experiments.

**Where things live:** Dependencies are pinned in **`requirements.txt`** at the repository root (same directory as `app.py`).

If you see **`ERROR: Could not open requirements file`**, you are probably in a folder with no `requirements.txt` (e.g. `docs/`). Run `pwd` and `ls requirements.txt`, then `cd` to repo root.

---

## 1. Environment

Create a virtual environment and install dependencies **once** per machine.

**From repository root** (directory that contains `docs/`, `src/`, `data/`, and `app.py`):

```bash
cd /path/to/your/NFTI-clone    # repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins heavy stacks (TensorFlow, XGBoost, etc.). For a **minimal CPU-only trial**, you can temporarily install only what `smoke_check.py` needs (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `imbalanced-learn`, `tqdm`) if full install is too large—but the interactive app expects Keras/TensorFlow if you use neural-network menus.

---

## 2. Quick automated trial (recommended first run)

Runs the bundled sample CSV, builds the dataset (including custom features and derived temporal/clinical columns registered from `header_definitions.csv`), trains an XGBoost pipeline, and performs basic assertions.

```bash
cd /path/to/your/NFTI-clone         # repo root: contains smoke_check.py
source .venv/bin/activate            # if using venv

python smoke_check.py
```

Artifacts (relative to repo root):

- Metrics JSON under `artifacts/reports/` (`xgb_metrics_nfti_positive_*.json`).
- Other outputs under `artifacts/` per `src/paths.py`.

To train on **more rows** of the sample file, edit `smoke_check.py` (`n_rows=200`) or run a one-off:

```bash
cd /path/to/your/NFTI-clone
python -c "
from pathlib import Path
from smoke_check import build_smoke_dataset
from src.paths import SAMPLES_DATA_DIR
from src.models.xgboost_model import train_xgboost_model
from src.config import TrainingConfig

ds = build_smoke_dataset(str(SAMPLES_DATA_DIR / 'dat5_limited.csv'), n_rows=2000, add_custom=True)
train_xgboost_model(ds, 'nfti_positive', config=TrainingConfig(random_seed=42, test_size=0.15, grid_search=False, cv_folds=3))
print('done')
"
```

---

## 3. Interactive workflow (`app.py`)

Full menu: pickle dataset → train models → imputation → exports (see `README.md` menu map).

```bash
cd /path/to/your/NFTI-clone
source .venv/bin/activate

python app.py
```

Typical sequence:

1. **Pickle the data** — you will be prompted to choose a CSV (your full registry extract).
2. Optionally load **`customs.csv`** path if you use a non-default file (menu item **7**).
3. **Train** via menu item **4** (ModelGen paths).
4. Toggle **testing mode** (menu **8**) when you want a held-out slice while iterating.

> **WSL / headless:** Tkinter file dialogs require a display. On WSL, use WSLg, an X server, or copy your CSV into `data/samples/` and temporarily point smoke/scripts at it instead of the GUI.

---

## 4. Hold-out testing menu (`test.py`)

Loads a **pickled** `TraumaDataset` and a **Keras** model file. Prepare artifacts first (pickle + train/export NN model), then:

```bash
cd /path/to/your/NFTI-clone
source .venv/bin/activate

python test.py
```

Defaults expected by the script (adjust paths inside `test.py` if yours differ):

- Pickled dataset: `artifacts/pickles/datasets/trauma_dataset.pkl`
- Keras model: `artifacts/models/keras/nfti_model.h5`

If those files are missing, complete steps from **`app.py`** (pickle + train/export) or adapt the paths.

---

## 5. Regenerate project docs (optional)

```bash
cd /path/to/your/NFTI-clone
python scripts/generate_docs.py
```

---

## 6. Sanity checks

- **`data/schemas/header_definitions.csv`** must align with your CSV column names for metadata and usage flags.
- **`data/schemas/customs.csv`** — custom features require dependencies present as dataset columns.
- Derived columns (**temporal / clinical**) are registered at runtime; they do not need to appear in the raw CSV.

---

## 7. Troubleshooting

| Issue | What to try |
|--------|--------------|
| `ModuleNotFoundError: src` | Run Python from the repository root (the one with `src/`), not from a subfolder like `docs/`. |
| `Could not open requirements file` | Run `pip install -r requirements.txt` from repo root — see §1. Not from `docs/` or other subfolders. |
| `No matching distribution found for tensorflow-intel` | **Expected on Linux/WSL/macOS** — that package targets limited platforms. The lockfile uses **`tensorflow` only**; reinstall from an updated `requirements.txt`. Optional on Windows: `pip install tensorflow-intel==2.17.0` after the main install. |
| Smoke check asserts threshold / OOF | Increase `n_rows` or ensure both outcome classes exist in the sample. |
| GUI file picker fails | Use WSLg/X11 or run programmatic training via `smoke_check` / a small Python script with an explicit CSV path. |
