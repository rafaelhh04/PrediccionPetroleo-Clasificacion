# Usage

How to install the project, get the data, run every command and configure a run.

## Installation and commands

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12 if needed).

```bash
git clone https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion.git
cd PrediccionPetroleo-Clasificacion

uv sync                      # create .venv from uv.lock
uv run brent data download   # fetch the Kaggle dataset into data/raw/ (needs Kaggle credentials)
uv run brent train           # full pipeline: preprocess, tune, train, evaluate (~11 min on 4 cores)
uv run brent report          # statistical report of the last run (CIs, tests, calibration)
uv run brent explain         # permutation importance and SHAP of the trained models
uv run brent registry show   # champion / challenger in the MLflow model registry
uv run brent data ingest     # append recent market data (Yahoo Finance, FRED) to data/live/
uv run brent predict         # P(up) for the next trading day with the champion model
```

**No Kaggle credentials?** `make demo` runs the whole system on a synthetic dataset with the Kaggle
schema (a random walk, so there is nothing to predict): `brent data synthetic` writes it to `data/raw/`,
then training with the fast `configs/demo.yaml` overlay, the statistical report with the champion
promotion and the explanations. `make serve` and `make dashboard` then show the result. Every page and
report labels these numbers as synthetic.

Or run every stage with DVC, which skips the stages whose inputs did not change:

```bash
uv run dvc repro             # download → validate → featurize → train → evaluate → explain
uv run dvc metrics show      # metrics.json, report.json and validation.json
uv run dvc push              # cache the data and models in the DVC remote
```

Outputs are written to `results/` (ignored by git):

| Path | Content |
|------|---------|
| `results/metrics.json` | Evaluation protocol, development-CV AUC, out-of-sample metrics and per-window metrics of every model |
| `results/predictions.csv` | Out-of-sample probability of every model for every test day (with its walk-forward window) |
| `results/models/*.joblib` | Every pipeline refitted on all available data |
| `results/run.log` | Full log of the run |
| `results/plots/` | Out-of-sample ROC, AUC per window, confusion matrices, learning curves, MLP training curves, reliability diagram |
| `results/report.md`, `report.json` | Statistical report written by `brent report` |
| `data/processed/dataset.parquet` | Featurized dataset (`brent featurize`, DVC-tracked) |
| `results/validation.json` | Raw-data validation summary (`brent data validate`) |
| `mlruns/` | MLflow store (SQLite) and artefacts of every tracked run; open with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` |
| `data/live/` | Prices file extended with recent days (`brent data ingest`) |
| `results/prediction.json` | Latest prediction (`brent predict`), with the alias, version and candidate used |
| `results/explain/` | Permutation and SHAP importances (CSV) and `explanations.md`, written by `brent explain` |

## Data

The CSVs are not versioned. `brent data download` downloads
[kavyadhyani/global-oil-prices-andgeopolitical-events](https://www.kaggle.com/datasets/kavyadhyani/global-oil-prices-andgeopolitical-events)
with `kagglehub`, identifies each file by its header, copies it to `data/raw/` under its canonical name and
verifies its SHA-256 against `configs/data_checksums.json`.

| Canonical file | Content |
|----------------|---------|
| `oil_geopolitics_dataset_2010_2026.csv` | ~4,000 daily rows × 23 columns: prices, returns, lags, volatilities, DXY, VIX, GPR, spread, event flags |
| `geopolitical_events_timeline.csv` | 35 events: `date`, `event_type`, `event_description`, `event_severity` (1–10) |

**Credentials:** export `KAGGLE_USERNAME` and `KAGGLE_KEY` (from *Kaggle → Settings → API → Create New Token*)
or place `kaggle.json` in `~/.kaggle/`.

**Manual download:** download the ZIP from the Kaggle page, copy both CSVs to `data/raw/` with the canonical
names above and run `uv run brent data verify`.

**Checksums:** a digest stored as `null` is recorded on first use (trust on first use) and written back to
`configs/data_checksums.json` — commit that file. `brent data download --update-checksums` refreshes them
when the upstream dataset legitimately changes; any other mismatch aborts and removes the copied files.

## CLI

```text
brent [--config PATH ...] [--log-level LEVEL] [--log-format text|json] COMMAND

  featurize        Build the model-ready dataset from the raw files (data/processed/dataset.parquet)
  train            Run the full pipeline: load, preprocess, tune, train and evaluate on test  [--features]
  report           Build the statistical report (CIs, DeLong, binomial, calibration) of the last run
  explain          Explain the trained models: permutation importance and SHAP (global and local)
  registry show    Print the aliases (champion, challenger) of the registered model
  data download    Download the Kaggle dataset and verify checksums  [--force] [--update-checksums]
  data synthetic   Write a synthetic dataset with the Kaggle schema (demo, no credentials)  [--seed] [--force]
  data verify      Verify the SHA-256 checksums of the local data files
  data validate    Validate the raw files and write results/validation.json
  data ingest      Append recent market data to data/live/ (Yahoo Finance, FRED)  [--end DATE]
  predict          Predict the next trading day with a registered model  [--alias] [--live/--snapshot]
  serve            Serve the registered model over HTTP (FastAPI)  [--host] [--port]
  dashboard        Run the Streamlit demo (needs the `dashboard` extra)  [--host] [--port]
  config show      Print the resolved configuration as JSON
```

`python -m brent_forecast` is equivalent to `brent`.

## Configuration

All tunable values — paths, seed, split dates, CV folds, VIF and winsorisation thresholds, fixed
hyperparameters and search grids — live in [`configs/default.yaml`](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/configs/default.yaml) and are validated
with pydantic-settings. Any field can be overridden with a `BRENT_`-prefixed environment variable, using
`__` for nesting:

```bash
BRENT_SEED=7 BRENT_VALIDATION__MODE=rolling uv run brent train
BRENT_MODELS__RANDOM_FOREST__GRID='{"max_depth": [5, 10]}' uv run brent config show
uv run brent --config configs/my_experiment.yaml train
```

Precedence: explicit arguments > environment variables > YAML file. `--config` can be repeated to
layer overlays: later files are deep-merged over earlier ones (mappings key by key, lists replaced), so
[`configs/demo.yaml`](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/configs/demo.yaml) only lists the smaller search spaces of the demo:

```bash
uv run brent -c configs/default.yaml -c configs/demo.yaml train
```
