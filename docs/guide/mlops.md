# Reproducibility and MLOps

Data versioning, experiment tracking, the model registry and live inference.

## Reproducible pipeline (DVC)

`dvc.yaml` declares six stages, each a `brent` command, with their dependencies (data, source modules),
parameters (keys of `configs/default.yaml`) and outputs:

| Stage | Command | Outputs |
|-------|---------|---------|
| `download` | `brent data download` | `data/raw/` |
| `validate` | `brent data validate` | `results/validation.json` (metric) |
| `featurize` | `brent featurize` | `data/processed/dataset.parquet` |
| `train` | `brent train --features data/processed/dataset.parquet` | `results/models/`, `predictions.csv`, `metrics.json` |
| `evaluate` | `brent report` | `report.md`, `report.json` (metric), reliability and equity plots |
| `explain` | `brent explain` | `results/explain/` (permutation and SHAP importances, `explanations.md`) |

- `dvc.lock` (versioned) pins the hash of every input and output. `uv run dvc repro` re-runs only what
  changed, and `uv run dvc params diff` / `uv run dvc metrics diff` compare commits.
- Data, the featurized dataset and models live in the DVC cache, never in git. `uv run dvc push` and
  `uv run dvc pull` sync them with the default remote, a local folder next to the repository
  (`../brent-dvc-storage`).
- To use cloud storage, install the matching extra and add a remote:
  `uv add --dev "dvc[gs]"` then `uv run dvc remote add -d gcs gs://<bucket>/brent`, or `dvc[s3]` with
  `s3://<bucket>/brent`. Credentials come from the provider's usual environment (never committed).
- The `download` stage needs Kaggle access. On a machine without it, place the files in `data/raw/`
  (manual download plus `brent data verify`) and record them with `uv run dvc commit download`.
- `dvc.lock` is machine-independent:
    - `.dvcignore` excludes Python bytecode;
    - the files that depend on the MLflow store (`mlflow_run.json`, `promotion.json`) are not DVC outputs;
    - every model is single-threaded, so predictions are bit-for-bit reproducible.

    A clean clone + `dvc pull` + `dvc repro --force` gives the same lock.

- `tests/test_dvc_pipeline.py` checks without DVC that every stage command, parameter and dependency
  still exists, so a rename fails CI instead of `dvc repro`.

## Experiment tracking and model registry

With `tracking.enabled: true` (the default) every run is recorded in MLflow (`tracking.py`). The store is a
local SQLite database under `mlruns/`; point `tracking.uri` at a tracking server to share runs.

- `brent train` opens a parent run with the protocol (params), git commit/branch and data SHA-256 (tags), and
  `metrics.json`, `predictions.csv`, the resolved configuration and every plot (artefacts). It adds one nested
  run per candidate with its hyperparameters, CV and out-of-sample metrics, AUC per window (as steps) and the
  pipeline refitted on all data. Models are stored with **skops**, not pickle, with an explicit allowlist of
  trusted types.
- `brent report` adds the statistics to the same run and applies the **promotion rule**:
    - the best model by out-of-sample AUC becomes `champion` only if it beats the best baseline significantly
    (Holm-adjusted DeLong and a bootstrap interval above zero);
    - otherwise the best **baseline** is `champion` and the best model is `challenger`.

    Deploying a model that cannot beat a naive rule only adds risk. Both are registered as versions of
    `brent-direction-classifier` with the reason as a tag. `brent report` also writes
    `results/promotion.json`.

- Inference loads a model by alias: `mlflow.sklearn.load_model("models:/brent-direction-classifier@champion")`.

## Live data and prediction

`brent data ingest` (`data/ingest.py`) appends the days after the last available date to a **copy** of the
prices file in `data/live/`. The versioned raw snapshot is never modified.

- **Sources** (configured in the `live` section): Yahoo Finance's chart API for Brent (`BZ=F`), WTI (`CL=F`)
  and the dollar index (`DX-Y.NYB`), and FRED for the VIX (`VIXCLS`).
- **Trading days:** Brent defines them, and the other series are aligned to Brent.
- **Derived columns:** returns in %, lags, 7/30-day volatilities and the spread are recomputed for the new
  rows from `lookback_days` of history. Before appending, `derivation_gap` checks those definitions against
  the snapshot and warns if they differ.
- **Validation and reruns:** the extended file must satisfy the same Pandera contract as the snapshot.
  Re-running is idempotent.
- **Limitations:** there is no live source for the geopolitical risk index (its last value is carried
  forward, so `gpr_change` decays to 0) or for events (new days have none). Futures prices also differ
  slightly from spot quotes.
- **Testing:** all network access goes through an injectable `fetch_text(url)`, so the tests serve recorded
  Yahoo/FRED-shaped responses and never touch the network.

`brent predict` loads `models:/brent-direction-classifier@champion` (or `--alias challenger`). It builds the
features of the most recent day with the training code, but without a label, because that day's outcome is
still unknown. It validates them with `INFERENCE_SCHEMA` and prints and saves `P(up)`:

```json
{"as_of": "2026-03-12", "proba_up": 1.0, "direction": "up", "source": "snapshot",
 "alias": "champion", "version": "7", "candidate": "persistence"}
```

On the synthetic data the champion is the persistence baseline, so it predicts "up" after an up day.
