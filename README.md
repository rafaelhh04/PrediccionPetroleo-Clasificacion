# Brent Direction Classifier

> Next-day direction of the Brent crude oil price from market indicators and geopolitical events (2010–2026).

[![CI](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rafaelhh04/PrediccionPetroleo-Clasificacion/branch/main/graph/badge.svg)](https://codecov.io/gh/rafaelhh04/PrediccionPetroleo-Clasificacion)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/deps-uv-261230.svg)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/lint-ruff-d7ff64.svg)](https://docs.astral.sh/ruff/)
[![mypy strict](https://img.shields.io/badge/types-mypy%20strict-2a6db2.svg)](https://mypy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Versión en español: [README.es.md](README.es.md).*

Binary classification of whether Brent will close higher tomorrow, comparing five models — logistic
regression, RBF SVM, random forest, LightGBM (tuned with Optuna) and a **multilayer perceptron written from
scratch in NumPy**
(backpropagation, inverted dropout, He initialisation, early stopping) — under a strict, leakage-aware
chronological evaluation.

| Class | Condition | Meaning |
|-------|-----------|---------|
| `1` | `brent_return(t+1) > 0` | Price goes up |
| `0` | `brent_return(t+1) ≤ 0` | Price goes down or stays flat |

---

## Quickstart

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

### Data

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

---

## CLI

```text
brent [--config PATH] [--log-level LEVEL] COMMAND

  featurize        Build the model-ready dataset from the raw files (data/processed/dataset.parquet)
  train            Run the full pipeline: load, preprocess, tune, train and evaluate on test  [--features]
  report           Build the statistical report (CIs, DeLong, binomial, calibration) of the last run
  explain          Explain the trained models: permutation importance and SHAP (global and local)
  registry show    Print the aliases (champion, challenger) of the registered model
  data download    Download the Kaggle dataset and verify checksums  [--force] [--update-checksums]
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
hyperparameters and search grids — live in [`configs/default.yaml`](configs/default.yaml) and are validated
with pydantic-settings. Any field can be overridden with a `BRENT_`-prefixed environment variable, using
`__` for nesting:

```bash
BRENT_SEED=7 BRENT_VALIDATION__MODE=rolling uv run brent train
BRENT_MODELS__RANDOM_FOREST__GRID='{"max_depth": [5, 10]}' uv run brent config show
uv run brent --config configs/my_experiment.yaml train
```

Precedence: explicit arguments > environment variables > YAML file.

---

## Methodology

```text
load + left-join events ─► label (t+1) ─► feature engineering (past-only) ─► drop warm-up rows
   ─► per model, a scikit-learn Pipeline:
        VIFSelector ─► Winsorizer (return features, 1–99 %) ─► StandardScaler ─► model
   ─► development period (< 2024-01-01): grid search with purged walk-forward CV (5 folds)
   ─► out-of-sample period (≥ 2024-01-01): walk-forward, refit every 63 trading days on all older data
```

- **Purged walk-forward validation** (`validation/walk_forward.py`): every model is trained only on rows
  strictly older than its test window, with a gap of `purge` (1 day: day *t*'s label is realised on *t+1*,
  López de Prado) + `embargo` (5 days of buffer, since features use rolling windows). Expanding window by
  default, rolling with `validation.mode: rolling`. The same splitter drives tuning, learning curves and the
  out-of-sample evaluation.
- **Leakage controls:** same-day `wti_return` removed; VIF selection, winsorisation percentiles and scaler are
  steps of each model `Pipeline`, refitted on the training rows of every fold and window; tests prove that
  corrupting the future never changes earlier features, training labels or predictions.
- **Features** (all computed from data up to day *t*; one truncation test per feature proves it):
  - returns and risk: log returns over 1/3/7 days, 21- and 63-day momentum, 7/30-day volatility and their
    ratio, Brent–WTI spread;
  - technical indicators (`features/technical.py`): RSI(14) with Wilder smoothing, MACD(12, 26, 9) line and
    histogram divided by the price, Bollinger(20, 2) %B and bandwidth;
  - market context: 1- and 5-day log changes of the VIX and the dollar index (DXY), plus their levels;
  - geopolitics: 21-day change of the risk index, event and high-severity flags;
  - calendar: sine/cosine of weekday and month.

  The VIF > 10 filter then removes collinear features inside each training window.
- **Metrics:** accuracy, macro precision / recall / F1 and ROC AUC (primary, used for model selection),
  over the pooled out-of-sample predictions and per walk-forward window (stability over time).

### Models

| Model | Implementation | Search space |
|-------|----------------|--------------|
| Logistic regression (L2) | scikit-learn | `C` |
| SVM, RBF kernel (sigmoid-calibrated probabilities) | scikit-learn | `C`, `gamma` |
| Random forest | scikit-learn | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **MLP 64 → 32 → 1** | **pure NumPy core**, scikit-learn estimator API (`NumpyMLPClassifier`) | hidden sizes, learning rate |
| LightGBM (gradient boosting) | `lightgbm` | 8 hyperparameters with **Optuna** |

LightGBM is tuned with Optuna's TPE sampler (seeded, 50 trials, `tuning` section of the config) instead of a
grid. Each trial is scored with the same purged walk-forward folds; with median pruning a trial whose running
mean AUC falls below the median of earlier trials at the same fold is stopped early. All trials are saved to
`results/optuna_lightgbm.csv`.


### Baselines

Every model must beat naive baselines that go through exactly the same walk-forward evaluation
(`models/baselines.py`):

| Baseline | Prediction |
|----------|------------|
| Majority class | constant `P(up)` = training frequency of up days |
| Persistence | tomorrow repeats today (up if today's log return > 0) |
| Stratified random | random guesses with the training class frequencies |
| Buy & hold | always up (in the backtest: always long) |

A constant score has AUC 0.5 within a window; pooled over windows the majority baseline can deviate from 0.5
because its constant is re-estimated at every refit.

### Statistical evaluation

`brent report` reads `predictions.csv` and `metrics.json` (no retraining) and writes `results/report.md`
(`evaluation/statistics.py`, `evaluation/report.py`):

- **Confidence intervals** for AUC and accuracy with a circular **block bootstrap** (blocks of 21 days,
  2000 resamples): daily outcomes are serially dependent, so resampling single days would understate the
  uncertainty. All candidates are resampled on the same days (paired).
- **DeLong test** of each model's AUC against the reference baseline (the baseline with the highest
  out-of-sample AUC), with **Holm** correction for the four comparisons. A model is called significantly
  better only if the Holm-adjusted p-value is below α = 0.05 *and* the block-bootstrap interval of the AUC
  difference is above zero (DeLong assumes independent days; the bootstrap does not).
- **Binomial test** of accuracy against the no-information rate (always predicting the majority class).
- **Calibration**: Brier score, Brier skill score against the majority-class (climatological) forecast and
  a reliability diagram.
- **Economic backtest** (`evaluation/backtest.py`): each candidate becomes a long/flat strategy (long when
  `P(up) >= 0.5`, optionally long/short) on the next day's return, paying 5 bps per unit of turnover. The
  report gives CAGR, volatility, Sharpe ratio, maximum drawdown, hit ratio, exposure and number of trades,
  and a paired block-bootstrap interval of the Sharpe difference against buy & hold. Settings live in the
  `backtest` section of the config.
- **Provenance**: the SHA-256 of the input files is stored in `metrics.json`; the report says whether
  they match the recorded Kaggle checksums or are unverified.

### Data contracts (Pandera)

Every data boundary has a Pandera schema (`data/schemas.py`), validated lazily so that one error message
lists every failed check:

| Schema | Where | Main checks |
|--------|-------|-------------|
| `OIL_SCHEMA` | prices CSV, on load | required columns, numeric coercion, unique dates, Brent in (0, 1000) USD, Brent return in (−100, 100) %, VIX in (0, 200], DXY in [50, 200], non-negative volatility and GPR |
| `EVENTS_SCHEMA` | events CSV, on load | unique dates (a duplicate would duplicate rows in the join), severity 0–10 |
| `merged_schema` | after the join | dates unique and increasing, one row per oil day, events filled, date range from `data.expected_*` |
| `FEATURES_SCHEMA` | `build_dataset`, `read_features` | label in {0, 1}, every feature numeric, non-null and finite, dates increasing |
| `INFERENCE_SCHEMA` | rows sent to a model | every feature numeric, non-null and finite |

The bounds are loose plausibility checks: they catch unit errors, wrong files and corrupted values, not
market moves. WTI is not range-checked because it settled at −37 USD on 2020-04-20. Invalid data stops the
run with `DataValidationError`, for example:

```text
oil prices failed validation:
  - column 'brent_price' failed in_range(0.0, 1000.0) for 2 row(s), e.g. -1.0, -2.0
  - missing required column(s): 'gpr_index'
```

### Reproducible pipeline (DVC)

`dvc.yaml` declares six stages, each a `brent` command, with their dependencies (data, source modules),
parameters (keys of `configs/default.yaml`) and outputs:

| Stage | Command | Outputs |
|-------|---------|---------|
| `download` | `brent data download` | `data/raw/` |
| `validate` | `brent data validate` | `results/validation.json` (metric) |
| `featurize` | `brent featurize` | `data/processed/dataset.parquet` |
| `train` | `brent train --features data/processed/dataset.parquet` | `results/models/`, `predictions.csv`, `metrics.json` |
| `evaluate` | `brent report` | `report.md`, `report.json` (metric), reliability and equity plots |
| `explain` | `brent explain` | `data/live/` | Prices file extended with recent days (`brent data ingest`) |
| `results/prediction.json` | Latest prediction (`brent predict`), with the alias, version and candidate used |
| `results/explain/` |

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

### Experiment tracking and model registry

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

### Live data and prediction

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

### Serving (HTTP API)

`brent serve` (`api/`, FastAPI + Pydantic v2) serves the same registered model over HTTP:

```bash
uv run brent serve --host 127.0.0.1 --port 8000   # interactive docs at http://127.0.0.1:8000/docs
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict            # the latest available day
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"features": {"dxy_index": 103.2, "vix": 17.4, "...": 0.0}, "as_of": "2026-03-12"}'
```

| Endpoint | Purpose | Codes |
|----------|---------|-------|
| `GET /health` | Readiness: the model is loaded (`model_loaded`, `version`, or why not in `detail`) | 200 / 503 |
| `GET /health/live` | Liveness: the process answers | 200 |
| `GET /model-info` | Alias, version, candidate, kind, promotion reason, training run, threshold and the **expected feature names** (input contract) | 200 / 503 |
| `POST /predict` | `P(up)` and direction for the body's `features`, or for the latest day when the body is empty | 200 / 422 / 503 |

- **Model loading.** At start-up (FastAPI lifespan) the service loads `models:/<registered_model>@<alias>`
  (`serving.alias`, `champion` by default; `BRENT_SERVING__ALIAS=challenger` switches it) together with its
  registry tags and the column names of its MLflow signature. If the registry is empty the API still starts,
  answers 503 and retries at most every `serving.reload_interval_seconds`.
- **Input validation.** Pydantic rejects strings, booleans, NaN/infinity and unknown body fields; the
  feature names must match the model's signature exactly (missing → `missing`, unknown → `extra_forbidden`);
  the row is then checked with `INFERENCE_SCHEMA`. All errors are 422 with FastAPI's
  `{"detail": [{"loc", "msg", "type"}]}` format (the offending input is not echoed).
- **One prediction path.** The API and `brent predict` both call `predict.predict_features`. For the latest
  day the API reuses `latest_features` and caches the row until a data file changes.
- **Response.** `{"as_of", "proba_up", "direction", "threshold", "source", "model": {"alias", "version",
  "candidate"}}`, where `source` is `live`, `snapshot` or `request`.

**Tests.** `tests/test_api.py` covers the endpoints and errors with an in-memory model.
`tests/test_api_registry.py` serves the champion and the challenger from a real MLflow registry in a
temporary directory (SQLite, no network). `tests/test_api_contract.py` compares the OpenAPI schema with the
reviewed snapshot `tests/snapshots/openapi.json`: any change to a path, status code or field fails until the
snapshot is regenerated on purpose (`UPDATE_SNAPSHOTS=1 uv run pytest tests/test_api_contract.py`).

**Load test** (Locust, `tests/load/locustfile.py`, outside the pytest suite: `make serve` then
`make load-test`). The mix is 4:4:1:1:1 latest-day predictions, client-feature predictions, `/health`,
`/model-info` and invalid requests (which must be 422). Measured with a single uvicorn process serving the
champion (persistence baseline) on the synthetic data, with Locust on the same 4-core machine:

| Profile | Throughput | Failures | p50 | p95 | p99 | p95 `POST /predict` |
|---------|-----------:|---------:|----:|----:|----:|--------------------:|
| Nominal: 10 users, 0.5–1.5 s think time | 9.9 req/s | 0 % | 28 ms | **72 ms** | 120 ms | 76–99 ms |
| Saturation: 20 users, 0.05–0.2 s think time | 38.7 req/s | 0 % | 380 ms | 790 ms | 1100 ms | 840–850 ms |

An isolated prediction takes 25–40 ms, and about 22 ms of that is the Pandera validation of the row. One
process saturates at about 40 req/s (the GIL), so latency beyond that is queueing. Scale out with more
processes or replicas rather than threads.

### Observability

`api/observability.py` adds three things to the service:

- **Request id.** Every request gets an id: the client's `X-Request-ID` when it is a safe token (at most
  128 characters from `[A-Za-z0-9._:-]`), a new UUID otherwise. The response returns it in the same header,
  and every log record emitted while serving the request carries it, including the data loading and model
  logs.
- **Structured logs.** `--log-format json` (or `BRENT_LOG_FORMAT=json`, the default in the image) writes one
  JSON object per line with `time`, `level`, `logger`, `message`, `request_id` and any `extra` fields. The
  API writes one access record per request, `{"method", "path", "status", "duration_ms"}`, instead of
  uvicorn's access log. Probes (`/health`, `/metrics`) are logged at DEBUG.
- **Prometheus metrics** at `GET /metrics` (outside the OpenAPI contract). HTTP metrics by handler, method
  and status come from `prometheus-fastapi-instrumentator`. The service adds its own:

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `brent_predictions_total` | counter | `alias`, `direction`, `source` | predictions served (`up` / `down/flat`; `live`, `snapshot`, `request`) |
| `brent_model_inference_seconds` | histogram | `alias` | model latency: input validation + `predict_proba` |
| `brent_model_loaded` | gauge | `alias` | 1 when a model is loaded |
| `brent_model_info` | gauge (1) | `alias`, `version`, `candidate`, `kind` | identity of the served model (spot a baseline champion) |
| `brent_model_load_failures_total` | counter | `alias` | failed attempts to load the model |

```json
{"time": "2026-10-13T15:11:18.210+00:00", "level": "INFO", "logger": "brent_forecast.api.access",
 "message": "POST /predict 200", "method": "POST", "path": "/predict", "status": 200,
 "duration_ms": 24.1, "request_id": "abc-123"}
```

Each app instance owns its metrics registry: one process per container. With several uvicorn workers,
Prometheus' multiprocess mode would be needed.

### Demo dashboard (Streamlit)

`brent dashboard` (`dashboard/`, the `dashboard` extra: `uv sync --extra dashboard`) opens a one-page demo
on `:8501`:

- **Data provenance banner.** It shows *real* only when the training data matched the committed Kaggle
  checksums (`metrics.json`, `matches_recorded_checksums`). Otherwise it warns that the results are
  synthetic, or that the files differ from the published dataset.
- **Prediction for the next trading day,** requested from the API (`dashboard.api_url`). It shows the
  probability, the direction, the last observed day and the served version and candidate, with a notice
  when the champion is a naive baseline.
- **Tabs:** price history (Brent and WTI, 1/3/5 years or all), the statistical report (conclusions, AUC with
  CIs, ΔAUC, Holm p-values, ROC and reliability figures), the backtest after costs (Sharpe, CAGR, drawdown,
  equity curves) and the explanations (top mean |SHAP| features, beeswarm and waterfall per model).

Every section reads the artefacts in `results/` and names the command that creates them when they are
missing. The page also works without the API: it shows an error and keeps the other sections. The loaders
are plain functions (`dashboard/data.py`); the page is tested headlessly with Streamlit's `AppTest`.

### Containers (Docker)

`Dockerfile` (multi-stage): a `builder` stage installs the locked **runtime** dependencies with uv
(`uv sync --frozen --no-dev`, no dev or load-test groups) and the project as a regular package. The
`runtime` stage (`python:3.12-slim-trixie`) copies only that virtual environment and `configs/`. It runs as
the unprivileged user `brent` (uid 10001) and declares a `HEALTHCHECK` on `/health`. `.dockerignore` keeps
the build context to `pyproject.toml`, `uv.lock`, `src/`, `configs/`, `README.md` and `LICENSE`, so no data,
results or credentials can reach the image.

`docker-compose.yml` runs the API (`:8000`), the Streamlit dashboard (`:8501`, image target `dashboard`)
and an MLflow tracking server with the UI (`:5000`) on the same registry. A one-off `trainer` service
(profile `train`) runs `brent train && brent report && brent explain`:

```bash
uv run brent data download                          # data/ is mounted read-only into the containers
docker compose build
docker compose --profile train run --rm trainer     # ~13 min: train, report + champion, explanations
docker compose up -d                                # API :8000/docs, dashboard :8501, MLflow :5000
curl -s localhost:8000/health                       # {"status": "ok", "model_loaded": true, ...}
```

- **Volumes.** `mlruns/` (SQLite store + artefacts) and `results/` are named volumes mounted at the same
  path (`/app/...`) in every container, so the artefact locations recorded by the trainer resolve in the API
  and in the MLflow server. A registry created on the host records host paths, so create it inside compose.
- **Readiness.** Before the first `brent report` the API is up but answers 503. The health probe keeps
  retrying, so it picks up the champion without a restart.
- **Settings.** `BRENT_*` variables work as everywhere else, e.g. `BRENT_SERVING__ALIAS=challenger docker
  compose up -d`.
- **CI.** The `docker` job builds the image, validates the compose file and smoke-tests the container
  (non-root user, liveness 200, readiness 503 without a registry). It then scans the image with **Trivy**:
  a HIGH/CRITICAL report for information, and a gate that fails on any CRITICAL vulnerability with a fix
  available. Dependabot also tracks the base images.
- **Two images, one Dockerfile.** `docker build .` produces the API image (`runtime`, the last stage);
  `--target dashboard` adds the `dashboard` extra (Streamlit) on the same base, so the API image does not
  carry it.
- **Size.** The API image is about 390 MB compressed (1.7 GB on disk) because the runtime dependencies include
  the training stack (MLflow server, SHAP/numba, LightGBM, SciPy). Splitting them into serving and training
  extras is a known follow-up.

### Cloud deployment (GCP)

[`infra/`](infra) describes the whole cloud stack in **Terraform** (Google provider 8.x), sized to stay
at **~0 €/month** for a demo:

- **Cloud Run**: the API and the dashboard per environment (`staging`, `prod`), plus a private MLflow UI.
  All scale to zero with request-based billing. The API runs 1 vCPU / 1 GiB with concurrency 8,
  sized from the Locust p95 (72 ms) and the ~40 req/s ceiling of one process.
- **Cloud Storage**: one versioned bucket with a lifecycle policy. `data/`, `mlruns/` (MLflow store and
  artefacts) and `results/` are mounted with GCS FUSE at the same paths as in compose, read-only in the API
  and the dashboard. `dvc/` is the DVC remote.
- **Artifact Registry** (immutable tags, cleanup policy) and **Secret Manager**. One **service account**
  per workload, granted roles on the bucket and on each secret only. The Terraform state is in GCS,
  with locking.
- **MLflow backend**: SQLite on the bucket by default (free, one writer at a time) or **Cloud SQL**
  PostgreSQL with one variable (`mlflow_backend = "cloudsql"`, ~9–10 USD/month). The trade-off is
  documented in [infra/README.md](infra/README.md).

**Continuous deployment** ([`cd.yml`](.github/workflows/cd.yml)) runs on every merge to `main` that
changes the image. GitHub Actions authenticates through **OIDC and Workload Identity Federation**, with no
keys or long-lived secrets. Each commit is built once and tagged with its SHA. The images are pushed to
Artifact Registry with a **build provenance attestation**, and the same digest is deployed to **staging**
and then, after a manual approval, to **production**. Each rollout starts a no-traffic `candidate`
revision and [smoke-tests](scripts/smoke_test.py) it (`/health` = 200 with the model loaded, a prediction,
a 422). Only then does the traffic move. Rollback is a traffic switch to the previous revision
([`rollback.yml`](.github/workflows/rollback.yml)).

**Scheduled retraining** runs as Cloud Run Jobs started by Cloud Scheduler: `brent data ingest` on
weekdays, and `brent retrain` weekly. A retraining run ingests the new days, then trains on the snapshot
plus those days. It reports, applies the **existing promotion rule** (a model replaces the baseline
only if it beats it with significance) and explains the models. It is **idempotent**: it is skipped when
the data, the training configuration and the code version are those of the last successful run. It
works on a local copy of the SQLite registry and publishes it only on success (`--stage-store`), then
sends the outcome to a chat webhook (`notify.webhook_url`, from Secret Manager). The API checks its
alias every `serving.refresh_interval_seconds` and loads a new version in the background, without a
restart (`brent_model_refreshes_total`).

```bash
uv run brent retrain --no-ingest           # same as the job, on local data; JSON outcome
uv run brent retrain --force               # retrain even if nothing changed
```

`make infra-check` runs `terraform fmt`/`validate` and plan-level `terraform test` with mocked providers,
so no credentials are needed. CI adds **tflint**, a **Trivy** IaC scan and **actionlint**, plus
`terraform plan` with a read-only identity once GCP is configured. Step-by-step deployment, the
architecture diagrams and the cost table are in [infra/README.md](infra/README.md).

### Explainability

`brent explain` (`evaluation/explain.py`) refits each model listed in `explain.models` on the development
period, with the hyperparameters of its saved pipeline, and explains it on the out-of-sample period. The
importances therefore describe what generalises, not what was memorised.

- **Permutation importance**: the drop in out-of-sample ROC AUC when each raw feature is shuffled
  (10 repeats, mean ± std). It is model-agnostic, and a feature removed by the VIF filter scores exactly 0.
- **SHAP**: exact TreeSHAP for LightGBM and the random forest, the closed form for logistic regression
  (log-odds), and the permutation explainer on `P(up)` for the SVM and the MLP. The global view is a
  beeswarm plot plus mean |SHAP|; the local view is a waterfall plot of the most recent out-of-sample day.
  Tests check additivity (base value + Σ SHAP = model output).

On the synthetic data the top permutation importances are within about 0.01–0.015 AUC with comparable
standard deviations, and the features ranked first differ between models. That pattern is what an absence
of stable signal looks like.

---

## Results

> **Which numbers are real?** Only the first table, an old run of the original code on the Kaggle data.
> Every table after it comes from the **synthetic** dataset (a random walk with the Kaggle schema) and
> demonstrates the evaluation machinery, not the Brent market. The [model card](docs/MODEL_CARD.md)
> gathers intended use, metrics with intervals, limitations and risks.

Reference run of the original project on the Kaggle data (ROC AUC). It predates the phase-3 fixes
(e.g. the last-row labelling bug) and has not been re-run on real data yet: Kaggle is not reachable from
the development environment, so later numbers are reproduced on the synthetic dataset used in tests.

| Model | Train | Validation | Test |
|-------|------:|-----------:|-----:|
| Logistic Regression | 0.5507 | 0.4800 | 0.4858 |
| SVM (RBF) | 0.5646 | 0.4700 | 0.4909 |
| Random Forest | 0.7342 | 0.4936 | 0.4927 |
| MLP NumPy | 0.5717 | 0.4933 | 0.5002 |

No model beats a coin flip out of sample — consistent with the weak-form efficient-market hypothesis for
daily returns.

**Statistical report on the synthetic dataset** (not real data; out of sample 2024-01-01 to 2026-03-11,
573 days, 95 % block-bootstrap intervals, reference baseline: Persistence, AUC 0.5073), with the v2
features:

| Model | OOS AUC [95 % CI] | ΔAUC vs Persistence [95 % CI] | p (Holm) | Brier skill |
|-------|-------------------|-------------------------------|---------:|------------:|
| Logistic Regression | 0.536 [0.489, 0.582] | +0.028 [−0.026, +0.082] | 1.000 | +0.0027 |
| SVM (RBF) | 0.491 [0.442, 0.543] | −0.016 [−0.075, +0.046] | 1.000 | −0.0006 |
| Random Forest | 0.510 [0.465, 0.552] | +0.003 [−0.052, +0.057] | 1.000 | −0.0053 |
| MLP NumPy | 0.501 [0.453, 0.549] | −0.007 [−0.069, +0.056] | 1.000 | −0.0073 |
| LightGBM | 0.450 [0.401, 0.497] | −0.057 [−0.116, +0.002] | 0.465 | −0.1245 |

Backtest on the same days (long/flat, 5 bps per trade):

| Strategy | CAGR | Sharpe | ΔSharpe vs buy & hold [95 % CI] | Max drawdown | Exposure |
|----------|-----:|-------:|---------------------------------|-------------:|---------:|
| Buy & hold | +32.8 % | 1.06 | — | −34.4 % | 100 % |
| Logistic Regression | +20.6 % | 1.11 | +0.06 [−1.09, +1.18] | −19.6 % | 35 % |
| SVM (RBF) | +5.0 % | 0.79 | −0.27 [−1.84, +1.21] | −4.4 % | 3 % |
| Random Forest | −6.9 % | −0.27 | −1.33 [−2.52, −0.21] | −28.6 % | 39 % |
| MLP NumPy | +19.8 % | 1.00 | −0.06 [−1.33, +1.15] | −28.1 % | 43 % |
| LightGBM | −7.0 % | −0.24 | −1.30 [−2.60, −0.07] | −36.8 % | 47 % |
| Persistence (baseline) | +19.7 % | 0.90 | −0.16 [−1.04, +0.72] | −23.3 % | 48 % |

Every AUC interval contains 0.5, no model beats persistence or the no-information rate and no strategy
beats buy & hold after costs: as expected on a random-walk price series, there is no skill to find. The
logistic regression's AUC of 0.536 and Sharpe of 1.11 look attractive in isolation; the intervals show they
are compatible with luck.

LightGBM illustrates the opposite trap. Its best Optuna trial reached a development CV AUC of 0.517, the
maximum of 50 noisy estimates (winner's curse), and it fell to 0.450 out of sample, with overconfident
probabilities (Brier skill −0.12). The most flexible model, tuned hardest, is the one that overfits the noise
most.

Effect of the v2 features (same protocol; v1 = 14 features, v2 = 29 before the VIF filter):

| Model | Dev CV AUC v1 → v2 | OOS AUC v1 → v2 | Sharpe v1 → v2 |
|-------|-------------------|-----------------|----------------|
| Logistic Regression | 0.501 → 0.505 | 0.508 → 0.536 | 0.10 → 1.11 |
| SVM (RBF) | 0.500 → 0.511 | 0.492 → 0.491 | 0.00 → 0.79 |
| Random Forest | 0.491 → 0.508 | 0.473 → 0.510 | −0.57 → −0.27 |
| MLP NumPy | 0.505 → 0.507 | 0.467 → 0.501 | 0.26 → 1.00 |

---

## Project structure

```text
├── configs/
│   ├── default.yaml            # single source of configuration values
│   └── data_checksums.json     # SHA-256 of the raw CSVs
├── docs/ROADMAP.md             # professionalisation plan and progress log
├── docs/MODEL_CARD.md          # intended use, data, metrics with CIs, limitations and risks
├── src/brent_forecast/
│   ├── cli.py                  # Typer CLI (`brent`)
│   ├── config.py               # typed settings (pydantic-settings)
│   ├── logging_config.py       # console + per-run file logging
│   ├── pipeline.py             # end-to-end training orchestration
│   ├── predict.py              # next-day prediction (shared by `brent predict` and the API)
│   ├── api/                    # FastAPI inference service (`brent serve`)
│   ├── dashboard/              # Streamlit demo (`brent dashboard`)
│   ├── data/                   # Kaggle download + checksums, loading and merge
│   ├── features/               # label, past-only features, split; VIFSelector / Winsorizer transformers
│   ├── models/                 # registry of model pipelines (incl. LightGBM), NumPy MLP, baselines, grid and Optuna tuning
│   └── evaluation/             # metrics, plots, learning curves, final test evaluation
├── tests/                      # unit, leakage-guard, gradient-check and property tests
├── .github/                    # CI workflow, Dependabot, templates, CODEOWNERS, ruleset
├── Dockerfile / .dockerignore  # multi-stage inference image (uv, non-root, HEALTHCHECK)
├── docker-compose.yml          # API + dashboard + MLflow server (+ one-off trainer)
├── infra/                      # Terraform for GCP: bootstrap (state, registry, WIF) + Cloud Run stack
├── scripts/smoke_test.py       # post-deployment check used by the CD workflow
├── Makefile
└── pyproject.toml / uv.lock
```

## Development

```bash
make install     # uv sync --frozen + pre-commit install
make lint        # all pre-commit hooks: ruff, codespell, mypy --strict, file hygiene
make test        # unit tests (~30 s; `make test-all` adds the slow end-to-end ones)
make coverage    # branch coverage, fails under 85 %
make help        # every target
```

Quality gates: ~130 tests on synthetic data (no real data, no network), data-leakage guards, a numerical
gradient check of the NumPy MLP and property-based tests (Hypothesis). CI runs lint, `mypy --strict` and the
tests on Python 3.12 and 3.13 for every pull request.

Conventions: [Conventional Commits](https://www.conventionalcommits.org/), one branch per feature, English
for code, docstrings (numpy style), logs and commits. Random seed `42` everywhere (`seed` in the config).
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Known limitations

- **Synthetic results.** Every number after the first results table comes from a synthetic dataset. The
  development environment cannot reach Kaggle, Yahoo Finance or FRED, so the pipeline still has to be run
  on the real data (see the [model card](docs/MODEL_CARD.md)).
- **Live data.** There is no live source for the geopolitical risk index or the events: the last GPR value
  is carried forward and new days have no event. Futures closes stand in for spot prices.
- **Derived columns.** The recomputed returns, lags and volatilities assume the Kaggle definitions used by
  the synthetic data. `brent data ingest` warns if the snapshot disagrees.
- **Backtest.** It ignores the futures roll, financing and variable slippage.
- **Runtime.** `brent train` takes about 12 minutes on 4 cores; `dvc repro` skips it when nothing changed.

## Author

Rafael Hernando Herias — originally an individual Machine Learning course project (2025-26).

## License

[MIT](LICENSE)
