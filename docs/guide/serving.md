# Serving

The HTTP API, its observability, the demo dashboard and the containers.

## HTTP API

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

## Observability

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

## Demo dashboard (Streamlit)

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

## Containers (Docker)

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
