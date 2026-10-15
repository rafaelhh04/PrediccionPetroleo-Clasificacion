# Cloud, CD and monitoring

GCP infrastructure, continuous deployment, scheduled retraining and drift monitoring.

## Cloud deployment (GCP)

[`infra/`](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/tree/main/infra) describes the whole cloud stack in **Terraform** (Google provider 8.x), sized to stay
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
  documented in [infra/README.md](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/infra/README.md).

## Continuous deployment

**Continuous deployment** ([`cd.yml`](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/.github/workflows/cd.yml)) runs on every merge to `main` that
changes the image. GitHub Actions authenticates through **OIDC and Workload Identity Federation**, with no
keys or long-lived secrets. Each commit is built once and tagged with its SHA. The images are pushed to
Artifact Registry with a **build provenance attestation**, and the same digest is deployed to **staging**
and then, after a manual approval, to **production**. Each rollout starts a no-traffic `candidate`
revision and [smoke-tests](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/scripts/smoke_test.py) it (`/health` = 200 with the model loaded, a prediction,
a 422). Only then does the traffic move. Rollback is a traffic switch to the previous revision
([`rollback.yml`](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/.github/workflows/rollback.yml)).

## Scheduled retraining

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

## Drift monitoring (Evidently)

`brent monitor drift` runs every weekday after the ingestion. It compares the feature rows of the
**last 60 trading days** with the **756 days before them**, using the live data once ingested.
Evidently's `DataDriftPreset` picks a test per column (Kolmogorov-Smirnov at these sizes). The served
model's **P(up)** on both windows is one more column: that is **prediction drift**. Calendar features
are excluded, since they always differ between a short and a long window. An **alert** fires when
`monitoring.drift_share` (50 %) of the features drifted, or when the predictions drifted. It goes to the
notification webhook. The HTML report, a JSON summary and a history line are written to
`results/drift/`, published in the bucket and shown in the dashboard's *Monitoring* tab. Without a
registered model the run reports data drift only. On the synthetic data the price-level features
(VIX, DXY, volatilities) drift and raise an alert, and the predictions do not. Drift is a reason to
look, not a verdict.

```bash
uv run brent monitor drift                 # JSON summary; results/drift/report.html
uv run brent monitor drift --fail-on-alert # exit code 2 on an alert (CI, cron)
```

## Checks without credentials

`make infra-check` runs `terraform fmt`/`validate` and plan-level `terraform test` with mocked providers,
so no credentials are needed. CI adds **tflint**, a **Trivy** IaC scan and **actionlint**, plus
`terraform plan` with a read-only identity once GCP is configured. Step-by-step deployment, the
architecture diagrams and the cost table are in [infra/README.md](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/infra/README.md).
