# ADR 0004 — MLflow backend: SQLite in the bucket by default, Cloud SQL as an option

- **Status:** accepted
- **Phase:** 4 (tracking), 6 (cloud)

## Context

MLflow needs a backend store for runs and the model registry. Locally it is a SQLite file under
`mlruns/`. In the cloud the obvious choice, a managed PostgreSQL, costs about 9–10 USD per month even
when idle, while everything else in the stack scales to zero. The project has a single writer: one
scheduled training job per week.

## Decision

- By default, keep `mlruns/mlflow.db` (SQLite) in the GCS bucket, mounted with GCS FUSE
  ([ADR 0005](0005-gcs-fuse-shared-paths.md)).
- Enforce **one writer at a time**:
    - the MLflow UI service is capped at one instance;
    - the retraining job works on a local copy (`brent retrain --stage-store`) and publishes it only
    after a successful run, with SQLite's online backup API.
- Offer **Cloud SQL PostgreSQL** behind one Terraform variable (`mlflow_backend = "cloudsql"`). The
  connection URI is kept in Secret Manager, and the driver ships in the `postgres` extra.

## Alternatives considered

- **Cloud SQL always**: correct for teams, but it would be the only fixed cost of a portfolio demo.
- **SQLite written in place over FUSE**: GCS has no file locks and rewrites the whole object on each
  write, so concurrent writers corrupt the store. Replacing the file (`os.replace`) also left MLflow's
  cached engine reading the deleted file.
- **A file store** (`mlruns/` without a database): deprecated for the registry.

## Consequences

- The demo costs ~0 €/month. Object versioning on the bucket allows restoring an earlier store.
- Edits made in the MLflow UI while a retraining publishes its copy are lost. Several writers require
  Cloud SQL, which is documented in `infra/README.md`.
