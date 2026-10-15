# Architecture Decision Records

Each record captures one significant decision: its context, the options that were weighed, the choice
and its consequences. The format is a lightweight
[MADR](https://adr.github.io/madr/): status, context, decision, alternatives, consequences.

The decisions were taken while the project was built phase by phase (see the
[roadmap](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/blob/main/docs/ROADMAP.md)).
They were written up as records in Phase 7. A record is never edited to reverse a decision: it is
superseded by a new one.

| ADR | Decision | Phase |
|-----|----------|-------|
| [0001](0001-src-layout-and-uv.md) | `src` layout, PEP 621 metadata and uv with a committed lockfile | 1 |
| [0002](0002-purged-walk-forward-validation.md) | Purged walk-forward validation instead of a fixed train/validation/test split | 3 |
| [0003](0003-promotion-rule-baseline-champion.md) | A model is promoted only if it beats the best baseline with significance | 3–4 |
| [0004](0004-mlflow-sqlite-vs-cloud-sql.md) | MLflow on SQLite in the bucket by default, Cloud SQL as an option | 4, 6 |
| [0005](0005-gcs-fuse-shared-paths.md) | GCS FUSE mounts at the same paths as docker compose | 6 |
| [0006](0006-cd-candidate-revision-oidc.md) | Keyless CD with OIDC, one digest per commit and a smoke-tested candidate revision | 6 |
| [0007](0007-drift-monitoring-evidently.md) | Data and prediction drift with Evidently on a sliding window | 6 |
| [0008](0008-synthetic-demo-layered-config.md) | A synthetic dataset and a layered demo configuration for the quickstart | 7 |
