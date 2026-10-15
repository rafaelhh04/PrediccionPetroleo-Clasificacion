# ADR 0005 — GCS FUSE mounts at the same paths as docker compose

- **Status:** accepted
- **Phase:** 5 (containers), 6 (cloud)

## Context

MLflow records **absolute** artefact locations (`file:///app/mlruns/artifacts/...`). A model registered
by the training container must be loadable by the API, the dashboard and the MLflow UI, each in its own
container. Locally, docker compose solves this with named volumes mounted at `/app/mlruns` and
`/app/results` in every service. Cloud Run containers have no shared disk.

## Decision

- Mount the bucket folders with Cloud Run's Cloud Storage volume mounts (GCS FUSE) at **the same paths
  as compose**: `/app/data`, `/app/mlruns`, `/app/results`.
- Mount **read-only** wherever possible (the API and the dashboard). Only the MLflow UI and the jobs
  write.
- Make the API able to run on a read-only registry. `load_from_registry` resolves alias → version and
  loads from a temporary copy, because MLflow writes `registered_model_meta` next to the artefacts it
  loads.

## Alternatives considered

- **A GCS artefact store** (`gs://` URIs): this changes the paths between local and cloud runs and
  needs credentials inside every container that reads a model.
- **Baking the model into the image**: couples releases to retraining and defeats the hot refresh of
  the champion alias.
- **Filestore (NFS)**: a fixed monthly cost.

## Consequences

- The same images, settings and registry work locally and in the cloud. Only the volume driver changes.
- FUSE caching and its eventual consistency are acceptable for one weekly writer. The API checks its
  alias every `serving.refresh_interval_seconds` and reloads in the background.
- The FUSE daemon's memory counts against the Cloud Run instance, which the 1 GiB sizing accounts for.
