# ADR 0008 — Synthetic dataset and layered demo configuration for the quickstart

- **Status:** accepted
- **Phase:** 7 (portfolio)

## Context

A recruiter or reviewer should be able to run the system in a few minutes, with no credentials.
Two obstacles stood in the way:

- the real dataset needs a Kaggle account, and Kaggle, Yahoo Finance and FRED were not reachable from
  the development environment at all;
- a full training run takes about 12 minutes.

The test suite already generated a seeded random walk with the Kaggle schema.

## Decision

- **`brent data synthetic`** writes that generator's output to `data/raw/`.
    - It refuses to overwrite existing files without `--force`.
    - It never records checksums, so every report and the dashboard label the run as synthetic.
    - The generator lives in `brent_forecast.synthetic`, outside `brent_forecast.data`. That package is a
    DVC stage dependency, and a demo helper must not invalidate `dvc.lock`.
- **Layered configuration.** `--config` is repeatable, and later YAML files are deep-merged over earlier
  ones (pydantic-settings `deep_merge`). `configs/demo.yaml` only lists smaller search spaces, fewer
  tuning folds and fewer bootstrap resamples. The protocol is unchanged, and a test asserts it.
- **`make demo`** chains install, data, train, report and explain, in about 6 minutes on 4 cores.

## Alternatives considered

- **Committing a sample of the real data**: the licence and the size make it a poor fit, and it would
  still not be reachable to regenerate.
- **A full copy of `default.yaml` for the demo**: two files to keep in sync.
- **Environment-variable overrides in the Makefile**: unreadable for nested grids.

## Consequences

- `git clone && make demo && make -j2 serve dashboard` is verified on a clean clone.
- Demo metrics differ slightly from those of a full run, and both are synthetic. The README states which
  is which.
