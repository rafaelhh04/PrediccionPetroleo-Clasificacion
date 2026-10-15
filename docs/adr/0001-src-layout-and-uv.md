# ADR 0001 — `src` layout, PEP 621 and uv

- **Status:** accepted
- **Phase:** 1 (repository foundations)

## Context

The original project was a set of scripts run with `python main.py`. It had no declared dependencies,
no environment file, and file names in the code that did not match the README, so `git clone` followed
by running the code failed. Tests (Phase 2) need to import the code exactly as a user would.

## Decision

- Package the code as `brent_forecast` under `src/` with PEP 621 metadata in `pyproject.toml` and the
  `uv_build` backend, exposing one Typer CLI (`brent`).
- Manage the environment with **uv** and commit `uv.lock`. CI, the Docker image and the pre-commit hooks
  all install from the lockfile (`uv sync --frozen`, `UV_FROZEN=1`). Ruff, codespell and mypy run as local
  hooks through `uv run`, so their versions are the locked ones.
- Optional runtime pieces are extras (`dashboard`, `postgres`, `monitoring`). Tooling lives in dependency
  groups (`dev`, `load`, `docs`).

## Alternatives considered

- **Flat layout** (`brent_forecast/` at the root): tests could import the working tree instead of the
  installed package and hide packaging errors.
- **pip + requirements.txt / pip-tools**: no single tool for Python versions, locking and running.
- **Poetry / conda**: heavier, slower to resolve, and conda is not needed (no native dependency that pip
  wheels do not cover).

## Consequences

- One command creates a reproducible environment (`uv sync`), and uv also installs the Python version.
- Every consumer (developer, CI, Docker, Dependabot) reads the same lockfile. Dependabot's `uv` ecosystem
  keeps it fresh.
- Contributors need uv installed. This is documented in the quickstart.
