# Contributing

Thanks for your interest in the Brent Direction Classifier. This guide covers the local setup, the
development workflow and the repository settings that keep `main` healthy.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12 if
needed) and `make`.

```bash
git clone https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion.git
cd PrediccionPetroleo-Clasificacion
make install          # uv sync --frozen + pre-commit git hook
```

The raw data is only needed to train (`make data`, see the README); the test suite uses synthetic data.

## Everyday commands

| Command | What it does |
|---------|--------------|
| `make lint` | every pre-commit hook: ruff (lint + format), codespell, `mypy --strict`, file hygiene |
| `make format` | auto-format and auto-fix with ruff |
| `make test` | unit tests (slow end-to-end tests excluded), ~30 s |
| `make coverage` | tests with branch coverage; fails under 85 %; HTML report in `htmlcov/` |
| `make test-all` | also runs the tests marked `slow` (full `brent train` through the CLI) |
| `make train` | full training pipeline on the real data |

Run a single test file or a single slow test with `uv run pytest tests/test_cli.py` or
`uv run pytest -m slow -k train`.

> **Tip:** `pre-commit run --all-files` only checks files Git already tracks. Run `git add` on new files
> before `make lint`.

## Tests

- Tests never use the real dataset or the network: fixtures in `tests/conftest.py` build seeded synthetic
  frames with the Kaggle schema and inject a fake `kagglehub` downloader.
- `tests/test_leakage.py` guards against data leakage and `tests/test_mlp_gradients.py` checks the NumPy
  MLP gradients numerically; keep both green when touching features, preprocessing or the network.
- Configuration overrides from your shell (`BRENT_*`) are removed automatically during tests.
- Mark tests slower than a few seconds with `@pytest.mark.slow`; they run in CI weekly and on demand.

## Workflow

1. Start from an up-to-date `main` and create one branch per change, named `type/short-description`:
   `feat/`, `fix/`, `refactor/`, `test/`, `ci/`, `docs/`, `chore/`.
2. Commit with [Conventional Commits](https://www.conventionalcommits.org/): `feat(data): ...`,
   `fix: ...`, `test(models): ...`. Keep commits small and atomic.
3. Code, docstrings (numpy style), log messages and commits are in English.
4. Do not change model logic and infrastructure in the same PR. If metrics change, include a before/after
   table in the PR description.
5. Open a pull request against `main` and fill in the template. CI must be green before merging; use a
   merge commit.

Never commit data, credentials or generated artefacts (`data/raw/`, `results/`, `mlruns/` are ignored).

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`:

- **Lint & type-check**: `ruff check`, `ruff format --check`, `mypy` (strict), `codespell`.
- **Tests (Python 3.12 / 3.13)**: `pytest` with branch coverage (`fail_under = 85`); the 3.12 job uploads
  `coverage.xml` to Codecov.
- **Slow end-to-end tests**: weekly (Monday) and on manual dispatch (*Actions → CI → Run workflow*).

Dependabot opens grouped weekly PRs for Python dependencies (`uv.lock`) and GitHub Actions.

## Repository settings (maintainers)

These live in the GitHub UI, not in the code, and must be configured once.

### Codecov token

1. Sign in at [codecov.io](https://codecov.io) with GitHub and add this repository.
2. Copy the repository upload token.
3. GitHub: *Settings → Secrets and variables → Actions → New repository secret*, name `CODECOV_TOKEN`.

Without the secret Codecov attempts a tokenless upload, which may be rejected; the step is non-blocking
(`fail_ci_if_error: false`) because coverage is already enforced by pytest.

### Protect `main`

The ruleset is versioned in [`.github/rulesets/protect-main.json`](.github/rulesets/protect-main.json):
pull request required, the three CI checks required and up to date with `main`, merge commits only,
conversations resolved, no force-pushes and no deletion.

**Import it (recommended):** *Settings → Rules → Rulesets → New ruleset → Import a ruleset* and select the
JSON file, then *Create*.

**Or with the GitHub CLI** (needs admin rights on the repository):

```bash
gh api --method POST repos/rafaelhh04/PrediccionPetroleo-Clasificacion/rulesets \
  --input .github/rulesets/protect-main.json
```

The required check names must match the job names in `ci.yml` (`Lint & type-check`,
`Tests (Python 3.12)`, `Tests (Python 3.13)`); update both together. `required_approving_review_count` is 0
because the project has a single maintainer, who cannot approve their own pull requests; raise it when
there are more reviewers.
