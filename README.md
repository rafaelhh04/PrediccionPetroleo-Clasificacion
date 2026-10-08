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

Binary classification of whether Brent will close higher tomorrow, comparing four models — logistic
regression, RBF SVM, random forest and a **multilayer perceptron written from scratch in NumPy**
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
uv run brent train           # full pipeline: preprocess, tune, train, evaluate (~3 min)
uv run brent report          # statistical report of the last run (CIs, tests, calibration)
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

  train            Run the full pipeline: load, preprocess, tune, train and evaluate on test
  report           Build the statistical report (CIs, DeLong, binomial, calibration) of the last run
  data download    Download the Kaggle dataset and verify checksums  [--force] [--update-checksums]
  data verify      Verify the SHA-256 checksums of the local data files
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
- **Features:** log returns over 1/3/7 days, 30-day volatility and 7/30 volatility ratio, Brent–WTI spread,
  21-day change of the geopolitical risk index, high-severity event flag, day of week and month
  (on the real data 10 features survived the VIF > 10 filter).
- **Metrics:** accuracy, macro precision / recall / F1 and ROC AUC (primary, used for model selection),
  over the pooled out-of-sample predictions and per walk-forward window (stability over time).

### Models

| Model | Implementation | Search space |
|-------|----------------|--------------|
| Logistic regression (L2) | scikit-learn | `C` |
| SVM, RBF kernel (sigmoid-calibrated probabilities) | scikit-learn | `C`, `gamma` |
| Random forest | scikit-learn | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **MLP 64 → 32 → 1** | **pure NumPy core**, scikit-learn estimator API (`NumpyMLPClassifier`) | hidden sizes, learning rate |


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
- **Provenance**: the SHA-256 of the input files is stored in `metrics.json`; the report says whether
  they match the recorded Kaggle checksums or are unverified.

---

## Results

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
573 days, 95 % block-bootstrap intervals, reference baseline: Persistence, AUC 0.5073):

| Model | OOS AUC [95 % CI] | ΔAUC vs Persistence [95 % CI] | p (Holm) | Brier skill |
|-------|-------------------|-------------------------------|---------:|------------:|
| Logistic Regression | 0.508 [0.470, 0.548] | +0.001 [−0.054, +0.056] | 1.000 | −0.0000 |
| SVM (RBF) | 0.492 [0.445, 0.541] | −0.015 [−0.065, +0.041] | 1.000 | −0.0002 |
| Random Forest | 0.473 [0.424, 0.518] | −0.035 [−0.096, +0.025] | 0.932 | −0.0201 |
| MLP NumPy | 0.467 [0.425, 0.512] | −0.040 [−0.101, +0.024] | 0.932 | −0.0140 |

Every AUC interval contains 0.5, no model beats persistence or the no-information rate, and none improves
on the climatological probability: as expected on a random-walk price series, there is no skill to find.

---

## Project structure

```text
├── configs/
│   ├── default.yaml            # single source of configuration values
│   └── data_checksums.json     # SHA-256 of the raw CSVs
├── docs/ROADMAP.md             # professionalisation plan and progress log
├── src/brent_forecast/
│   ├── cli.py                  # Typer CLI (`brent`)
│   ├── config.py               # typed settings (pydantic-settings)
│   ├── logging_config.py       # console + per-run file logging
│   ├── pipeline.py             # end-to-end training orchestration
│   ├── data/                   # Kaggle download + checksums, loading and merge
│   ├── features/               # label, past-only features, split; VIFSelector / Winsorizer transformers
│   ├── models/                 # registry of model pipelines, NumPy MLP (core + estimator), tuning
│   └── evaluation/             # metrics, plots, learning curves, final test evaluation
├── tests/                      # unit, leakage-guard, gradient-check and property tests
├── .github/                    # CI workflow, Dependabot, templates, CODEOWNERS, ruleset
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

- A single fixed train/validation/test split; walk-forward validation is planned.

## Author

Rafael Hernando Herias — originally an individual Machine Learning course project (2025-26).

## License

[MIT](LICENSE)
