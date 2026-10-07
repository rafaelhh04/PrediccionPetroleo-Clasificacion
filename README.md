# Brent Direction Classifier

> Next-day direction of the Brent crude oil price from market indicators and geopolitical events (2010–2026).

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
uv run brent train           # full pipeline: preprocess, tune, train, evaluate (~1-2 min)
```

Outputs are written to `results/` (ignored by git):

| Path | Content |
|------|---------|
| `results/metrics.json` | Train / validation / test metrics of every model |
| `results/run.log` | Full log of the run |
| `results/plots/` | ROC curves, confusion matrices, learning curves, MLP training curves |

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
BRENT_SEED=7 BRENT_SPLIT__TRAIN_END=2021-01-01 uv run brent train
BRENT_MODELS__RANDOM_FOREST__GRID='{"max_depth": [5, 10]}' uv run brent config show
uv run brent --config configs/my_experiment.yaml train
```

Precedence: explicit arguments > environment variables > YAML file.

---

## Methodology

```text
load + left-join events ─► label (t+1) ─► feature engineering ─► drop nulls ─► VIF filter (train only)
   ─► chronological split ─► winsorise 1–99 % (train percentiles) ─► StandardScaler (fit on train)
   ─► grid search with TimeSeriesSplit(5) on train ─► fit ─► validation ─► single pass on test
```

- **Chronological split:** train `< 2022-01-01`, validation `2022–2023`, test `≥ 2024-01-01`.
- **Leakage controls:** same-day `wti_return` removed; VIF selection, winsorisation percentiles and scaler are
  fitted on training rows only; tuning uses expanding-window `TimeSeriesSplit` on train only; the test set is
  evaluated exactly once.
- **Features:** log returns over 1/3/7 days, 30-day volatility and 7/30 volatility ratio, Brent–WTI spread,
  21-day change of the geopolitical risk index, high-severity event flag, day of week and month
  (10 features survive the VIF > 10 filter).
- **Metrics:** accuracy, macro precision / recall / F1 and ROC AUC (primary, used for model selection).

### Models

| Model | Implementation | Search space |
|-------|----------------|--------------|
| Logistic regression (L2) | scikit-learn | `C` |
| SVM, RBF kernel | scikit-learn | `C`, `gamma` |
| Random forest | scikit-learn | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **MLP 64 → 32 → 1** | **pure NumPy** | hidden sizes, learning rate |

---

## Results

Reference run on the Kaggle data (ROC AUC):

| Model | Train | Validation | Test |
|-------|------:|-----------:|-----:|
| Logistic Regression | 0.5507 | 0.4800 | 0.4858 |
| SVM (RBF) | 0.5646 | 0.4700 | 0.4909 |
| Random Forest | 0.7342 | 0.4936 | 0.4927 |
| MLP NumPy | 0.5717 | 0.4933 | 0.5002 |

No model beats a coin flip out of sample — consistent with the weak-form efficient-market hypothesis for
daily returns. Proving this rigorously (baselines, confidence intervals, statistical tests, economic
backtest) is the goal of the next phase of the [roadmap](docs/ROADMAP.md).

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
│   ├── features/               # label, feature engineering, VIF, split, winsorisation, scaling
│   ├── models/                 # LogReg, SVM, RF, NumPy MLP, shared tuning helpers
│   └── evaluation/             # metrics, plots, learning curves, final test evaluation
├── tests/
├── Makefile
└── pyproject.toml / uv.lock
```

## Development

```bash
make install     # uv sync --frozen + pre-commit install
make lint        # all pre-commit hooks: ruff, codespell, mypy --strict, file hygiene
make test        # pytest
make help        # every target
```

Conventions: [Conventional Commits](https://www.conventionalcommits.org/), one branch per feature, English
for code, docstrings (numpy style), logs and commits. Random seed `42` everywhere (`seed` in the config).

## Known limitations

- `create_label` keeps the last row with label `0` (its next-day return is unknown) instead of dropping it;
  the fix changes the metrics and is scheduled with the ML-rigour phase.
- A single fixed train/validation/test split; walk-forward validation is planned.

## Author

Rafael Hernando Herias — originally an individual Machine Learning course project (2025-26).

## License

[MIT](LICENSE)
