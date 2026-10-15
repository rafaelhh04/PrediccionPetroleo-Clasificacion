# Methodology

How the models are trained and evaluated without leaking the future, and how the evidence is weighed.

## Pipeline

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
- **Features** (all computed from data up to day *t*; one truncation test per feature proves it):
  - returns and risk: log returns over 1/3/7 days, 21- and 63-day momentum, 7/30-day volatility and their
    ratio, Brent–WTI spread;
  - technical indicators (`features/technical.py`): RSI(14) with Wilder smoothing, MACD(12, 26, 9) line and
    histogram divided by the price, Bollinger(20, 2) %B and bandwidth;
  - market context: 1- and 5-day log changes of the VIX and the dollar index (DXY), plus their levels;
  - geopolitics: 21-day change of the risk index, event and high-severity flags;
  - calendar: sine/cosine of weekday and month.

  The VIF > 10 filter then removes collinear features inside each training window.
- **Metrics:** accuracy, macro precision / recall / F1 and ROC AUC (primary, used for model selection),
  over the pooled out-of-sample predictions and per walk-forward window (stability over time).

## Models

| Model | Implementation | Search space |
|-------|----------------|--------------|
| Logistic regression (L2) | scikit-learn | `C` |
| SVM, RBF kernel (sigmoid-calibrated probabilities) | scikit-learn | `C`, `gamma` |
| Random forest | scikit-learn | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **MLP 64 → 32 → 1** | **pure NumPy core**, scikit-learn estimator API (`NumpyMLPClassifier`) | hidden sizes, learning rate |
| LightGBM (gradient boosting) | `lightgbm` | 8 hyperparameters with **Optuna** |

LightGBM is tuned with Optuna's TPE sampler (seeded, 50 trials, `tuning` section of the config) instead of a
grid. Each trial is scored with the same purged walk-forward folds; with median pruning a trial whose running
mean AUC falls below the median of earlier trials at the same fold is stopped early. All trials are saved to
`results/optuna_lightgbm.csv`.


## Baselines

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

## Statistical evaluation

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
- **Economic backtest** (`evaluation/backtest.py`): each candidate becomes a long/flat strategy (long when
  `P(up) >= 0.5`, optionally long/short) on the next day's return, paying 5 bps per unit of turnover. The
  report gives CAGR, volatility, Sharpe ratio, maximum drawdown, hit ratio, exposure and number of trades,
  and a paired block-bootstrap interval of the Sharpe difference against buy & hold. Settings live in the
  `backtest` section of the config.
- **Provenance**: the SHA-256 of the input files is stored in `metrics.json`; the report says whether
  they match the recorded Kaggle checksums or are unverified.

## Data contracts (Pandera)

Every data boundary has a Pandera schema (`data/schemas.py`), validated lazily so that one error message
lists every failed check:

| Schema | Where | Main checks |
|--------|-------|-------------|
| `OIL_SCHEMA` | prices CSV, on load | required columns, numeric coercion, unique dates, Brent in (0, 1000) USD, Brent return in (−100, 100) %, VIX in (0, 200], DXY in [50, 200], non-negative volatility and GPR |
| `EVENTS_SCHEMA` | events CSV, on load | unique dates (a duplicate would duplicate rows in the join), severity 0–10 |
| `merged_schema` | after the join | dates unique and increasing, one row per oil day, events filled, date range from `data.expected_*` |
| `FEATURES_SCHEMA` | `build_dataset`, `read_features` | label in {0, 1}, every feature numeric, non-null and finite, dates increasing |
| `INFERENCE_SCHEMA` | rows sent to a model | every feature numeric, non-null and finite |

The bounds are loose plausibility checks: they catch unit errors, wrong files and corrupted values, not
market moves. WTI is not range-checked because it settled at −37 USD on 2020-04-20. Invalid data stops the
run with `DataValidationError`, for example:

```text
oil prices failed validation:
  - column 'brent_price' failed in_range(0.0, 1000.0) for 2 row(s), e.g. -1.0, -2.0
  - missing required column(s): 'gpr_index'
```

## Explainability

`brent explain` (`evaluation/explain.py`) refits each model listed in `explain.models` on the development
period, with the hyperparameters of its saved pipeline, and explains it on the out-of-sample period. The
importances therefore describe what generalises, not what was memorised.

- **Permutation importance**: the drop in out-of-sample ROC AUC when each raw feature is shuffled
  (10 repeats, mean ± std). It is model-agnostic, and a feature removed by the VIF filter scores exactly 0.
- **SHAP**: exact TreeSHAP for LightGBM and the random forest, the closed form for logistic regression
  (log-odds), and the permutation explainer on `P(up)` for the SVM and the MLP. The global view is a
  beeswarm plot plus mean |SHAP|; the local view is a waterfall plot of the most recent out-of-sample day.
  Tests check additivity (base value + Σ SHAP = model output).

On the synthetic data the top permutation importances are within about 0.015–0.018 AUC with comparable
standard deviations, and the features ranked first differ between models. That pattern is what an absence
of stable signal looks like.
