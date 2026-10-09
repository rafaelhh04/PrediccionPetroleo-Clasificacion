# Model card — Brent next-day direction classifier

Format: Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

> **Read this first.** Every number in this card comes from a **synthetic** dataset with the schema of the
> Kaggle data (a random-walk price series), because Kaggle was not reachable from the development
> environment. `configs/data_checksums.json` records no digests yet, so `brent report` marks the provenance
> as *unverified*. The numbers show what the evaluation machinery does; they say nothing about the real
> Brent market until the pipeline is re-run on the published dataset (see [Reproducing](#reproducing)).

## Model details

| | |
|---|---|
| Task | Binary classification: will the Brent close of day *t+1* be above the close of day *t*? |
| Output | `P(up)` for day *t+1*, computed with information available at the close of day *t* |
| Candidates | L2 logistic regression, RBF SVM (sigmoid-calibrated), random forest, LightGBM, and a 64 → 32 → 1 MLP written in NumPy (scikit-learn estimator API) |
| Preprocessing | One scikit-learn `Pipeline` per model: `VIFSelector(10)` → `Winsorizer(1–99 %, return features)` → `StandardScaler` → model. Every step is fitted only on the training rows of each fold or window |
| Tuning | Grid search (LR, SVM, RF, MLP) or Optuna TPE with median pruning (LightGBM, 50 trials), both on purged walk-forward CV of the development period, maximising ROC AUC |
| Artefacts | `results/models/<model>.joblib`: each pipeline refitted on all labelled data |
| Version | `brent-forecast` 0.1.0, Phase 3 of `docs/ROADMAP.md` |
| Licence | MIT |

## Intended use

- **Primary**: an educational and portfolio project showing how to evaluate a financial classifier
  rigorously, with leakage controls, walk-forward validation, baselines, uncertainty, statistical tests,
  an economic backtest and explainability.
- **Users**: data-science students and reviewers of the methodology.

### Out of scope

- **Trading or investment decisions.** No model shows predictive skill (see below), and a strategy built
  on it is not expected to beat buy & hold after costs.
- Horizons other than one trading day, other assets, and intraday data.
- Any use with data that has not passed `brent data verify`.

## Data

- **Source**: Kaggle `kavyadhyani/global-oil-prices-andgeopolitical-events`. It contains daily Brent/WTI
  prices, DXY, VIX, a geopolitical-risk index and a timeline of geopolitical events, from 2010 to 2026.
- **Label**: `1` if the next day's Brent return is positive. Days without a known next return are
  dropped, never labelled 0.
- **Features** (29 candidates before VIF): returns over 1/3/7 days, momentum over 21/63 days, volatility
  and its 7/30 ratio, Brent–WTI spread, RSI(14), MACD(12, 26, 9), Bollinger(20, 2) %B and bandwidth, VIX
  and DXY levels with their 1- and 5-day changes, geopolitical risk change and event flags, and the
  weekday and month as sine/cosine. Each feature has a truncation test proving it only uses data up to
  its own day.
- **Data contracts**: Pandera schemas validate the raw files, the merged frame, the feature dataset
  and every inference row (types, ranges, unique increasing dates, finite features). Invalid data
  stops the pipeline with a message listing every failed check.
- **Split**:
  - Development period: before 2024-01-01, used for tuning.
  - Out-of-sample period: 2024-01-01 → 2026-03-11 (573 days in the synthetic run), evaluated by purged
    walk-forward in 10 windows of 63 days.
  - Each model is refitted before every window on all older data, minus purge (1 day) and embargo
    (5 days).

## Evaluation

All figures are out of sample, on the **synthetic** dataset. Intervals are 95 % circular block bootstrap
(blocks of 21 days, 2000 resamples, paired across candidates). The reference baseline is the one with the
highest out-of-sample AUC (persistence).

### Discrimination and calibration

| Candidate | AUC [95 % CI] | Accuracy [95 % CI] | ΔAUC vs persistence | p (DeLong, Holm) | Brier skill |
|---|---|---|---|---|---|
| Logistic Regression | 0.536 [0.489, 0.582] | 0.534 [0.497, 0.569] | +0.028 [−0.026, +0.082] | 1.000 | +0.003 |
| SVM (RBF) | 0.491 [0.442, 0.543] | 0.524 [0.483, 0.562] | −0.016 [−0.075, +0.046] | 1.000 | −0.001 |
| Random Forest | 0.510 [0.465, 0.552] | 0.494 [0.454, 0.532] | +0.003 [−0.052, +0.057] | 1.000 | −0.005 |
| MLP NumPy | 0.501 [0.453, 0.549] | 0.522 [0.482, 0.560] | −0.007 [−0.069, +0.056] | 1.000 | −0.007 |
| LightGBM | 0.450 [0.401, 0.497] | 0.445 [0.403, 0.487] | −0.057 [−0.116, +0.002] | 0.465 | −0.125 |
| *Persistence* | 0.507 [0.474, 0.538] | 0.508 [0.475, 0.539] | — | — | — |
| *Majority class* | 0.476 [0.432, 0.527] | 0.517 [0.475, 0.555] | — | — | 0 (reference) |

The no-information rate is 0.517. No model's accuracy is significantly above it (one-sided binomial test).

### Economic backtest

The strategy goes long when `P(up) ≥ 0.5` and stays flat otherwise, pays 5 bps per unit of turnover, and
assumes a zero risk-free rate.

| Strategy | CAGR | Sharpe | ΔSharpe vs buy & hold [95 % CI] | Max drawdown |
|---|---|---|---|---|
| Buy & hold | +32.8 % | 1.06 | — | −34.4 % |
| Logistic Regression | +20.6 % | 1.11 | +0.06 [−1.09, +1.18] | −19.6 % |
| MLP NumPy | +19.8 % | 1.00 | −0.06 [−1.33, +1.15] | −28.1 % |
| SVM (RBF) | +5.0 % | 0.79 | −0.27 [−1.84, +1.21] | −4.4 % |
| Random Forest | −6.9 % | −0.27 | −1.33 [−2.52, −0.21] | −28.6 % |
| LightGBM | −7.0 % | −0.24 | −1.30 [−2.60, −0.07] | −36.8 % |

### Conclusion (generated by `brent report`)

No model has a significantly higher out-of-sample AUC than the best naive baseline. Every model's AUC
interval contains 0.5, and no strategy beats buy & hold after costs. The evidence does not support
next-day directional skill. On a random-walk series this is the correct answer, and it is consistent with
the weak-form efficient-market hypothesis for daily returns.

Two results deserve attention because they look like findings and are not:

- **Logistic regression**, with an AUC of 0.536 and a Sharpe of 1.11, would look promising without
  intervals. Its intervals contain chance level and buy & hold.
- **LightGBM**:
  - Its best Optuna trial scored 0.517 in development CV, the maximum of 50 noisy estimates (the
    winner's curse).
  - Out of sample it scored 0.450, with overconfident probabilities.
  - It is the most flexible model and was tuned the hardest, and it overfits the noise the most.

### Explainability

`brent explain` refits the models on the development period and explains them on the out-of-sample
period. The top permutation importances are about 0.01–0.015 AUC with standard deviations of the same
order, and the top features differ between models (spread and Bollinger width for LR; momentum and DXY
for the trees). This pattern is what an absence of stable signal looks like. Treat these importances as
diagnostics, not as evidence of economic drivers.

## Deployment and promotion

`brent report` decides which candidate is served (`tracking.py`, `select_champion`). The best model by
out-of-sample AUC becomes the `champion` alias of the MLflow registered model only if it beats the best
naive baseline significantly (Holm-adjusted DeLong *and* a block-bootstrap interval of the AUC difference
above zero). Otherwise the best baseline is the `champion` and the best model the `challenger`. On the
synthetic data no model qualifies, so the champion is a **naive baseline**. That is the honest choice: a
model that cannot beat a naive rule only adds risk.

## Quantitative safeguards

- Leakage tests:
  - fitted preprocessing state depends on training rows only;
  - rewriting the future never changes past features, labels or predictions;
  - a truncation test for every feature;
  - spies prove that tuning, learning curves and explanations only see development rows.
- Walk-forward geometry is tested with Hypothesis properties: strict gap, order and non-overlap.
- The MLP's analytic gradients match finite differences (relative error < 1e-6), and it passes
  scikit-learn's `check_estimator`.
- CI runs on Python 3.12 and 3.13, with `mypy --strict` (no `type: ignore` in `src/`) and branch
  coverage ≥ 85 % (currently ~99 %).

## Ethical considerations and risks

- **Financial harm.** Presenting a chance-level model as a trading signal can cause losses. The report
  therefore always prints intervals, tests against baselines, and a cost-aware backtest next to any
  headline number.
- **Selection bias.** Trying more models, features or hyperparameters on the same out-of-sample period
  inflates the best result. The Holm correction covers the five model comparisons, but not the history
  of experiments run during development.
- **Data.** There is no personal data. The geopolitical-event labels come from a third-party dataset
  whose curation process is unknown.

## Caveats and recommendations

- **Re-run on the real data first.** Then commit the recorded checksums and replace the tables above.
- DeLong and binomial tests assume independent days. A comparison is only called significant when the
  block bootstrap agrees.
- The backtest ignores slippage beyond the fixed cost, financing, and the futures roll. A positive
  backtest would need those before any conclusion.
- Tuning optimism is not measured. Nested walk-forward, or a separate tuning-validation window, would
  quantify it.
- The macroeconomic features (VIX, DXY) are assumed known at the close of day *t*. With live data,
  check publication times.

## Reproducing

```bash
uv sync
uv run brent data download     # or manual download + `uv run brent data verify`
uv run brent train             # ~11 min on 4 cores
uv run brent report            # results/report.md
uv run brent explain           # results/explain/explanations.md
```

The seed (`seed: 42`), the single-threaded deterministic LightGBM and the seeded TPE sampler make every
step reproducible bit for bit on the same platform.
