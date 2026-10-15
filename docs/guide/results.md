# Results

What the evaluation found. Read the warning first: almost every number here is synthetic.

> **Which numbers are real?** Only the first table, an old run of the original code on the Kaggle data.
> Every table after it comes from the **synthetic** dataset (a random walk with the Kaggle schema) and
> demonstrates the evaluation machinery, not the Brent market. The [model card](../MODEL_CARD.md)
> gathers intended use, metrics with intervals, limitations and risks.

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
573 days, 95 % block-bootstrap intervals, reference baseline: Persistence, AUC 0.5073), with the v2
features:

| Model | OOS AUC [95 % CI] | ΔAUC vs Persistence [95 % CI] | p (Holm) | Brier skill |
|-------|-------------------|-------------------------------|---------:|------------:|
| Logistic Regression | 0.536 [0.489, 0.582] | +0.028 [−0.026, +0.082] | 1.000 | +0.0027 |
| SVM (RBF) | 0.491 [0.442, 0.543] | −0.016 [−0.075, +0.046] | 1.000 | −0.0006 |
| Random Forest | 0.510 [0.465, 0.552] | +0.003 [−0.052, +0.057] | 1.000 | −0.0053 |
| MLP NumPy | 0.501 [0.453, 0.549] | −0.007 [−0.069, +0.056] | 1.000 | −0.0073 |
| LightGBM | 0.450 [0.401, 0.497] | −0.057 [−0.116, +0.002] | 0.465 | −0.1245 |

Backtest on the same days (long/flat, 5 bps per trade):

| Strategy | CAGR | Sharpe | ΔSharpe vs buy & hold [95 % CI] | Max drawdown | Exposure |
|----------|-----:|-------:|---------------------------------|-------------:|---------:|
| Buy & hold | +32.8 % | 1.06 | — | −34.4 % | 100 % |
| Logistic Regression | +20.6 % | 1.11 | +0.06 [−1.09, +1.18] | −19.6 % | 35 % |
| SVM (RBF) | +5.0 % | 0.79 | −0.27 [−1.84, +1.21] | −4.4 % | 3 % |
| Random Forest | −6.9 % | −0.27 | −1.33 [−2.52, −0.21] | −28.6 % | 39 % |
| MLP NumPy | +19.8 % | 1.00 | −0.06 [−1.33, +1.15] | −28.1 % | 43 % |
| LightGBM | −7.0 % | −0.24 | −1.30 [−2.60, −0.07] | −36.8 % | 47 % |
| Persistence (baseline) | +19.7 % | 0.90 | −0.16 [−1.04, +0.72] | −23.3 % | 48 % |

Every AUC interval contains 0.5, no model beats persistence or the no-information rate and no strategy
beats buy & hold after costs: as expected on a random-walk price series, there is no skill to find. The
logistic regression's AUC of 0.536 and Sharpe of 1.11 look attractive in isolation; the intervals show they
are compatible with luck.

LightGBM illustrates the opposite trap. Its best Optuna trial reached a development CV AUC of 0.517, the
maximum of 50 noisy estimates (winner's curse), and it fell to 0.450 out of sample, with overconfident
probabilities (Brier skill −0.12). The most flexible model, tuned hardest, is the one that overfits the noise
most.

Effect of the v2 features (same protocol; v1 = 14 features, v2 = 29 before the VIF filter):

| Model | Dev CV AUC v1 → v2 | OOS AUC v1 → v2 | Sharpe v1 → v2 |
|-------|-------------------|-----------------|----------------|
| Logistic Regression | 0.501 → 0.505 | 0.508 → 0.536 | 0.10 → 1.11 |
| SVM (RBF) | 0.500 → 0.511 | 0.492 → 0.491 | 0.00 → 0.79 |
| Random Forest | 0.491 → 0.508 | 0.473 → 0.510 | −0.57 → −0.27 |
| MLP NumPy | 0.505 → 0.507 | 0.467 → 0.501 | 0.26 → 1.00 |
