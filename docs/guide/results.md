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

**Statistical report on the synthetic dataset** (not real data). Full run with `configs/default.yaml` on
`brent data synthetic` (seed 0): out of sample 2024-01-01 to 2026-03-11, 573 days, 95 % block-bootstrap
intervals. The reference baseline is persistence (AUC 0.5073):

| Model | Dev CV AUC | OOS AUC [95 % CI] | ΔAUC vs Persistence [95 % CI] | p (Holm) | Brier skill |
|-------|-----------:|-------------------|-------------------------------|---------:|------------:|
| Logistic Regression | 0.509 | 0.532 [0.485, 0.581] | +0.024 [−0.031, +0.078] | 1.000 | +0.0011 |
| SVM (RBF) | 0.514 | 0.492 [0.441, 0.539] | −0.015 [−0.072, +0.041] | 1.000 | −0.0002 |
| Random Forest | 0.508 | 0.519 [0.473, 0.562] | +0.011 [−0.043, +0.062] | 1.000 | −0.0005 |
| MLP NumPy | 0.504 | 0.510 [0.465, 0.556] | +0.003 [−0.058, +0.061] | 1.000 | −0.0021 |
| LightGBM | 0.510 | 0.512 [0.467, 0.560] | +0.005 [−0.051, +0.059] | 1.000 | −0.0169 |

Backtest on the same days (long/flat, 5 bps per trade):

| Strategy | CAGR | Sharpe | ΔSharpe vs buy & hold [95 % CI] | Max drawdown | Exposure |
|----------|-----:|-------:|---------------------------------|-------------:|---------:|
| Buy & hold | +32.8 % | 1.06 | — | −34.4 % | 100 % |
| Logistic Regression | +24.7 % | 1.16 | +0.11 [−1.06, +1.32] | −22.6 % | 45 % |
| SVM (RBF) | −0.8 % | −0.68 | −1.74 [−3.17, +0.02] | −1.8 % | 0 % |
| Random Forest | +19.2 % | 0.96 | −0.10 [−1.19, +0.96] | −17.6 % | 39 % |
| MLP NumPy | +17.2 % | 0.87 | −0.19 [−1.10, +0.78] | −21.1 % | 46 % |
| LightGBM | +24.8 % | 1.17 | +0.11 [−0.85, +1.00] | −20.3 % | 43 % |
| Persistence (baseline) | +19.7 % | 0.90 | −0.16 [−1.04, +0.72] | −23.3 % | 48 % |

Every AUC interval contains 0.5, no model beats persistence or the no-information rate (0.517) and no
strategy beats buy & hold after costs: as expected on a random-walk price series, there is no skill to
find. The logistic regression's AUC of 0.532 and the Sharpe ratios of 1.16–1.17 of the logistic regression
and LightGBM look attractive in isolation; the intervals show they are compatible with luck.

The development CV shows the winner's curse. The SVM had the best development AUC (0.514, the best of 9
grid points) and is the worst model out of sample (0.492), barely trading because its calibrated
probabilities stay below 0.5. LightGBM's best Optuna trial is the maximum of 50 noisy estimates. Selecting
the maximum of many noisy scores guarantees an optimistic estimate; only the out-of-sample protocol and the
intervals reveal it.

The run takes about 10 minutes on 4 cores (`brent train && brent report && brent explain`) and is
deterministic: two runs on the same lockfile give the same numbers. `make demo` uses the smaller search
spaces of `configs/demo.yaml`, so its numbers differ slightly (e.g. random forest 0.538) with the same
conclusion.

Effect of the v2 features (Phase 3 run, with the dependency versions of that time, so its v2 column differs
from the tables above; same protocol; v1 = 14 features, v2 = 29 before the VIF filter):

| Model | Dev CV AUC v1 → v2 | OOS AUC v1 → v2 | Sharpe v1 → v2 |
|-------|-------------------|-----------------|----------------|
| Logistic Regression | 0.501 → 0.505 | 0.508 → 0.536 | 0.10 → 1.11 |
| SVM (RBF) | 0.500 → 0.511 | 0.492 → 0.491 | 0.00 → 0.79 |
| Random Forest | 0.491 → 0.508 | 0.473 → 0.510 | −0.57 → −0.27 |
| MLP NumPy | 0.505 → 0.507 | 0.467 → 0.501 | 0.26 → 1.00 |
