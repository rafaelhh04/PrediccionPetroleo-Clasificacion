# Why 4 models can't beat a coin flip on Brent — and how to prove it

*Rafael Hernando Herias · October 2026 · about 15 minutes to read*

My machine learning course project asked a simple question: **will Brent crude close higher tomorrow?**
I trained four classifiers on sixteen years of daily prices, volatility, the dollar index, the VIX and
a geopolitical risk index:

- a logistic regression;
- an RBF SVM;
- a random forest;
- a multilayer perceptron written from scratch in NumPy.

The test-set ROC AUCs came out at **0.486, 0.491, 0.493 and 0.500**.

The tempting reading is "the models failed". The honest one is "I have not shown anything yet". An AUC of
0.49 on one test split could be bad luck hiding a weak signal, or good luck hiding a worse model. It
could also be leakage that happens to cancel out. Calling the result a failure, or claiming it confirms
market efficiency, both need **evidence**. This article is about producing that evidence, and then about
building a system that acts on it.

!!! warning "About the numbers in this article"
    The original AUCs above come from the Kaggle dataset. Every other figure comes from a **synthetic
    random walk** with exactly the same schema, because Kaggle, Yahoo Finance and FRED were unreachable
    from the environment where the project was rebuilt. That is less of a handicap than it sounds. On a
    random walk the right answer is known: *there is nothing to predict*. The synthetic run is therefore
    a calibration test of the methodology itself. A protocol that "finds" skill on a random walk is
    broken. The pipeline is ready to re-run on the real data (`brent data download && dvc repro`).

## The setup

- **Label.** `1` if tomorrow's Brent return is positive, `0` otherwise. The last day, whose outcome is
  unknown, is dropped. The original code silently labelled it `0`, because `NaN > 0` is `False`: the first
  of several bugs that only surface when you write tests for "obvious" code.
- **Features.** 29 candidates, all computed from data up to day *t*:
    - lagged returns, 21- and 63-day momentum, 7/30-day volatility and their ratio;
    - RSI, MACD and Bollinger bands;
    - VIX and dollar-index levels and changes;
    - geopolitical risk and event flags;
    - the weekday and month as sine/cosine.

    A VIF filter drops collinear ones inside each training window.

- **Models.** The four above plus LightGBM tuned with Optuna. I added it later to try harder, and it did
  not change the conclusion.

## Pitfall 1: leakage, the silent optimist

Leakage makes a model look better than it is, so a *null* result does not rule it out: the leak might
be hiding an even worse model. I removed it at three levels.

**Same-day information.** The dataset contained WTI's return for day *t*. WTI and Brent move together,
so that feature partly reveals the target's own day, and it is removed.

**Preprocessing outside the model.** The original code already fitted VIF selection and winsorisation on
the training split only, but by hand, on NumPy arrays. That is correct for one split and wrong as soon as
the data is split again, for example by cross-validation folds or refits. Every transformation is now a
scikit-learn `Pipeline` step, `VIFSelector → Winsorizer → StandardScaler → model`, so it is refitted on
the training rows of every fold automatically.

**Look-ahead through the split.** A random K-fold trains on the future. Even a chronological split leaks
at its border: the label of day *t* is realised on *t + 1*, and rolling features make neighbouring days
near-duplicates. The fix is **purged walk-forward validation** (López de Prado, 2018):

```text
time ───────────────────────────────────────────────────────────────────────────►
 development period (before 2024)           │ out of sample: 10 windows × 63 days
 window 1 [===========================] pe  [test 1]
 window 2 [=====================================] pe  [test 2]
 window 3 [===============================================] pe  [test 3]
                                                                         …
 train = every strictly older row   p = purge (1 day)   e = embargo (5 days)
```

Before each 63-day test window, the model is refitted on all strictly older rows, minus a **purge** of
1 day (the label horizon) and an **embargo** of 5 more. The same splitter drives hyperparameter search,
learning curves and the final evaluation, so tuning cannot use a laxer protocol than the test.

Then I tested the property instead of trusting it:

- corrupting every row after a date must not change any feature, training label or prediction before
  that date;
- every feature has its own truncation test.

## Pitfall 2: the wrong bar

"Better than 0.5" is not the bar, for two reasons.

First, **naive rules can look skilled.** Persistence ("tomorrow repeats today") reached an
out-of-sample AUC of **0.507** on the synthetic series. A model has to beat the *best* naive rule, so
four baselines go through exactly the same walk-forward protocol as the models: majority class,
persistence, stratified random and buy & hold.

Second, **pooling hides surprises.** A constant prediction has AUC 0.5 inside one window. But the
majority-class baseline re-estimates its constant at each refit, and its pooled AUC over ten windows was
**0.476**. If you compare against a hard-coded 0.5 you are comparing against the wrong thing.

## Pitfall 3: point estimates

Here are the out-of-sample results (573 days, 2024-01-01 → 2026-03-11, synthetic):

| Model | AUC [95 % CI] | ΔAUC vs persistence [95 % CI] | p (DeLong, Holm) |
|-------|---------------|-------------------------------|-----------------:|
| Logistic regression | 0.532 [0.485, 0.581] | +0.024 [−0.031, +0.078] | 1.000 |
| Random forest | 0.519 [0.473, 0.562] | +0.011 [−0.043, +0.062] | 1.000 |
| LightGBM | 0.512 [0.467, 0.560] | +0.005 [−0.051, +0.059] | 1.000 |
| MLP (NumPy) | 0.510 [0.465, 0.556] | +0.003 [−0.058, +0.061] | 1.000 |
| SVM (RBF) | 0.492 [0.441, 0.539] | −0.015 [−0.072, +0.041] | 1.000 |

Without the brackets, a logistic regression at 0.532 on a random walk would look like a discovery. Two
tools turn the column into evidence.

**Block-bootstrap intervals.** Daily outcomes are serially dependent: volatility clusters, and features
overlap through rolling windows. Resampling single days pretends there are more independent observations
than there are, which gives intervals that are too narrow. A **circular block bootstrap** with 21-day
blocks (Politis & Romano, 1992) keeps that short-range dependence. All candidates are resampled on the
same days, so differences are paired.

**DeLong with Holm.** The DeLong test (DeLong et al., 1988) compares two correlated AUCs on the same days.
Five models against one baseline make five tests, so the p-values get a **Holm** correction (Holm, 1979).
DeLong assumes independent days, so a model only counts as better if **both** of these hold:

- the Holm-adjusted p-value is below 0.05;
- the bootstrap interval of ΔAUC is above zero.

Every AUC interval contains 0.5 and every ΔAUC interval contains 0. No accuracy beats the
no-information rate of 0.517 (binomial test). The Brier skill scores against the climatological forecast
are all within ±0.02 of zero.

## Pitfall 4: the winner's curse

Selecting the maximum of many noisy scores guarantees an optimistic estimate. The synthetic run shows it
cleanly. In the development period, the **SVM had the best cross-validated AUC (0.514)**, the best of nine
grid points. Out of sample it was the **worst model (0.492)**, and it barely traded, because its
calibrated probabilities almost never crossed 0.5.

The same mechanism is why 50 Optuna trials are a liability on noise: the best trial is the luckiest
one. Only an untouched out-of-sample period, plus intervals that account for dependence, can reveal it.

## The backtest: a Sharpe of 1.16 means nothing alone

AUC is not money, so each candidate also became a long/flat strategy: long when `P(up) ≥ 0.5`, paying
5 bps per unit of turnover.

| Strategy | CAGR | Sharpe | ΔSharpe vs buy & hold [95 % CI] | Max drawdown |
|----------|-----:|-------:|---------------------------------|-------------:|
| Buy & hold | +32.8 % | 1.06 | — | −34.4 % |
| LightGBM | +24.8 % | 1.17 | +0.11 [−0.85, +1.00] | −20.3 % |
| Logistic regression | +24.7 % | 1.16 | +0.11 [−1.06, +1.32] | −22.6 % |
| Persistence | +19.7 % | 0.90 | −0.16 [−1.04, +0.72] | −23.3 % |

A Sharpe of 1.16 with a shallower drawdown than buy & hold is a slide that gets funded. Two facts take it
apart.

- **Sitting out is not skill.** The strategies are long about 45 % of the time. Being out of the market
  during part of a drawdown lowers volatility whether or not the timing is informed.
- **The interval of the difference is about two Sharpe units wide.** With two years of daily data, a
  Sharpe ratio is simply not estimable to the precision the point value suggests.

## Explanations as a diagnostic

SHAP and permutation importance are usually shown as "what drives the model". Here they were most useful
as a **symptom**:

- the top permutation importances were 0.015–0.018 AUC, with standard deviations of the same size;
- the top features differed between models: Bollinger width for the logistic regression, momentum and
  the volatility ratio for the trees.

A real signal produces importances that are large relative to their noise and stable across model
families. This was neither.

## From a null result to an engineering decision

A notebook would end here. A system has to *do* something with the conclusion. The rule I implemented is
the core design decision of the project:

> A model is promoted to `champion` in the model registry only if it beats the best baseline with
> significance: Holm-adjusted DeLong **and** a bootstrap interval above zero. Otherwise the best
> **baseline** is the champion and the best model is kept as `challenger`.

So the FastAPI service in front of the registry serves **persistence**, and it says so in three places:

- `/model-info`;
- the dashboard banner;
- a `brent_model_info{kind="baseline"}` Prometheus metric.

The weekly retraining job applies the same rule unattended. If real data ever contained a signal, it
would be promoted without code changes. Until then, no amount of retraining can push a lucky model into
production.

The rest of the platform exists to keep that decision trustworthy:

- **reproducibility:** DVC stages with a lock file and MLflow runs;
- **input validation:** Pandera contracts on every data boundary;
- **interface stability:** a snapshot of the OpenAPI contract;
- **operations:** Evidently drift monitoring that alerts when the inputs move.

Drift monitoring shows the same humility. On the random walk, **57 %** of the features drift between the
last 60 days and the 756 before. Price levels and volatilities are non-stationary, so that is expected.
The predictions, however, do not drift. An alert is a reason to look, not a verdict.

## Lessons

1. **A null result is a result only with a protocol that could have found a positive one.** Synthetic
   data with a known answer is an excellent test of that protocol.
2. **Compare against the best naive rule, through the same pipeline.** 0.5 is not the bar.
3. **Report intervals that respect the dependence in the data.** Most "skill" on daily returns fits
   inside them.
4. **Distrust the winner.** The best score after a search is the most optimistic one.
5. **Encode the conclusion.** A promotion rule in code, with tests, outlives the analyst's good
   intentions.

## Reproduce it

```bash
git clone https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion.git && cd PrediccionPetroleo-Clasificacion
make demo                    # synthetic data, training, statistical report, champion, explanations (~6 min)
make -j2 serve dashboard     # API on :8000/docs, dashboard on :8501
```

The full protocol (`uv run brent train && uv run brent report`) takes about 10 minutes on 4 cores and
gives exactly the numbers above. The [methodology guide](../guide/methodology.md) and the
[model card](../MODEL_CARD.md) have the details. The design decisions are recorded as
[ADRs](../adr/index.md).

## References

- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more
  correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3).
- Fama, E. F. (1970). Efficient capital markets: a review of theory and empirical work. *The Journal of
  Finance*, 25(2).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics*, 6(2).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, ch. 7 (cross-validation in
  finance).
- Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\* '19*.
- Politis, D. N. and Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In
  *Exploring the Limits of Bootstrap*, Wiley.
