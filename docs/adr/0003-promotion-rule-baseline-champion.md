# ADR 0003 — Promotion rule: the baseline is champion unless a model beats it with significance

- **Status:** accepted
- **Phase:** 3 (statistics), 4 (registry), 6 (scheduled retraining)

## Context

With five candidates, one of them will always have the highest out-of-sample AUC, even on pure noise:
that is the winner's curse. Tuning shows the same effect: on the synthetic data the SVM had the best
development-period AUC (0.514) and is the worst model out of sample (0.492). Deploying "the best model"
by point estimate would ship a model whose apparent skill is luck. Since Phase 6 the rule also runs unattended in the
weekly retraining job, so it has to be automatic and conservative.

## Decision

Naive baselines (majority class, persistence, stratified random, buy & hold) go through the same
walk-forward protocol as the models. The best model by out-of-sample AUC becomes the `champion` alias in
the MLflow registry **only if both** of these hold against the best baseline:

1. the DeLong test of the AUC difference is significant after **Holm** correction for the four
   comparisons (α = 0.05);
2. the paired **block-bootstrap** 95 % interval of ΔAUC (21-day blocks) lies above zero, since DeLong
   assumes independent days.

Otherwise the best **baseline** becomes `champion`, and the best model becomes `challenger`. The reason
is stored as a registry tag and in `results/promotion.json`.

## Alternatives considered

- **Highest AUC wins**: ships noise.
- **A fixed AUC margin** (e.g. > 0.52): arbitrary, and it ignores the sample size and the dependence
  between days.
- **DeLong only**: too optimistic with serially dependent outcomes.

## Consequences

- On the data available today, the persistence baseline is served. The API, the dashboard and the
  Prometheus metric `brent_model_info{kind="baseline"}` say so openly.
- Retraining can never replace the champion with a model that is only lucky. A real signal would be
  promoted without code changes.
- The rule is code (`tracking.select_champion`) with unit tests for every branch, not a manual decision.
