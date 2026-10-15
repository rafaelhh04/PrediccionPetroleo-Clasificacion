# ADR 0002 — Purged walk-forward validation

- **Status:** accepted
- **Phase:** 3 (ML rigour)

## Context

The original evaluation used one fixed chronological split (train / validation / test) and fitted the
preprocessing with NumPy arrays outside any pipeline. That gave one noisy estimate per model and made
leakage easy to reintroduce. Daily returns are serially dependent, and the label of day *t* is
realised on day *t+1*. Some features use rolling windows of up to 63 days.

## Decision

- Wrap every model in a scikit-learn `Pipeline`:
  `VIFSelector → Winsorizer → StandardScaler → model`. Each step is refitted on the training rows of
  every fold and window.
- Use one splitter, `PurgedWalkForwardSplit`, for tuning, learning curves and the out-of-sample
  evaluation:
    - **Development period** (before 2024-01-01): hyperparameters are tuned with 5 walk-forward folds.
    - **Out-of-sample period**: the model is refitted every 63 trading days on all older rows. The window
    is expanding by default and rolling as an option.
    - Training rows end `purge` (1 day, the label horizon) + `embargo` (5 days of buffer) before each
    test window (López de Prado, *Advances in Financial Machine Learning*, ch. 7).
- Prove the absence of look-ahead with tests. Truncating or corrupting the future must never change
  earlier features, training labels or predictions, and every feature has a truncation test.

## Alternatives considered

- **Random K-fold**: trains on the future and leaks through overlapping rolling features.
- **One fixed split**: one estimate per model, no view of the stability across regimes.
- **Combinatorial purged CV**: more paths, but harder to explain and costlier, for no different
  conclusion on this problem.

## Consequences

- Ten out-of-sample windows give a pooled AUC plus per-window AUCs, so stability over time is visible.
- The same splitter in every stage means tuning cannot use a laxer protocol than the evaluation.
- Training is slower: every model is refitted per window and per fold. A full run takes about 12 minutes
  on 4 cores. The demo overlay ([ADR 0008](0008-synthetic-demo-layered-config.md)) shrinks the search
  spaces but keeps the protocol.
