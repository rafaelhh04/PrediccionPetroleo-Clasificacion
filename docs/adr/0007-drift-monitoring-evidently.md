# ADR 0007 — Drift monitoring with Evidently on a sliding window

- **Status:** accepted
- **Phase:** 6 (cloud and CD)

## Context

A deployed model must be watched for changes in its inputs (data drift) and outputs (prediction drift).
The service keeps no log of served predictions, and ground truth arrives one day later. Market features
are non-stationary: price levels and volatilities move between regimes. GitHub Pages was reserved for
the documentation site.

## Decision

- A daily job, `brent monitor drift`, compares the feature rows of the **last 60 trading days** with
  the **756 days before them**.
- Evidently's `DataDriftPreset` picks a test per column (Kolmogorov–Smirnov at these sample sizes).
- The served model's `P(up)` on both windows is one more column: that is **prediction drift**, and it
  needs no prediction log.
- Calendar features are excluded, since they always differ between a short and a long window.
- An alert fires when at least 50 % of the features drift, or when the predictions drift. It goes to the
  same chat webhook as the retraining notifications.
- The HTML report, a JSON summary and a history line are written to `results/drift/`, published in the
  bucket and shown in the dashboard's *Monitoring* tab.

## Alternatives considered

- **Logging every prediction and comparing outputs**: needs a store and a write path in the API, for
  little gain with one prediction per day.
- **Custom KS/PSI code**: Evidently gives tested statistics, a readable report and per-column methods.
- **Publishing the reports to GitHub Pages**: needs credentials in Actions and mixes data into the docs.
- **Opening issues automatically**: needs a PAT or GCP credentials in Actions. A webhook is enough.

## Consequences

- On the synthetic random walk, 57 % of the features drift (non-stationary levels) and the predictions
  do not. Drift is a reason to look, not a verdict, and the docs say so.
- Evidently is an optional extra (`monitoring`), present in the job image and in the dev group.
