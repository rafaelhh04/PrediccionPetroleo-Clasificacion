"""Statistical tests and confidence intervals for out-of-sample predictions.

Daily direction labels and scores are serially dependent (volatility
clustering, regimes), so resampling days independently would understate the
uncertainty. Confidence intervals therefore use a **circular block bootstrap**
(Politis & Romano, 1992): blocks of ``block_length`` consecutive days are
drawn with replacement, wrapping around the end of the series.

Tests:

- **DeLong** (DeLong et al., 1988; fast form of Sun & Xu, 2014) compares two
  correlated ROC AUCs computed on the same days. It assumes independent days,
  so it is reported together with the block-bootstrap interval of the AUC
  difference, which does not.
- **One-sided binomial test** of accuracy against the no-information rate
  (accuracy of always predicting the most frequent class), as in caret's
  ``confusionMatrix``. Also assumes independent days.
- **Holm** step-down correction for the family of model-vs-baseline tests.

Calibration: Brier score, Brier skill score against a reference forecast and
the reliability diagram.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from brent_forecast._types import FloatArray, IntArray
from brent_forecast.evaluation.metrics import save_figure

ScoreFn = Callable[..., float]
"""Metric of ``(y, *scores)``; may return NaN when undefined on a resample."""


class BootstrapOptions(TypedDict):
    """Keyword arguments of :func:`bootstrap_interval`, shared by every call of a report."""

    block_length: int
    n_resamples: int
    confidence: float
    seed: int


@dataclass(frozen=True)
class Interval:
    """Point estimate and percentile bootstrap interval."""

    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        """Plain-dict form for JSON."""
        return asdict(self)


@dataclass(frozen=True)
class DeLongResult:
    """DeLong test of ``H0: AUC_1 = AUC_2`` (two-sided)."""

    auc_1: float
    auc_2: float
    z: float
    p_value: float


@dataclass(frozen=True)
class BinomialResult:
    """One-sided binomial test of ``H0: accuracy <= no-information rate``."""

    accuracy: float
    no_information_rate: float
    p_value: float


# ── bootstrap ─────────────────────────────────────────────────


def block_bootstrap_indices(
    n: int, block_length: int, n_resamples: int, rng: np.random.Generator
) -> IntArray:
    """Draw the indices of ``n_resamples`` circular block-bootstrap samples of length ``n``.

    Returns
    -------
    ndarray of shape (n_resamples, n)
        Each row concatenates blocks of ``block_length`` consecutive indices
        (mod ``n``) with uniformly random starts, truncated to ``n``.
    """
    if n < 1 or block_length < 1 or n_resamples < 1:
        raise ValueError("n, block_length and n_resamples must be >= 1")
    block_length = min(block_length, n)
    n_blocks = -(-n // block_length)
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)) % n
    return idx.reshape(n_resamples, -1)[:, :n]


def bootstrap_interval(
    metric: ScoreFn,
    y: npt.ArrayLike,
    *scores: npt.ArrayLike,
    block_length: int,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Interval:
    """Percentile block-bootstrap interval of ``metric(y, *scores)``.

    Resamples on which the metric is undefined (NaN, e.g. AUC with a single
    class) are discarded.
    """
    y_arr = np.asarray(y, dtype=float)
    arrays = [np.asarray(s, dtype=float) for s in scores]
    if any(len(a) != len(y_arr) for a in arrays):
        raise ValueError("y and every score array must have the same length")
    idx = block_bootstrap_indices(
        len(y_arr), block_length, n_resamples, np.random.default_rng(seed)
    )
    draws = np.array([metric(y_arr[i], *(a[i] for a in arrays)) for i in idx])
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        raise ValueError("The metric is undefined on every bootstrap resample")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(draws, [tail, 100 - tail])
    return Interval(float(metric(y_arr, *arrays)), float(low), float(high))


def auc(y: FloatArray, proba: FloatArray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (midranks), or NaN with a single class.

    Equal to ``sklearn.metrics.roc_auc_score`` but much cheaper, which matters
    inside the bootstrap loop.
    """
    positive = y == 1
    m = int(positive.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan")
    ranks = stats.rankdata(proba)
    return float((ranks[positive].sum() - m * (m + 1) / 2) / (m * n))


def auc_difference(y: FloatArray, proba_1: FloatArray, proba_2: FloatArray) -> float:
    """``AUC(proba_1) - AUC(proba_2)`` on the same days (NaN with a single class)."""
    return auc(y, proba_1) - auc(y, proba_2)


def accuracy(y: FloatArray, proba: FloatArray) -> float:
    """Accuracy of the 0.5-thresholded probabilities."""
    return float(np.mean((proba >= 0.5) == (y == 1)))


# ── tests ─────────────────────────────────────────────────────


def _midrank_components(y: FloatArray, proba: FloatArray) -> tuple[float, FloatArray, FloatArray]:
    """AUC and DeLong structural components ``V10`` (positives) and ``V01`` (negatives)."""
    pos, neg = proba[y == 1], proba[y == 0]
    m, n = len(pos), len(neg)
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    pos_ranks, neg_ranks = ranks[:m], ranks[m:]
    v10 = (pos_ranks - stats.rankdata(pos)) / n
    v01 = 1.0 - (neg_ranks - stats.rankdata(neg)) / m
    return float(v10.mean()), v10, v01


def delong_test(y: npt.ArrayLike, proba_1: npt.ArrayLike, proba_2: npt.ArrayLike) -> DeLongResult:
    """Two-sided DeLong test for the difference of two correlated ROC AUCs."""
    y_arr = np.asarray(y, dtype=float)
    if set(np.unique(y_arr)) != {0.0, 1.0}:
        raise ValueError("DeLong's test needs binary 0/1 labels with both classes present")
    auc_1, v10_1, v01_1 = _midrank_components(y_arr, np.asarray(proba_1, dtype=float))
    auc_2, v10_2, v01_2 = _midrank_components(y_arr, np.asarray(proba_2, dtype=float))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    cov = s10 / len(v10_1) + s01 / len(v01_1)
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = auc_1 - auc_2
    if variance <= 1e-15:  # identical rankings (e.g. two constant scores)
        z, p_value = (0.0, 1.0) if abs(diff) < 1e-12 else (float(np.sign(diff) * np.inf), 0.0)
    else:
        z = diff / np.sqrt(variance)
        p_value = float(2 * stats.norm.sf(abs(z)))
    return DeLongResult(auc_1, auc_2, float(z), p_value)


def accuracy_vs_no_information(y: npt.ArrayLike, proba: npt.ArrayLike) -> BinomialResult:
    """One-sided binomial test of accuracy against the no-information rate."""
    y_arr = np.asarray(y, dtype=float)
    hits = int(np.sum((np.asarray(proba, dtype=float) >= 0.5) == (y_arr == 1)))
    rate = float(max(y_arr.mean(), 1 - y_arr.mean()))
    test = stats.binomtest(hits, len(y_arr), rate, alternative="greater")
    return BinomialResult(hits / len(y_arr), rate, float(test.pvalue))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error rate control)."""
    names = sorted(p_values, key=lambda k: p_values[k])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * p_values[name]))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


# ── calibration ───────────────────────────────────────────────


def brier(y: npt.ArrayLike, proba: npt.ArrayLike) -> float:
    """Brier score (mean squared error of the probability; lower is better)."""
    return float(brier_score_loss(y, proba))


def brier_skill(y: npt.ArrayLike, proba: npt.ArrayLike, reference: npt.ArrayLike) -> float:
    """Brier skill score ``1 - BS / BS_ref``: > 0 beats the reference forecast."""
    return 1.0 - brier(y, proba) / brier(y, reference)


def plot_reliability(
    y: npt.ArrayLike, probas: Mapping[str, Any], n_bins: int, plots_dir: Path
) -> Path:
    """Reliability diagram (quantile bins) of every probabilistic forecast."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted P(up)")
    ax.set_ylabel("Observed frequency of up days")
    ax.set_title("Reliability diagram (out-of-sample)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, plots_dir, "reliability_diagram.png")
