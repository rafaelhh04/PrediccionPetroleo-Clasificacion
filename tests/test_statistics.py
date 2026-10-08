"""Tests for brent_forecast.evaluation.statistics."""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from brent_forecast.evaluation.statistics import (
    accuracy,
    accuracy_vs_no_information,
    auc,
    auc_difference,
    block_bootstrap_indices,
    bootstrap_interval,
    brier,
    brier_skill,
    delong_test,
    holm,
    plot_reliability,
)

BOOT = {"block_length": 10, "n_resamples": 500, "confidence": 0.95, "seed": 0}


def _signal(n: int = 400, strength: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(float)
    proba = 1 / (1 + np.exp(-(strength * (2 * y - 1) + rng.normal(size=n))))
    return y, proba


# ── block bootstrap ───────────────────────────────────────────


def test_block_bootstrap_indices_are_circular_blocks() -> None:
    idx = block_bootstrap_indices(23, 5, 50, np.random.default_rng(0))

    assert idx.shape == (50, 23)
    assert idx.min() >= 0
    assert idx.max() < 23
    blocks = idx[:, :20].reshape(50, 4, 5)
    np.testing.assert_array_equal(np.diff(blocks, axis=2) % 23, 1)  # consecutive mod n


def test_block_bootstrap_is_reproducible_and_validates() -> None:
    a = block_bootstrap_indices(30, 4, 10, np.random.default_rng(7))
    b = block_bootstrap_indices(30, 4, 10, np.random.default_rng(7))

    np.testing.assert_array_equal(a, b)
    assert block_bootstrap_indices(3, 10, 2, np.random.default_rng(0)).shape == (2, 3)
    with pytest.raises(ValueError, match=">= 1"):
        block_bootstrap_indices(0, 1, 1, np.random.default_rng(0))


def test_blocks_widen_the_interval_of_a_serially_dependent_series() -> None:
    """With persistent regimes, i.i.d. resampling (block 1) understates the uncertainty."""
    y = np.repeat(np.random.default_rng(0).random(40) < 0.5, 25).astype(float)

    def mean(values: np.ndarray) -> float:
        return float(values.mean())

    iid = bootstrap_interval(mean, y, **{**BOOT, "block_length": 1})
    block = bootstrap_interval(mean, y, **{**BOOT, "block_length": 50})

    assert block.high - block.low > 2 * (iid.high - iid.low)


def test_bootstrap_interval_brackets_the_estimate() -> None:
    y, proba = _signal()

    interval = bootstrap_interval(auc, y, proba, **BOOT)

    assert interval.estimate == pytest.approx(roc_auc_score(y, proba))
    assert interval.low < interval.estimate < interval.high
    assert interval.low > 0.5  # a real signal is detected
    assert interval.as_dict() == {
        "estimate": interval.estimate,
        "low": interval.low,
        "high": interval.high,
    }


def test_bootstrap_interval_contains_chance_for_noise() -> None:
    y, _ = _signal()
    noise = np.random.default_rng(1).random(len(y))

    interval = bootstrap_interval(auc, y, noise, **BOOT)

    assert interval.low < 0.5 < interval.high


def test_bootstrap_interval_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_interval(auc, np.ones(5), np.ones(4), **BOOT)
    with pytest.raises(ValueError, match="undefined"):
        bootstrap_interval(auc, np.ones(20), np.linspace(0, 1, 20), **BOOT)


def test_metric_helpers() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    p = np.array([0.2, 0.7, 0.4, 0.6])

    assert accuracy(y, p) == 0.5
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))
    ties = np.round(np.random.default_rng(0).random(50), 1)
    labels = (np.random.default_rng(1).random(50) < 0.5).astype(float)
    assert auc(labels, ties) == pytest.approx(roc_auc_score(labels, ties))
    assert np.isnan(auc(np.ones(4), p))
    assert auc_difference(y, p, 1 - p) == pytest.approx(auc(y, p) - auc(y, 1 - p))


# ── DeLong ────────────────────────────────────────────────────


def _delong_brute_force(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """O(m·n) DeLong z statistic straight from the definition."""

    def components(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, neg = p[y == 1], p[y == 0]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(axis=1), psi.mean(axis=0)

    v10_1, v01_1 = components(p1)
    v10_2, v01_2 = components(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s = s10 / len(v10_1) + s01 / len(v01_1)
    return float((v10_1.mean() - v10_2.mean()) / np.sqrt(s[0, 0] + s[1, 1] - 2 * s[0, 1]))


def test_delong_matches_the_definition_with_ties() -> None:
    y, p1 = _signal(n=200)
    p2 = np.round(np.random.default_rng(3).random(200), 1)  # many ties

    result = delong_test(y, p1, p2)

    assert result.auc_1 == pytest.approx(roc_auc_score(y, p1))
    assert result.auc_2 == pytest.approx(roc_auc_score(y, p2))
    assert result.z == pytest.approx(_delong_brute_force(y, p1, p2))
    assert result.p_value == pytest.approx(2 * stats.norm.sf(abs(result.z)))


def test_delong_detects_a_better_model_and_not_identical_ones() -> None:
    y, strong = _signal(strength=1.5)
    weak = np.random.default_rng(5).random(len(y))

    assert delong_test(y, strong, weak).p_value < 1e-6
    assert delong_test(y, strong, strong).p_value == 1.0
    constant = np.full(len(y), 0.5)
    assert delong_test(y, constant, constant).p_value == 1.0


def test_delong_has_the_nominal_size_under_the_null() -> None:
    rng = np.random.default_rng(0)
    rejections = 0
    n_sims = 400
    for _ in range(n_sims):
        y = (rng.random(200) < 0.5).astype(float)
        rejections += delong_test(y, rng.random(200), rng.random(200)).p_value < 0.05

    assert 0.02 < rejections / n_sims < 0.09


def test_delong_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        delong_test(np.ones(5), np.ones(5), np.ones(5))


# ── binomial, Holm, calibration ───────────────────────────────


def test_accuracy_vs_no_information_rate() -> None:
    y = np.array([1.0] * 60 + [0.0] * 40)

    majority = accuracy_vs_no_information(y, np.full(100, 0.9))
    perfect = accuracy_vs_no_information(y, y)

    assert majority.accuracy == majority.no_information_rate == 0.6
    assert majority.p_value > 0.5  # always predicting the majority is not skill
    assert perfect.p_value == pytest.approx(stats.binomtest(100, 100, 0.6, "greater").pvalue)
    assert perfect.p_value < 1e-10


def test_holm_matches_statsmodels() -> None:
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}

    adjusted = holm(p)

    expected = multipletests(list(p.values()), method="holm")[1]
    assert list(adjusted) == list(p)
    np.testing.assert_allclose(list(adjusted.values()), expected)


def test_brier_and_skill() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    climatology = np.full(4, 0.5)

    assert brier(y, y) == 0.0
    assert brier(y, climatology) == 0.25
    assert brier_skill(y, y, climatology) == 1.0
    assert brier_skill(y, climatology, climatology) == 0.0
    assert brier_skill(y, 1 - y, climatology) < 0


def test_plot_reliability(plots_dir: Path) -> None:
    y, proba = _signal()

    path = plot_reliability(y, {"model": proba, "noise": np.full(len(y), 0.5)}, 5, plots_dir)

    assert path.name == "reliability_diagram.png"
    assert path.stat().st_size > 0
