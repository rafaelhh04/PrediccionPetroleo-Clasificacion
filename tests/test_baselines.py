"""Tests for brent_forecast.models.baselines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from brent_forecast.models.baselines import (
    BASELINE_NAMES,
    PERSISTENCE_FEATURE,
    PersistenceClassifier,
    build_baseline,
)
from brent_forecast.validation.walk_forward import PurgedWalkForwardSplit, walk_forward_predict


def _data(n: int = 400, seed: int = 0, p_up: float = 0.6) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({PERSISTENCE_FEATURE: rng.normal(size=n), "vix": rng.normal(20, 2, n)})
    y = pd.Series((rng.random(n) < p_up).astype(float))
    return X, y


def test_majority_predicts_the_training_frequency() -> None:
    X, y = _data()

    proba = build_baseline("majority", seed=0).fit(X, y).predict_proba(X)[:, 1]

    np.testing.assert_allclose(proba, y.mean())


def test_buy_and_hold_always_predicts_up() -> None:
    X, y = _data()

    model = build_baseline("buy_and_hold", seed=0).fit(X, y)

    assert (model.predict(X) == 1.0).all()
    assert (model.predict_proba(X)[:, 1] == 1.0).all()


def test_stratified_is_random_reproducible_and_uninformative() -> None:
    X, y = _data(n=4000)

    a = build_baseline("stratified", seed=3).fit(X, y).predict_proba(X)[:, 1]
    b = build_baseline("stratified", seed=3).fit(X, y).predict_proba(X)[:, 1]

    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) == {0.0, 1.0}
    assert abs(a.mean() - y.mean()) < 0.03
    assert abs(roc_auc_score(y, a) - 0.5) < 0.03


def test_persistence_repeats_todays_direction() -> None:
    X = pd.DataFrame({PERSISTENCE_FEATURE: [0.02, -0.01, 0.0, 0.5], "vix": 20.0})
    y = pd.Series([1.0, 0.0, 1.0, 0.0])

    model = PersistenceClassifier().fit(X, y)

    assert model.predict(X).tolist() == [1.0, 0.0, 0.0, 1.0]  # a flat day counts as "down"
    assert model.predict_proba(X)[:, 1].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_persistence_is_perfect_on_a_trending_series() -> None:
    returns = pd.Series([0.1] * 5 + [-0.1] * 5)
    X = pd.DataFrame({PERSISTENCE_FEATURE: returns})
    y = (returns.shift(-1).fillna(-0.1) > 0).astype(float)  # tomorrow's direction

    proba = PersistenceClassifier().fit(X, y).predict_proba(X)[:, 1]

    assert (proba == y).mean() == 0.9  # only the turning point is missed


def test_persistence_input_validation() -> None:
    X, y = _data()

    with pytest.raises(ValueError, match="not found"):
        PersistenceClassifier(column="missing").fit(X, y)
    with pytest.raises(ValueError, match="Only binary"):
        PersistenceClassifier().fit(X, np.arange(len(y)) % 3)


def test_unknown_baseline() -> None:
    with pytest.raises(KeyError, match="Unknown baseline"):
        build_baseline("oracle", seed=0)


@pytest.mark.parametrize("key", list(BASELINE_NAMES))
def test_every_baseline_runs_through_walk_forward(key: str) -> None:
    X, y = _data()
    cv = PurgedWalkForwardSplit(None, test_size=50, first_test_index=300, purge=1, embargo=2)

    preds = walk_forward_predict(build_baseline(key, seed=0), X, y, cv)

    assert len(preds.proba) == 100
    assert preds.proba.between(0, 1).all()
