"""Tests for brent_forecast.features.transformers (FeatureEngineer, VIFSelector, Winsorizer)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from brent_forecast.features.preprocessing import create_label, engineer_features
from brent_forecast.features.transformers import FeatureEngineer, VIFSelector, Winsorizer


@parametrize_with_checks([VIFSelector(), Winsorizer()])
def test_sklearn_compatible_transformer(estimator, check) -> None:
    check(estimator)


# ── FeatureEngineer ───────────────────────────────────────────


def test_feature_engineer_is_stateless_and_matches_engineer_features(
    merged_frame: pd.DataFrame,
) -> None:
    labelled = create_label(merged_frame.copy())
    expected, cols = engineer_features(labelled.copy())

    fe = FeatureEngineer().fit(labelled.iloc[:100])  # fitting on any subset changes nothing
    out = fe.transform(labelled)

    pd.testing.assert_frame_equal(out, expected[cols])
    assert fe.get_feature_names_out().tolist() == cols


# ── VIFSelector ───────────────────────────────────────────────


def _collinear(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, n))
    return pd.DataFrame({"a": a, "b": b, "a_copy": a + 1e-3 * rng.normal(size=n), "c": c})


def test_vif_drops_one_of_two_collinear_features() -> None:
    selector = VIFSelector(threshold=10.0).fit(_collinear())

    kept = selector.get_feature_names_out().tolist()

    assert ("a" in kept) != ("a_copy" in kept)
    assert {"b", "c"} <= set(kept)
    assert len(selector.dropped_vifs_) == 1
    assert max(selector.dropped_vifs_.values()) > 10.0
    assert max(selector.vifs_) <= 10.0


def test_vif_keeps_everything_under_a_high_threshold() -> None:
    selector = VIFSelector(threshold=1e12).fit(_collinear())

    assert selector.get_feature_names_out().tolist() == ["a", "b", "a_copy", "c"]


def test_vif_handles_perfect_collinearity() -> None:
    X = _collinear()
    X["a_exact"] = X["a"]

    kept = VIFSelector(threshold=10.0).fit(X).get_feature_names_out().tolist()

    assert sum(name in kept for name in ("a", "a_copy", "a_exact")) == 1


def test_vif_keeps_a_single_column() -> None:
    X = _collinear()[["c"]]

    assert VIFSelector(threshold=0.5).fit(X).transform(X).shape == (len(X), 1)


def test_vif_transform_selects_the_fitted_columns_only() -> None:
    train = _collinear()
    future = _collinear(50) * 1000  # anything at transform time

    selector = VIFSelector().fit(train)
    out = selector.transform(future)

    np.testing.assert_array_equal(out, future.to_numpy()[:, selector.support_])


# ── Winsorizer ────────────────────────────────────────────────


def test_winsorizer_clips_matching_columns_with_train_percentiles() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"lag_ret_1": rng.normal(size=1000), "vix": rng.normal(size=1000)})
    new = pd.DataFrame({"lag_ret_1": [100.0, -100.0], "vix": [100.0, -100.0]})

    w = Winsorizer(0.01, 0.99, name_contains="ret").fit(train)
    out = w.transform(new)

    lo, hi = np.percentile(train["lag_ret_1"], [1, 99])
    assert out[:, 0].tolist() == pytest.approx([hi, lo])
    assert out[:, 1].tolist() == [100.0, -100.0]  # "vix" is not a return feature
    assert w.columns_.tolist() == [0]


def test_winsorizer_does_not_modify_its_input() -> None:
    X = np.arange(100.0).reshape(-1, 1)
    original = X.copy()

    Winsorizer(0.1, 0.9).fit(X).transform(X)

    np.testing.assert_array_equal(X, original)


def test_winsorizer_without_matching_columns_is_identity() -> None:
    X = pd.DataFrame({"vix": [1.0, 2.0, 300.0]})

    out = Winsorizer(name_contains="ret").fit(X).transform(X)

    np.testing.assert_array_equal(out, X.to_numpy())


@pytest.mark.parametrize(("lower", "upper"), [(0.5, 0.5), (0.9, 0.1), (-0.1, 0.5), (0.1, 1.5)])
def test_winsorizer_rejects_invalid_bounds(lower: float, upper: float) -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        Winsorizer(lower, upper).fit(np.ones((5, 1)))


def test_transformers_keep_feature_names_in_a_pandas_pipeline() -> None:
    X = _collinear()
    X.columns = ["ret_a", "b", "ret_a_copy", "c"]

    pipe = make_pipeline(VIFSelector(), Winsorizer(name_contains="ret")).set_output(
        transform="pandas"
    )
    out = pipe.fit_transform(X)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == pipe.get_feature_names_out().tolist()
    assert "b" in out.columns
