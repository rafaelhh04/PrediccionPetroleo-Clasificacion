"""Tests for brent_forecast.features.preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from brent_forecast.config import PreprocessingSettings, SplitSettings
from brent_forecast.features.preprocessing import (
    create_label,
    engineer_features,
    filter_features_by_vif,
    handle_nulls,
    preprocess,
    scale_features,
    split_temporal,
    winsorize_features,
)

TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-01-01")
EXPECTED_FEATURES = [
    "dxy_index",
    "vix",
    "brent_volatility_7d",
    "brent_volatility_30d",
    "brent_wti_spread",
    "lag_ret_1",
    "lag_ret_3",
    "lag_ret_7",
    "gpr_change",
    "event_flag_binary",
    "high_severity_flag",
    "vol_ratio",
    "day_of_week",
    "month",
]


@pytest.fixture
def labelled(merged_frame: pd.DataFrame) -> pd.DataFrame:
    return create_label(merged_frame.copy())


@pytest.fixture
def engineered(labelled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return engineer_features(labelled)


# ── create_label ──────────────────────────────────────────────


def test_create_label_is_next_day_direction(labelled: pd.DataFrame) -> None:
    next_up = (labelled["brent_return"].shift(-1) > 0).astype(float)

    pd.testing.assert_series_equal(
        labelled["label"].iloc[:-1], next_up.iloc[:-1], check_names=False
    )
    assert set(labelled["label"].unique()) <= {0.0, 1.0}


def test_create_label_sorts_by_date(merged_frame: pd.DataFrame) -> None:
    shuffled = merged_frame.sample(frac=1.0, random_state=0)

    out = create_label(shuffled)

    assert out["date"].is_monotonic_increasing
    pd.testing.assert_frame_equal(out, create_label(merged_frame.copy()))


@pytest.mark.xfail(
    strict=True,
    reason="Known bug (fixed in Phase 3): NaN > 0 is False, so the last row keeps label 0 "
    "instead of being dropped.",
)
def test_create_label_drops_the_last_row_without_future(merged_frame: pd.DataFrame) -> None:
    out = create_label(merged_frame.copy())

    assert len(out) == len(merged_frame) - 1


# ── engineer_features ─────────────────────────────────────────


def test_engineer_features_selects_expected_columns(
    engineered: tuple[pd.DataFrame, list[str]],
) -> None:
    df, cols = engineered

    assert cols == EXPECTED_FEATURES
    assert all(df[c].dtype != object for c in cols)


@pytest.mark.parametrize(
    "excluded",
    [
        "date",
        "label",
        "brent_return",
        "wti_return",
        "brent_price",
        "wti_price",
        "gpr_index",
        "event_flag",
        "event_severity",
        "wti_volatility_7d",
        "brent_lag_1",
        "wti_lag_7",
    ],
)
def test_engineer_features_excludes_leaky_or_raw_columns(
    engineered: tuple[pd.DataFrame, list[str]], excluded: str
) -> None:
    _, cols = engineered

    assert excluded not in cols


def test_engineer_features_values(engineered: tuple[pd.DataFrame, list[str]]) -> None:
    df, _ = engineered
    row = df.iloc[100]

    assert row["lag_ret_1"] == pytest.approx(np.log(row["brent_price"] / df.iloc[99]["brent_price"]))
    assert np.isfinite(row["gpr_change"])  # defined once the 21-day window is full
    assert row["day_of_week"] == row["date"].dayofweek
    assert row["month"] == row["date"].month
    assert row["vol_ratio"] == pytest.approx(
        row["brent_volatility_7d"] / (row["brent_volatility_30d"] + 1e-10)
    )
    assert "wti_return" not in df.columns
    assert "gpr_index" not in df.columns


def test_engineer_features_event_flags() -> None:
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=30),
            "brent_price": 50.0,
            "brent_lag_1": 50.0,
            "brent_lag_3": 50.0,
            "brent_lag_7": 50.0,
            "gpr_index": 100.0,
            "brent_volatility_7d": 1.0,
            "brent_volatility_30d": 1.0,
            "event_severity": [0, 3, 6, 7, 10] * 6,
        }
    )

    out, _ = engineer_features(df)

    assert out["event_flag_binary"].tolist()[:5] == [0, 1, 1, 1, 1]
    assert out["high_severity_flag"].tolist()[:5] == [0, 0, 0, 1, 1]


# ── handle_nulls ──────────────────────────────────────────────


def test_handle_nulls_drops_warmup_rows_and_ffills_market_data(
    engineered: tuple[pd.DataFrame, list[str]],
) -> None:
    df, cols = engineered
    n_before = len(df)

    out = handle_nulls(df.copy(), cols)

    assert not out[cols].isna().any().any()
    assert 21 <= n_before - len(out) <= 35  # gpr_change warm-up dominates
    assert out.index.tolist() == list(range(len(out)))


def test_handle_nulls_forward_fills_vix() -> None:
    df = pd.DataFrame({"vix": [10.0, np.nan, 12.0], "x": [1.0, 2.0, 3.0]})

    out = handle_nulls(df, ["vix", "x"])

    assert out["vix"].tolist() == [10.0, 10.0, 12.0]


# ── VIF filter ────────────────────────────────────────────────


def _collinear_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2019-01-01", periods=n),
            "a": a,
            "b": b,
            "a_copy": a + 1e-3 * rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )


def test_vif_drops_collinear_features() -> None:
    df = _collinear_frame()

    kept = filter_features_by_vif(df, ["a", "b", "a_copy", "c"], TRAIN_END, vif_threshold=10.0)

    assert ("a" in kept) != ("a_copy" in kept)
    assert {"b", "c"} <= set(kept)


def test_vif_keeps_everything_under_a_high_threshold() -> None:
    df = _collinear_frame()

    kept = filter_features_by_vif(df, ["a", "b", "a_copy", "c"], TRAIN_END, vif_threshold=1e12)

    assert kept == ["a", "b", "a_copy", "c"]


def test_vif_preserves_order_and_stops_at_one_feature() -> None:
    df = _collinear_frame()

    assert filter_features_by_vif(df, ["c"], TRAIN_END, vif_threshold=0.5) == ["c"]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vif_handles_perfect_collinearity() -> None:
    df = _collinear_frame()
    df["a_exact"] = df["a"]

    kept = filter_features_by_vif(df, ["a", "a_exact", "b"], TRAIN_END, vif_threshold=10.0)

    assert ("a" in kept) != ("a_exact" in kept)
    assert "b" in kept


# ── split_temporal ────────────────────────────────────────────


def _split_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2021-12-30", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2024-06-03"]
    )
    return pd.DataFrame({"date": dates, "x": np.arange(6.0), "label": [0, 1, 0, 1, 0, 1.0]})


def test_split_temporal_boundaries() -> None:
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_temporal(_split_frame(), ["x"], TRAIN_END, VAL_END)

    assert X_tr.ravel().tolist() == [0.0, 1.0]  # date < train_end
    assert X_va.ravel().tolist() == [2.0, 3.0]  # train_end <= date < val_end
    assert X_te.ravel().tolist() == [4.0, 5.0]  # date >= val_end
    assert y_tr.tolist() == [0.0, 1.0]
    assert y_va.tolist() == [0.0, 1.0]
    assert y_te.tolist() == [0.0, 1.0]


def test_split_temporal_is_a_partition(engineered: tuple[pd.DataFrame, list[str]]) -> None:
    df, cols = engineered
    df = handle_nulls(df.copy(), cols)

    X_tr, X_va, X_te, *_ = split_temporal(df, cols, TRAIN_END, VAL_END)

    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
    assert min(len(X_tr), len(X_va), len(X_te)) > 0


# ── winsorize / scale ─────────────────────────────────────────


def test_winsorize_clips_only_return_features_with_train_percentiles() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(1000, 2))
    X_val = np.array([[100.0, 100.0], [-100.0, -100.0]])
    X_test = X_val.copy()
    lo = np.percentile(X_train[:, 0], 1)
    hi = np.percentile(X_train[:, 0], 99)

    out_tr, out_va, out_te = winsorize_features(
        X_train, X_val, X_test, ["lag_ret_1", "vix"], lower=0.01, upper=0.99
    )

    assert out_va[:, 0].tolist() == pytest.approx([hi, lo])
    assert out_te[:, 0].tolist() == pytest.approx([hi, lo])
    assert out_va[:, 1].tolist() == [100.0, -100.0]  # "vix" is not a return feature
    assert out_tr[:, 0].min() >= lo
    assert out_tr[:, 0].max() <= hi


def test_winsorize_modifies_in_place() -> None:
    X = np.arange(100.0).reshape(-1, 1)

    out, _, _ = winsorize_features(X, X[:1].copy(), X[:1].copy(), ["ret"], 0.0, 0.9)

    assert out is X
    assert X.max() == pytest.approx(np.percentile(np.arange(100.0), 90))


def test_scale_features_fits_on_train_only() -> None:
    rng = np.random.default_rng(1)
    X_train = rng.normal(5, 2, size=(500, 3))
    X_val = rng.normal(50, 20, size=(50, 3))

    tr, va, te, scaler = scale_features(X_train, X_val, X_val.copy())

    assert isinstance(scaler, StandardScaler)
    np.testing.assert_allclose(tr.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(tr.std(axis=0), 1, atol=1e-12)
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
    np.testing.assert_allclose(va, te)
    assert va.mean() > 5  # validation is transformed, not re-fitted


# ── full pipeline ─────────────────────────────────────────────


def test_preprocess_end_to_end(merged_frame: pd.DataFrame) -> None:
    split = SplitSettings(train_end="2022-01-01", val_end="2024-01-01")
    prep = PreprocessingSettings(vif_threshold=10.0, winsor_lower=0.01, winsor_upper=0.99)

    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, cols = preprocess(merged_frame.copy(), split, prep)

    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1] == len(cols) == scaler.n_features_in_
    assert set(cols) <= set(EXPECTED_FEATURES)
    assert len(X_tr) == len(y_tr)
    assert len(X_va) == len(y_va)
    assert len(X_te) == len(y_te)
    assert not np.isnan(X_tr).any()
    np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=1e-10)


def test_gpr_change_is_a_21_day_difference(labelled: pd.DataFrame) -> None:
    gpr = labelled["gpr_index"].copy()

    df, _ = engineer_features(labelled.copy())

    expected = gpr - gpr.shift(21)
    pd.testing.assert_series_equal(df["gpr_change"], expected, check_names=False)
