"""Tests for brent_forecast.features.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import SplitSettings
from brent_forecast.features.preprocessing import (
    _check_label_alignment,
    build_dataset,
    create_label,
    engineer_features,
    handle_nulls,
    split_temporal,
)

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


def test_create_label_drops_the_last_row_without_future(merged_frame: pd.DataFrame) -> None:
    out = create_label(merged_frame.copy())

    assert len(out) == len(merged_frame) - 1
    assert out["date"].iloc[-1] == merged_frame["date"].iloc[-2]
    assert not out["label"].isna().any()


def test_create_label_drops_days_followed_by_a_missing_return() -> None:
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=5),
            "brent_return": [0.5, -0.2, np.nan, 0.3, 0.1],
        }
    )

    out = create_label(df)

    # Day 1 (next return NaN) and day 4 (no next day) have no label.
    assert out["date"].tolist() == list(df["date"].iloc[[0, 2, 3]])
    assert out["label"].tolist() == [0.0, 1.0, 1.0]


def test_label_alignment_check_rejects_a_wrong_label() -> None:
    next_return = pd.Series([0.4, -0.1, np.nan])

    with pytest.raises(ValueError, match="Misaligned label at row 1"):
        _check_label_alignment(pd.Series([1.0, 1.0, np.nan]), next_return)


def test_label_alignment_check_rejects_a_label_without_future() -> None:
    next_return = pd.Series([0.4, np.nan])

    with pytest.raises(ValueError, match="must not be labelled"):
        _check_label_alignment(pd.Series([1.0, 0.0]), next_return)


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

    assert row["lag_ret_1"] == pytest.approx(
        np.log(row["brent_price"] / df.iloc[99]["brent_price"])
    )
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


# ── split_temporal ────────────────────────────────────────────

SPLIT = SplitSettings(train_end="2022-01-01", val_end="2024-01-01")


def _split_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2021-12-30", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2024-06-03"]
    )
    return pd.DataFrame({"date": dates, "x": np.arange(6.0), "label": [0, 1, 0, 1, 0, 1.0]})


def test_split_temporal_boundaries() -> None:
    parts = split_temporal(_split_frame(), SPLIT)

    assert parts.train["x"].tolist() == [0.0, 1.0]  # date < train_end
    assert parts.val["x"].tolist() == [2.0, 3.0]  # train_end <= date < val_end
    assert parts.test["x"].tolist() == [4.0, 5.0]  # date >= val_end


def test_split_temporal_is_a_partition(merged_frame: pd.DataFrame) -> None:
    frame = build_dataset(merged_frame.copy()).frame

    parts = split_temporal(frame, SPLIT)

    assert len(parts.train) + len(parts.val) + len(parts.test) == len(frame)
    assert min(len(parts.train), len(parts.val), len(parts.test)) > 0
    assert parts.train["date"].max() < parts.val["date"].min()
    assert parts.val["date"].max() < parts.test["date"].min()


# ── build_dataset ─────────────────────────────────────────────


def test_build_dataset(merged_frame: pd.DataFrame) -> None:
    ds = build_dataset(merged_frame.copy())

    assert ds.feature_cols == EXPECTED_FEATURES
    assert list(ds.frame.columns) == ["date", "label", *EXPECTED_FEATURES]
    assert not ds.frame.isna().any().any()
    assert ds.frame.index.tolist() == list(range(len(ds.frame)))
    pd.testing.assert_frame_equal(ds.features, ds.frame[EXPECTED_FEATURES])
    pd.testing.assert_series_equal(ds.target, ds.frame["label"])
    pd.testing.assert_series_equal(ds.dates, ds.frame["date"])


def test_gpr_change_is_a_21_day_difference(labelled: pd.DataFrame) -> None:
    gpr = labelled["gpr_index"].copy()

    df, _ = engineer_features(labelled.copy())

    expected = gpr - gpr.shift(21)
    pd.testing.assert_series_equal(df["gpr_change"], expected, check_names=False)
