"""Tests for brent_forecast.data.schemas: every contract rejects corrupted data."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brent_forecast.data.load import load_geopolitical_events, load_oil_prices
from brent_forecast.data.schemas import (
    EVENTS_SCHEMA,
    FEATURES_SCHEMA,
    INFERENCE_SCHEMA,
    OIL_SCHEMA,
    DataValidationError,
    validate,
)
from brent_forecast.data.stages import read_features
from brent_forecast.features.preprocessing import build_dataset
from conftest import make_events_frame, make_oil_frame

Corruption = Callable[[pd.DataFrame], None]


@pytest.fixture
def oil() -> pd.DataFrame:
    return make_oil_frame().head(300).reset_index(drop=True)


def test_the_synthetic_datasets_satisfy_every_contract(merged_frame: pd.DataFrame) -> None:
    oil = make_oil_frame()

    validate(oil, OIL_SCHEMA)
    validate(make_events_frame(oil), EVENTS_SCHEMA)
    dataset = build_dataset(merged_frame.copy())  # validates FEATURES_SCHEMA internally
    validate(dataset.features, INFERENCE_SCHEMA)


def _set(column: str, row: int, value: object) -> Corruption:
    def corrupt(df: pd.DataFrame) -> None:
        df[column] = df[column].astype(object)
        df.loc[row, column] = value

    return corrupt


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_set("brent_price", 5, -3.0), "column 'brent_price' failed in_range"),
        (_set("brent_price", 5, 8_150.0), "column 'brent_price' failed in_range"),  # cents
        (_set("brent_price", 5, np.nan), "column 'brent_price' failed not_nullable"),
        (_set("brent_return", 5, -150.0), "column 'brent_return' failed in_range"),
        (_set("vix", 5, 0.0), "column 'vix' failed in_range"),
        (_set("dxy_index", 5, 9_000.0), "column 'dxy_index' failed in_range"),
        (_set("vix", 5, "n/a"), "column 'vix' failed coerce_dtype"),
        (_set("brent_volatility_7d", 5, -1.0), "'brent_volatility_7d' failed greater_than"),
        (_set("date", 5, pd.Timestamp("2010-02-17")), "column 'date' failed field_uniqueness"),
        (
            lambda df: df.drop(columns=["gpr_index"], inplace=True),
            "missing required column\\(s\\): 'gpr_index'",
        ),
    ],
)
def test_oil_schema_rejects_corrupted_values(
    oil: pd.DataFrame, corrupt: Corruption, message: str
) -> None:
    corrupt(oil)

    with pytest.raises(DataValidationError, match=message):
        validate(oil, OIL_SCHEMA)


def test_oil_schema_allows_a_negative_wti_and_missing_market_data(oil: pd.DataFrame) -> None:
    oil.loc[10, "wti_price"] = -37.6  # 2020-04-20 really happened
    oil.loc[11, ["vix", "dxy_index", "gpr_index"]] = np.nan  # forward-filled later

    validate(oil, OIL_SCHEMA)


def test_every_failure_is_reported_at_once(oil: pd.DataFrame) -> None:
    oil.loc[3, "brent_price"] = -1.0
    oil.loc[4, "brent_price"] = -2.0
    oil.loc[5, "vix"] = 0.0

    with pytest.raises(DataValidationError) as info:
        validate(oil, OIL_SCHEMA)

    text = str(info.value)
    assert text.startswith("oil prices failed validation:")
    assert "brent_price' failed in_range(0.0, 1000.0) for 2 row(s), e.g. -1.0, -2.0" in text
    assert "column 'vix'" in text


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_set("event_severity", 0, 11), "column 'event_severity' failed in_range"),
        (_set("event_type", 0, None), "column 'event_type' failed not_nullable"),
    ],
)
def test_events_schema_rejects_corrupted_values(corrupt: Corruption, message: str) -> None:
    events = make_events_frame(make_oil_frame()).reset_index(drop=True)
    corrupt(events)

    with pytest.raises(DataValidationError, match=message):
        validate(events, EVENTS_SCHEMA)


def test_loaders_fail_fast_on_corrupted_files(tmp_path: Path) -> None:
    oil = make_oil_frame().head(20)
    oil.loc[3, "brent_price"] = -5.0
    oil.to_csv(tmp_path / "oil.csv", index=False)
    events = make_events_frame(make_oil_frame()).head(5)
    events.loc[1, "event_severity"] = 42
    events.to_csv(tmp_path / "events.csv", index=False)

    with pytest.raises(DataValidationError, match="oil prices"):
        load_oil_prices(tmp_path / "oil.csv")
    with pytest.raises(DataValidationError, match="geopolitical events"):
        load_geopolitical_events(tmp_path / "events.csv")


@pytest.fixture
def features(merged_frame: pd.DataFrame) -> pd.DataFrame:
    return build_dataset(merged_frame.copy()).frame.head(200).copy()


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_set("label", 3, 2.0), "column 'label' failed isin"),
        (_set("vix", 3, np.nan), "column 'vix' failed not_nullable"),
        (_set("rsi_14", 3, np.inf), "column 'rsi_14' failed must be finite"),
        (_set("next_return", 3, np.nan), "column 'next_return' failed not_nullable"),
        (lambda df: df.sort_values("date", ascending=False, inplace=True), "increasing order"),
    ],
)
def test_feature_schema_rejects_corrupted_values(
    features: pd.DataFrame, corrupt: Corruption, message: str
) -> None:
    corrupt(features)

    with pytest.raises(DataValidationError, match=message):
        validate(features, FEATURES_SCHEMA)


def test_a_tampered_parquet_dataset_is_rejected(features: pd.DataFrame, tmp_path: Path) -> None:
    features.loc[7, "macd"] = np.inf
    features.to_parquet(tmp_path / "dataset.parquet", index=False)

    with pytest.raises(DataValidationError, match="'macd' failed must be finite"):
        read_features(tmp_path / "dataset.parquet")


def test_inference_schema_needs_only_finite_features(features: pd.DataFrame) -> None:
    rows = features.drop(columns=["date", "label", "next_return"]).tail(3)

    validate(rows, INFERENCE_SCHEMA)
    rows.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError, match="inference features"):
        validate(rows, INFERENCE_SCHEMA)
