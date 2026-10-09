"""Tests for brent_forecast.data.load: reading, merging and validating the datasets."""

from pathlib import Path

import pandas as pd
import pytest

from brent_forecast.data.load import (
    _validate,
    load_geopolitical_events,
    load_oil_data,
    load_oil_prices,
)
from brent_forecast.data.schemas import DataValidationError
from conftest import DATE_BOUNDS, EVENTS_FILENAME, OIL_FILENAME, make_oil_frame, write_csv


def test_load_oil_prices_parses_dates_and_strips_headers(tmp_path: Path) -> None:
    oil = make_oil_frame().head(3)
    path = tmp_path / "oil.csv"
    oil.rename(columns={"date": " date "}).to_csv(path, index=False)

    df = load_oil_prices(path)

    assert list(df.columns) == list(oil.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 3


def test_load_oil_prices_drops_unparsable_dates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    oil = make_oil_frame().head(3).astype({"date": str})
    oil.loc[1, "date"] = "not-a-date"
    path = tmp_path / "oil.csv"
    oil.to_csv(path, index=False)

    df = load_oil_prices(path)

    assert len(df) == 2
    assert "unparsable" in caplog.text


@pytest.mark.parametrize("loader", [load_oil_prices, load_geopolitical_events])
def test_loaders_require_a_date_column(tmp_path: Path, loader) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("day,value\n2010-01-04,1\n")

    with pytest.raises(ValueError, match="Column 'date' not found"):
        loader(path)


def test_load_geopolitical_events_prefixes_event_columns(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "date,event_type,event_description,event_severity\n2011-02-15,war,Libya,8\nbad,opec,cut,5\n"
    )

    df = load_geopolitical_events(path)

    assert list(df.columns) == [
        "date",
        "geo_event_type",
        "geo_event_description",
        "geo_event_severity",
    ]
    assert len(df) == 1  # the unparsable date is dropped


def test_merge_keeps_one_row_per_oil_day(
    merged_frame: pd.DataFrame, oil_frame: pd.DataFrame
) -> None:
    assert len(merged_frame) == len(oil_frame)
    assert merged_frame["date"].is_monotonic_increasing
    assert not merged_frame["date"].duplicated().any()


def test_merge_promotes_geo_columns_and_keeps_oil_ones(merged_frame: pd.DataFrame) -> None:
    cols = set(merged_frame.columns)
    assert {"event_type", "event_description", "event_severity"} <= cols
    assert {"oil_event_type", "oil_event_description", "oil_event_severity"} <= cols
    assert not any(c.startswith("geo_") for c in cols)


def test_merge_fills_days_without_event(
    merged_frame: pd.DataFrame, events_frame: pd.DataFrame
) -> None:
    event_days = set(pd.to_datetime(events_frame["date"]))
    no_event = merged_frame[~merged_frame["date"].isin(event_days)]
    with_event = merged_frame[merged_frame["date"].isin(event_days)]

    assert (no_event["event_type"] == "none").all()
    assert (no_event["event_description"] == "none").all()
    assert (no_event["event_severity"] == 0).all()
    assert len(with_event) == len(events_frame)
    assert (with_event["event_severity"] > 0).all()


def test_load_oil_data_from_csv_files(raw_data_dir: Path, oil_frame: pd.DataFrame) -> None:
    df = load_oil_data(raw_data_dir / OIL_FILENAME, raw_data_dir / EVENTS_FILENAME, DATE_BOUNDS)

    assert len(df) == len(oil_frame)


def test_load_oil_data_rejects_duplicated_event_dates(
    tmp_path: Path, oil_frame: pd.DataFrame, events_frame: pd.DataFrame
) -> None:
    events = pd.concat([events_frame, events_frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="geopolitical events failed validation"):
        load_oil_data(
            write_csv(oil_frame, tmp_path / OIL_FILENAME),
            write_csv(events, tmp_path / EVENTS_FILENAME),
            DATE_BOUNDS,
        )


# ── _validate: one test per check ─────────────────────────────


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-02-17", "2018-06-01", "2026-03-12"]),
            "event_type": ["none", "war", "none"],
            "event_severity": [0, 8, 0],
        }
    )


def test_validate_accepts_a_clean_frame(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")

    _validate(_valid_frame(), expected_rows=3, bounds=DATE_BOUNDS)

    assert "Validation passed: 3 rows" in caplog.text


def test_validate_rejects_duplicated_dates(caplog: pytest.LogCaptureFixture) -> None:
    df = _valid_frame()
    df.loc[1, "date"] = df.loc[0, "date"]

    with pytest.raises(DataValidationError, match="column 'date' failed field_uniqueness"):
        _validate(df, expected_rows=3, bounds=DATE_BOUNDS)
    assert "merged dataset failed validation" in caplog.text


def test_validate_rejects_unsorted_dates() -> None:
    df = _valid_frame().iloc[[1, 0, 2]].reset_index(drop=True)

    with pytest.raises(DataValidationError, match="sorted in increasing order"):
        _validate(df, expected_rows=3, bounds=DATE_BOUNDS)


def test_validate_rejects_row_count_mismatch() -> None:
    with pytest.raises(DataValidationError, match="must keep 4 rows"):
        _validate(_valid_frame(), expected_rows=4, bounds=DATE_BOUNDS)


@pytest.mark.parametrize("column", ["event_severity", "event_type"])
def test_validate_rejects_null_events(column: str) -> None:
    df = _valid_frame()
    df[column] = df[column].astype(object)
    df.loc[0, column] = None

    with pytest.raises(DataValidationError, match=f"column '{column}' failed not_nullable"):
        _validate(df, expected_rows=3, bounds=DATE_BOUNDS)


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (["2005-01-03", "2018-06-01", "2026-03-12"], "first date must be between"),
        (["2012-01-03", "2018-06-01", "2026-03-12"], "first date must be between"),
        (["2010-02-17", "2018-06-01", "2023-12-29"], "last date must be on or after"),
    ],
)
def test_validate_rejects_unexpected_date_range(dates: list[str], message: str) -> None:
    df = _valid_frame()
    df["date"] = pd.to_datetime(dates)

    with pytest.raises(DataValidationError, match=message):
        _validate(df, expected_rows=3, bounds=DATE_BOUNDS)


def test_validate_reports_every_error_at_once() -> None:
    df = _valid_frame()
    df.loc[1, "date"] = df.loc[0, "date"]

    with pytest.raises(DataValidationError) as info:
        _validate(df, expected_rows=5, bounds=DATE_BOUNDS)

    assert "field_uniqueness" in str(info.value)
    assert "must keep 5 rows" in str(info.value)


def test_validate_uses_the_configured_bounds() -> None:
    from datetime import date

    from brent_forecast.data.load import DateBounds

    strict = DateBounds(date(2010, 3, 1), date(2010, 12, 31), date(2027, 1, 1))

    with pytest.raises(DataValidationError) as info:
        _validate(_valid_frame(), expected_rows=3, bounds=strict)

    assert "between 2010-03-01 and 2010-12-31" in str(info.value)
    assert "on or after 2027-01-01" in str(info.value)
