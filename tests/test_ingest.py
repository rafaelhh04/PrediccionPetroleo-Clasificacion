"""Tests for brent_forecast.data.ingest: incremental ingestion without network access."""

import json
import urllib.error
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import Settings
from brent_forecast.data.ingest import (
    IngestionError,
    derivation_gap,
    derive_market_columns,
    fred_series,
    http_get,
    ingest,
    yahoo_series,
)
from brent_forecast.data.schemas import DataValidationError
from conftest import EVENTS_FILENAME, OIL_FILENAME, make_events_frame, make_oil_frame, write_csv

HOLD_OUT = 20
SYMBOLS = {"BZ=F": "brent_price", "CL=F": "wti_price", "DX-Y.NYB": "dxy_index"}


def _yahoo_body(values: pd.Series) -> str:
    """Chart API response: one timestamp per day at 14:00 UTC (morning in New York)."""
    stamps = [int((day + pd.Timedelta(hours=14)).timestamp()) for day in values.index]
    closes = [None if pd.isna(v) else float(v) for v in values]
    result = {
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": stamps,
        "indicators": {"quote": [{"close": closes}]},
    }
    return json.dumps({"chart": {"result": [result], "error": None}})


def _fred_body(values: pd.Series) -> str:
    rows = [f"{d.date()},{'.' if pd.isna(v) else v}" for d, v in values.items()]
    return "observation_date,VIXCLS\n" + "\n".join(rows) + "\n"


class FakeMarket:
    """Serves the synthetic market as Yahoo/FRED responses and records the URLs asked."""

    def __init__(self, full: pd.DataFrame) -> None:
        self.full = full.set_index("date")
        self.urls: list[str] = []

    def __call__(self, url: str) -> str:
        self.urls.append(url)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "yahoo" in parsed.netloc:
            symbol = unquote(parsed.path.rsplit("/", 1)[-1])
            start = pd.Timestamp(int(query["period1"][0]), unit="s")
            end = pd.Timestamp(int(query["period2"][0]), unit="s")
            column = self.full[SYMBOLS[symbol]]
            return _yahoo_body(column[(column.index >= start) & (column.index < end)])
        start, end = pd.Timestamp(query["cosd"][0]), pd.Timestamp(query["coed"][0])
        return _fred_body(self.full["vix"].loc[start:end])


@pytest.fixture
def market(settings: Settings) -> tuple[pd.DataFrame, FakeMarket]:
    """Raw snapshot without its last HOLD_OUT days; the fake market still has them."""
    full = make_oil_frame()
    data_dir = settings.paths.data_dir
    data_dir.mkdir(parents=True)
    write_csv(full.iloc[:-HOLD_OUT], data_dir / OIL_FILENAME)
    write_csv(make_events_frame(full), data_dir / EVENTS_FILENAME)
    return full, FakeMarket(full)


def test_derived_columns_match_the_snapshot_definitions() -> None:
    oil = make_oil_frame().head(400)

    derived = derive_market_columns(oil)

    for column in ("brent_return", "brent_lag_7", "brent_volatility_30d", "wti_volatility_7d"):
        np.testing.assert_allclose(derived[column], oil[column], equal_nan=True)
    assert derivation_gap(oil) < 1e-9


def test_derivation_gap_detects_a_different_definition() -> None:
    oil = make_oil_frame().head(400)
    oil["brent_return"] = oil["brent_return"] / 100  # fraction instead of percent

    assert derivation_gap(oil) > 0.1


def test_ingest_appends_only_the_new_days_with_consistent_columns(
    settings: Settings, market: tuple[pd.DataFrame, FakeMarket]
) -> None:
    full, fake = market
    last_day = full["date"].iloc[-1].date()

    result = ingest(settings, end=last_day, fetch_text=fake)

    live = pd.read_csv(result.path, parse_dates=["date"])
    assert result.new_rows == HOLD_OUT
    assert result.first_new == full["date"].iloc[-HOLD_OUT].date()
    assert result.last_date == last_day
    assert result.path == settings.paths.live_dir / OIL_FILENAME
    assert len(live) == len(full)
    new = live.tail(HOLD_OUT).reset_index(drop=True)
    expected = full.tail(HOLD_OUT).reset_index(drop=True)
    for column in ("brent_price", "wti_price", "dxy_index", "brent_return", "brent_lag_3"):
        np.testing.assert_allclose(new[column], expected[column], equal_nan=True, rtol=1e-9)
    np.testing.assert_allclose(new["brent_volatility_30d"], expected["brent_volatility_30d"])
    assert (new["gpr_index"] == full["gpr_index"].iloc[-HOLD_OUT - 1]).all()  # carried forward
    assert (new["event_type"] == "none").all()
    assert (new["event_severity"] == 0).all()
    # The versioned raw snapshot is never modified.
    raw = pd.read_csv(settings.paths.data_dir / OIL_FILENAME)
    assert len(raw) == len(full) - HOLD_OUT


def test_ingest_is_idempotent_and_extends_the_live_file(
    settings: Settings, market: tuple[pd.DataFrame, FakeMarket]
) -> None:
    full, fake = market
    middle = full["date"].iloc[-10].date()
    last_day = full["date"].iloc[-1].date()

    first = ingest(settings, end=middle, fetch_text=fake)
    again = ingest(settings, end=middle, fetch_text=fake)
    rest = ingest(settings, end=last_day, fetch_text=fake)

    assert first.new_rows == HOLD_OUT - 9
    assert again.new_rows == 0
    assert rest.new_rows == 9
    assert len(pd.read_csv(rest.path)) == len(full)


def test_ingest_requests_the_configured_sources(
    settings: Settings, market: tuple[pd.DataFrame, FakeMarket]
) -> None:
    full, fake = market

    ingest(settings, end=full["date"].iloc[-1].date(), fetch_text=fake)

    hosts = sorted(urlparse(u).netloc for u in fake.urls)
    assert hosts.count("query1.finance.yahoo.com") == 3
    assert hosts.count("fred.stlouisfed.org") == 1
    assert any("DX-Y.NYB" in unquote(u) for u in fake.urls)
    first_new = full["date"].iloc[-HOLD_OUT].date().isoformat()
    assert any(f"cosd={first_new}" in u for u in fake.urls)


def test_ingest_without_new_brent_days(
    settings: Settings, market: tuple[pd.DataFrame, FakeMarket]
) -> None:
    full, _ = market

    def quiet(url: str) -> str:
        return _fred_body(pd.Series(dtype=float)) if "fred" in url else _yahoo_body(pd.Series())

    result = ingest(settings, end=full["date"].iloc[-1].date(), fetch_text=quiet)

    assert result.new_rows == 0
    assert result.path.is_file()


def test_ingest_rejects_corrupted_live_data(
    settings: Settings, market: tuple[pd.DataFrame, FakeMarket]
) -> None:
    full, fake = market
    fake.full.loc[fake.full.index[-3], "brent_price"] = -5.0  # a bad tick from the source

    with pytest.raises(DataValidationError, match="brent_price"):
        ingest(settings, end=full["date"].iloc[-1].date(), fetch_text=fake)
    assert not (settings.paths.live_dir / OIL_FILENAME).exists()


def test_source_parsers() -> None:
    days = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
    values = pd.Series([70.0, None, 71.5], index=days)

    brent = yahoo_series(
        "BZ=F", date(2026, 3, 1), date(2026, 3, 4), lambda url: _yahoo_body(values)
    )
    vix = fred_series("VIXCLS", date(2026, 3, 1), date(2026, 3, 4), lambda url: _fred_body(values))

    assert brent.index.tolist() == [days[0], days[2]]  # missing closes dropped
    assert brent.tolist() == [70.0, 71.5]
    assert vix.tolist() == [70.0, 71.5]


@pytest.mark.parametrize("body", ["not json", json.dumps({"chart": {"result": []}})])
def test_unexpected_yahoo_responses_raise(body: str) -> None:
    with pytest.raises(IngestionError, match="Unexpected Yahoo response"):
        yahoo_series("BZ=F", date(2026, 1, 1), date(2026, 1, 5), lambda url: body)


def test_unexpected_fred_responses_raise() -> None:
    with pytest.raises(IngestionError, match="Unexpected FRED response"):
        fred_series("VIXCLS", date(2026, 1, 1), date(2026, 1, 5), lambda url: "only-one-column\n")


def test_http_get_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("blocked")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)

    with pytest.raises(IngestionError, match="Could not fetch"):
        http_get("https://example.invalid/data")


def test_http_get_returns_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())

    assert http_get("https://example.invalid/data") == "ok"


def test_live_sources_must_include_brent(settings: Settings) -> None:
    from pydantic import ValidationError

    from brent_forecast.config import LiveSettings

    with pytest.raises(ValidationError, match="brent_price"):
        LiveSettings(
            lookback_days=60,
            timeout_seconds=5,
            sources={"vix": {"provider": "fred", "symbol": "VIXCLS"}},
        )


def test_default_paths(settings: Settings, tmp_path: Path) -> None:
    assert settings.paths.live_dir == tmp_path / "data" / "live"
    assert settings.live.sources["brent_price"].symbol == "BZ=F"
