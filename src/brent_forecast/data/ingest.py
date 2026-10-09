"""Incremental ingestion of recent market data (``brent data ingest``).

The Kaggle snapshot ends on a fixed date. To predict on current data, the days
after the last available date are fetched from public sources and appended to
a **copy** of the prices file in ``data/live/`` (the versioned raw snapshot in
``data/raw/`` is never modified):

- Yahoo Finance chart API: Brent (``BZ=F``), WTI (``CL=F``), dollar index
  (``DX-Y.NYB``);
- FRED: VIX (``VIXCLS``).

Brent defines the trading days; the other series are aligned to them (missing
values stay missing, as in the original data). The derived columns of the
Kaggle schema (returns in percent, lags, 7/30-day volatilities, spread) are
recomputed for the new rows only, using ``lookback_days`` of history;
:func:`derivation_gap` checks those definitions against the snapshot itself.

There is no live source for the geopolitical risk index or the events: the last
GPR value is carried forward and new days have no event. The result is
validated with the same Pandera contract as the snapshot.

All network access goes through an injectable ``fetch_text(url) -> str``, so the
tests never touch the network.
"""

import io
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from brent_forecast.config import LiveSource, Settings
from brent_forecast.data.load import load_oil_prices
from brent_forecast.data.schemas import OIL_SCHEMA, validate

logger = logging.getLogger(__name__)

FetchText = Callable[[str], str]
"""``url -> response body``; raises :class:`IngestionError` on failure."""

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class IngestionError(RuntimeError):
    """A source could not be reached or returned unusable data."""


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of one ingestion run."""

    path: Path
    new_rows: int
    first_new: date | None
    last_date: date


def http_get(url: str, timeout: float = 20.0) -> str:
    """Download ``url`` as text (the default ``fetch_text``)."""
    request = urllib.request.Request(url, headers={"User-Agent": "brent-forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body: bytes = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise IngestionError(f"Could not fetch {url}: {exc}") from exc
    return body.decode("utf-8")


# ── sources ───────────────────────────────────────────────────


def yahoo_series(symbol: str, start: date, end: date, fetch_text: FetchText) -> pd.Series:
    """Daily closes of ``symbol`` between ``start`` and ``end`` (both inclusive)."""
    period1 = int(datetime.combine(start, time.min, UTC).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=1), time.min, UTC).timestamp())
    url = (
        f"{YAHOO_URL.format(symbol=quote(symbol))}?period1={period1}&period2={period2}&interval=1d"
    )
    try:
        result = json.loads(fetch_text(url))["chart"]["result"][0]
        stamps = result.get("timestamp") or []
        closes = result["indicators"]["quote"][0]["close"] if stamps else []
        zone = result["meta"]["exchangeTimezoneName"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise IngestionError(f"Unexpected Yahoo response for {symbol}: {exc}") from exc
    days = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(zone).tz_localize(None)
    series = pd.Series(closes, index=days.normalize(), dtype=float, name=symbol)
    return _clip(series.dropna(), start, end)


def fred_series(series_id: str, start: date, end: date, fetch_text: FetchText) -> pd.Series:
    """Daily observations of FRED ``series_id`` (``.`` marks a missing value)."""
    url = f"{FRED_URL}?id={quote(series_id)}&cosd={start.isoformat()}&coed={end.isoformat()}"
    try:
        frame = pd.read_csv(io.StringIO(fetch_text(url)))
        days = pd.to_datetime(frame.iloc[:, 0])
        values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    except (ValueError, IndexError, pd.errors.ParserError) as exc:
        raise IngestionError(f"Unexpected FRED response for {series_id}: {exc}") from exc
    series = pd.Series(values.to_numpy(), index=days, dtype=float, name=series_id)
    return _clip(series.dropna(), start, end)


def _clip(series: pd.Series, start: date, end: date) -> pd.Series:
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def fetch(source: LiveSource, start: date, end: date, fetch_text: FetchText) -> pd.Series:
    """Fetch one configured series."""
    reader = yahoo_series if source.provider == "yahoo" else fred_series
    return reader(source.symbol, start, end, fetch_text)


# ── derived columns of the Kaggle schema ──────────────────────


def derive_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute returns (percent), lags, 7/30-day volatilities and the Brent–WTI spread."""
    out = df.copy()
    for name in ("brent", "wti"):
        price = out[f"{name}_price"]
        out[f"{name}_return"] = price.pct_change(fill_method=None) * 100
        for k in (1, 3, 7):
            out[f"{name}_lag_{k}"] = price.shift(k)
        out[f"{name}_volatility_7d"] = out[f"{name}_return"].rolling(7).std()
        out[f"{name}_volatility_30d"] = out[f"{name}_return"].rolling(30).std()
    out["brent_wti_spread"] = out["brent_price"] - out["wti_price"]
    return out


DERIVED_COLUMNS = ("brent_return", "brent_lag_1", "brent_volatility_7d", "brent_wti_spread")


def derivation_gap(history: pd.DataFrame, tail: int = 250) -> float:
    """Largest absolute difference between the snapshot's derived columns and ours.

    A large gap means the source dataset defines those columns differently, and
    the appended rows would not be comparable to the training data.
    """
    recent = history.tail(tail).reset_index(drop=True)
    derived = derive_market_columns(recent)
    gaps = [
        float(np.nanmax((derived[c] - recent[c]).abs().iloc[31:].to_numpy()))
        for c in DERIVED_COLUMNS
        if c in recent.columns
    ]
    return max(gaps, default=0.0)


# ── ingestion ─────────────────────────────────────────────────


def ingest(
    settings: Settings,
    *,
    end: date | None = None,
    fetch_text: FetchText | None = None,
) -> IngestionResult:
    """Append the days after the last available date to ``data/live/<prices file>``.

    The live file is extended if it exists, otherwise it starts from the raw
    snapshot. Re-running is idempotent: only dates after the last one are added.

    Raises
    ------
    IngestionError
        If a source fails or Brent returns no data for a non-empty range.
    DataValidationError
        If the extended file breaks the prices contract.
    """
    paths, live = settings.paths, settings.live
    fetch_text = fetch_text or (lambda url: http_get(url, live.timeout_seconds))
    target = paths.live_dir / settings.data.oil_filename
    base = load_oil_prices(
        target if target.is_file() else paths.data_dir / settings.data.oil_filename
    )
    base = base.sort_values("date").reset_index(drop=True)
    last = base["date"].max().date()
    end = end or date.today()
    start = last + timedelta(days=1)
    if start > end:
        logger.info("Live data already up to date (%s)", last)
        return IngestionResult(_write(base, target), 0, None, last)

    gap = derivation_gap(base)
    if gap > 1e-6:
        logger.warning(
            "The snapshot's derived columns differ from the recomputed ones by up to %.3g; "
            "appended rows may not be comparable to the training data.",
            gap,
        )

    series = {col: fetch(src, start, end, fetch_text) for col, src in live.sources.items()}
    brent_days = series["brent_price"].index
    if brent_days.empty:
        logger.info("No new Brent trading day between %s and %s", start, end)
        return IngestionResult(_write(base, target), 0, None, last)

    new = pd.DataFrame({"date": brent_days})
    for col, values in series.items():
        new[col] = values.reindex(brent_days).to_numpy()
    new["gpr_index"] = base["gpr_index"].ffill().iloc[-1]  # no live source: carried forward
    no_event: dict[str, str | int] = {
        "event_flag": 0,
        "event_type": "none",
        "event_description": "none",
        "event_severity": 0,
    }
    for name, value in no_event.items():
        if name in base.columns:
            new[name] = value

    history = base.tail(live.lookback_days)
    combined = derive_market_columns(pd.concat([history, new], ignore_index=True))
    appended = combined.tail(len(new))[base.columns.intersection(combined.columns)]
    extended = validate(pd.concat([base, appended], ignore_index=True), OIL_SCHEMA)

    logger.info(
        "Appended %d day(s) (%s -> %s) to %s",
        len(new),
        brent_days.min().date(),
        brent_days.max().date(),
        target,
    )
    return IngestionResult(
        _write(extended, target), len(new), brent_days.min().date(), brent_days.max().date()
    )


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
