"""Tests for brent_forecast.features.technical."""

import numpy as np
import pandas as pd
import pytest

from brent_forecast.features.technical import bollinger, cyclical, log_return, macd, rsi


def _prices(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(70 * np.exp(np.cumsum(rng.normal(0, 0.02, n))))


def _wilder_rsi(price: pd.Series, window: int) -> pd.Series:
    """Reference RSI written as Wilder's explicit recursion (seeded with the first value)."""
    change = price.diff().to_numpy()
    gain = loss = np.nan
    out = np.full(len(price), np.nan)
    for t in range(1, len(price)):
        up, down = max(change[t], 0.0), max(-change[t], 0.0)
        if t == 1:
            gain, loss = up, down
        else:
            gain = gain + (up - gain) / window
            loss = loss + (down - loss) / window
        if t >= window:
            out[t] = 100 * gain / (gain + loss)
    return pd.Series(out)


def test_rsi_matches_wilder_recursion() -> None:
    price = _prices()

    np.testing.assert_allclose(rsi(price, 14), _wilder_rsi(price, 14), equal_nan=True)


def test_rsi_bounds_and_extremes() -> None:
    up = pd.Series(np.arange(1.0, 50.0))
    down = up[::-1].reset_index(drop=True)

    values = rsi(_prices(), 14).dropna()

    assert values.between(0, 100).all()
    assert rsi(up, 14).dropna().eq(100).all()
    assert rsi(down, 14).dropna().eq(0).all()
    assert rsi(up, 14).isna().sum() == 14  # warm-up


def test_macd_is_zero_on_a_flat_price_and_scale_free() -> None:
    flat = pd.Series(np.full(100, 60.0))
    price = _prices()

    line, hist = macd(flat)
    line_a, hist_a = macd(price)
    line_b, hist_b = macd(price * 1000)

    assert line.dropna().eq(0).all()
    assert hist.dropna().eq(0).all()
    np.testing.assert_allclose(line_a, line_b, equal_nan=True)
    np.testing.assert_allclose(hist_a, hist_b, equal_nan=True)
    assert line_a.isna().sum() == 25  # slow EMA warm-up
    assert hist_a.isna().sum() == 25 + 8  # plus the signal EMA


def test_macd_line_is_positive_in_an_uptrend() -> None:
    line, _ = macd(pd.Series(np.exp(np.linspace(0, 1, 200))))

    assert (line.dropna() > 0).all()


def test_bollinger_percent_b_and_bandwidth() -> None:
    price = _prices()

    pct_b, width = bollinger(price, 20, 2.0)

    window = price.iloc[-20:]
    lower = window.mean() - 2 * window.std()
    upper = window.mean() + 2 * window.std()
    assert pct_b.iloc[-1] == pytest.approx((price.iloc[-1] - lower) / (upper - lower))
    assert width.iloc[-1] == pytest.approx((upper - lower) / window.mean())
    assert (width.dropna() > 0).all()
    np.testing.assert_allclose(bollinger(price * 7, 20, 2.0)[0], pct_b, equal_nan=True)


def test_log_return() -> None:
    series = pd.Series([1.0, np.e, np.e**3])

    np.testing.assert_allclose(log_return(series, 1), [np.nan, 1.0, 2.0], equal_nan=True)
    np.testing.assert_allclose(log_return(series, 2), [np.nan, np.nan, 3.0], equal_nan=True)


def test_cyclical_encoding_wraps_around() -> None:
    sin, cos = cyclical(pd.Series(np.arange(12)), 12)

    np.testing.assert_allclose(sin**2 + cos**2, 1.0)
    december_to_january = np.hypot(sin[11] - sin[0], cos[11] - cos[0])
    january_to_july = np.hypot(sin[6] - sin[0], cos[6] - cos[0])
    assert december_to_january < january_to_july
