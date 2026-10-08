"""Technical indicators and calendar encodings (stateless and past-only).

Every function maps a price series to a feature whose value on day ``t`` only
uses observations up to ``t``: rolling windows end at ``t`` and exponential
averages are recursive. Indicators are expressed in scale-free units
(ratios, log returns, bounded oscillators) so that they are comparable across
price regimes. The leakage tests check each one by truncation.

References: Wilder (1978) for the RSI; Appel (1979) for the MACD;
Bollinger (2001) for the bands.
"""

import numpy as np
import pandas as pd


def log_return(series: pd.Series, periods: int) -> pd.Series:
    """Log change over ``periods`` rows: ``log(x_t / x_{t-periods})`` (momentum)."""
    return pd.Series(np.log(series / series.shift(periods)), index=series.index)


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index in [0, 100] with Wilder's smoothing (``alpha = 1/window``)."""
    change = price.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    # 100 - 100 / (1 + gain / loss), written so that a window without losses gives 100.
    return 100 * gain / (gain + loss)


def macd(
    price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line and histogram, both divided by the price.

    Returns
    -------
    line : pandas.Series
        ``(EMA_fast - EMA_slow) / price``.
    histogram : pandas.Series
        ``line - EMA_signal(line)``: momentum of the trend.
    """
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = (ema_fast - ema_slow) / price
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, line - signal_line


def bollinger(price: pd.Series, window: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Bollinger %B and bandwidth.

    Returns
    -------
    percent_b : pandas.Series
        Position of the price inside the bands: 0 at the lower band, 1 at the upper.
    bandwidth : pandas.Series
        ``(upper - lower) / middle``: relative volatility.
    """
    middle = price.rolling(window).mean()
    std = price.rolling(window).std()
    upper, lower = middle + k * std, middle - k * std
    return (price - lower) / (upper - lower), (upper - lower) / middle


def cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Sine/cosine encoding, so that the end of a cycle is close to its start."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)
