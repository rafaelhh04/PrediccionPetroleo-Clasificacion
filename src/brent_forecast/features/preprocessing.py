"""Stateless dataset preparation for next-day Brent direction classification.

Execution order (it matters):

1. ``create_label``       -> label created BEFORE any filtering
2. ``engineer_features``  -> past-only transformations and feature list
3. ``handle_nulls``       -> drop warm-up rows with null features
4. ``split_by_date``      -> development / out-of-sample split

Nothing here learns from the data. The fitted steps (VIF selection,
winsorisation, scaling) live in the model pipelines
(:mod:`brent_forecast.models.registry`), so they are fitted on training rows
only, inside every split and cross-validation fold.

After ``load_oil_data`` the ``event_type`` / ``event_severity`` /
``event_description`` columns are the GEOPOLITICAL ones; the original oil
dataset columns live in ``oil_event_*``.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from brent_forecast.config import SplitSettings
from brent_forecast.features import technical

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Target variable
# ──────────────────────────────────────────────


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target: ``label = 1`` if ``brent_return(t+1) > 0``, else 0.

    ``shift(-1)`` aligns every row with the next day's return. It must run
    BEFORE any filtering or reordering so that features and label stay
    aligned. Rows whose next-day return is unknown (the last row, or a day
    followed by a missing return) get no label and are dropped.

    Parameters
    ----------
    df
        Merged dataset with ``date`` and ``brent_return``.

    Returns
    -------
    pandas.DataFrame
        Dataset sorted by date with a float ``label`` column (0.0 or 1.0).

    Raises
    ------
    ValueError
        If a label does not match the sign of the next day's return.
    """
    df = df.sort_values("date").reset_index(drop=True)
    next_return = df["brent_return"].shift(-1)
    # ``NaN > 0`` is False, so mask unknown futures explicitly instead of labelling them 0.
    df["label"] = (next_return > 0).astype(float).where(next_return.notna())
    _check_label_alignment(df["label"], next_return)

    n_before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    logger.info(
        "Labelled %d rows (%d without a known next-day return dropped). Class distribution:\n%s",
        len(df),
        n_before - len(df),
        df["label"].value_counts(normalize=True).round(3),
    )
    return df


def _check_label_alignment(label: pd.Series, next_return: pd.Series) -> None:
    """Verify that every defined label equals the sign of the next day's return."""
    known = next_return.notna()
    expected = (next_return[known] > 0).astype(float)
    mismatched = label[known] != expected
    if mismatched.any():
        row = mismatched.index[np.flatnonzero(mismatched.to_numpy())[0]]
        raise ValueError(
            f"Misaligned label at row {row}: next return={next_return.loc[row]:.4f}, "
            f"label={label.loc[row]}"
        )
    if label[~known].notna().any():
        raise ValueError("Rows without a next-day return must not be labelled.")


# ──────────────────────────────────────────────
# 2. Feature engineering
# ──────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the feature transformations and select the feature columns.

    - P1: drop ``wti_return`` (same-day leakage with ``brent_return``).
    - P2: turn price lags into log returns (stationarity).
    - P3: replace ``gpr_index`` (weekly constant) with ``gpr_change`` (21-day change).
    - P4: replace sparse event columns with dense ``event_flag_binary`` and
      ``high_severity_flag``.
    - P5: drop redundant WTI volatilities; add ``vol_ratio``.
    - Technical indicators (v2): RSI(14), MACD(12, 26, 9) line and histogram,
      Bollinger(20, 2) %B and bandwidth, 21- and 63-day momentum.
    - Market context (v2): 1- and 5-day log changes of the VIX and the dollar index.
    - Seasonality (v2): sine/cosine of the weekday and the month.

    Parameters
    ----------
    df
        Labelled dataset.

    Returns
    -------
    df : pandas.DataFrame
        Dataset with the engineered columns.
    feature_cols : list of str
        Names of the columns used as features.
    """
    # P1 — drop wti_return (same-day leakage)
    df = df.drop(columns=["wti_return"], errors="ignore")

    # P2 — price lags -> log returns
    for n in [1, 3, 7]:
        df[f"lag_ret_{n}"] = np.log(df["brent_price"] / df[f"brent_lag_{n}"])

    lag_cols = [f"brent_lag_{n}" for n in [1, 3, 7]] + [f"wti_lag_{n}" for n in [1, 3, 7]]
    df = df.drop(columns=lag_cols, errors="ignore")

    # P3 — gpr_index -> gpr_change over 21 trading days (~1 month)
    df["gpr_change"] = df["gpr_index"] - df["gpr_index"].shift(21)
    df = df.drop(columns=["gpr_index"], errors="ignore")

    # P4 — dense binary flags from the geopolitical event_severity
    df["event_flag_binary"] = (df["event_severity"] > 0).astype(int)
    df["high_severity_flag"] = (df["event_severity"] >= 7).astype(int)

    # P5 — drop WTI volatilities (corr > 0.95 with Brent)
    df = df.drop(columns=["wti_volatility_7d", "wti_volatility_30d"], errors="ignore")
    df["vol_ratio"] = df["brent_volatility_7d"] / (df["brent_volatility_30d"] + 1e-10)

    # Technical indicators on the Brent price (see features.technical)
    price = df["brent_price"]
    df["rsi_14"] = technical.rsi(price, 14)
    df["macd"], df["macd_hist"] = technical.macd(price, 12, 26, 9)
    df["bb_pct_b"], df["bb_width"] = technical.bollinger(price, 20, 2.0)
    for n in (21, 63):
        df[f"mom_{n}"] = technical.log_return(price, n)

    # Market context: changes of the VIX and the dollar index (gaps forward-filled, past-only)
    for col, name in (("vix", "vix_chg"), ("dxy_index", "dxy_ret")):
        level = df[col].ffill()
        for n in (1, 5):
            df[f"{name}_{n}"] = technical.log_return(level, n)

    # Seasonality, cyclically encoded (Friday is next to Monday, December to January)
    df["dow_sin"], df["dow_cos"] = technical.cyclical(df["date"].dt.dayofweek, 5)
    df["month_sin"], df["month_cos"] = technical.cyclical(df["date"].dt.month - 1, 12)

    # Columns EXCLUDED from the feature set:
    # - date / label: metadata
    # - brent_return: source of the label (leakage)
    # - brent_price / wti_price: non-stationary price levels; the predictive
    #   signal is kept through lag_ret_n (stationary log returns)
    # - event_*: textual geopolitical columns, captured by the binary flags
    # - oil_event_*: leftovers of the rename in data.load
    # - event_flag: implied by event_flag_binary; avoids duplicated signal
    # - next_return: tomorrow's realised return, kept only for the backtest (leakage)
    exclude = {
        "date",
        "label",
        "next_return",
        "brent_return",
        "brent_price",
        "wti_price",
        "event_type",
        "event_description",
        "event_severity",
        "oil_event_type",
        "oil_event_description",
        "oil_event_severity",
        "event_flag",
    }

    feature_cols = [str(c) for c in df.columns if c not in exclude and df[c].dtype != object]

    logger.info("Selected %d features: %s", len(feature_cols), feature_cols)
    return df, feature_cols


# ──────────────────────────────────────────────
# 3. Null handling
# ──────────────────────────────────────────────


def handle_nulls(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Forward-fill VIX/DXY and drop rows with null features.

    Lags and rolling windows create nulls at the start; the longest warm-up is
    ``mom_63`` (63 rows), so about 65 rows out of ~4,000 are dropped (< 2%).
    """
    for col in ["vix", "dxy_index"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    n_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    logger.info(
        "Rows dropped because of null features: %d (%.1f%%)", n_dropped, n_dropped / n_before * 100
    )
    return df


# ──────────────────────────────────────────────
# 4. Dataset assembly and development / out-of-sample split
# ──────────────────────────────────────────────


class Dataset(NamedTuple):
    """Model-ready frame: one row per labelled trading day."""

    frame: pd.DataFrame
    """``date``, ``label``, ``next_return`` and every engineered feature.

    ``next_return`` is the simple return from day ``t`` to ``t+1`` (the
    outcome the label encodes); it is never a feature and only feeds the
    economic backtest.
    """
    feature_cols: list[str]
    """Candidate feature columns, before the fitted VIF selection."""

    @property
    def features(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return self.frame[self.feature_cols]

    @property
    def target(self) -> pd.Series:
        """Return the binary target."""
        return self.frame["label"]

    @property
    def dates(self) -> pd.Series:
        """Return the trading day of every row."""
        return self.frame["date"]


def build_dataset(df: pd.DataFrame) -> Dataset:
    """Label, engineer features and drop warm-up rows: every stateless step.

    The fitted steps (VIF selection, winsorisation, scaling) are part of each
    model pipeline instead, see :mod:`brent_forecast.models.registry`.
    """
    logger.info("Building the dataset")
    df = df.sort_values("date").reset_index(drop=True)
    df["next_return"] = df["brent_price"].shift(-1) / df["brent_price"] - 1
    df = create_label(df)
    df, feature_cols = engineer_features(df)
    df = handle_nulls(df, feature_cols)
    frame = df[["date", "label", "next_return", *feature_cols]].reset_index(drop=True)
    return Dataset(frame=frame, feature_cols=feature_cols)


class Split(NamedTuple):
    """Chronological partition of a :class:`Dataset` frame."""

    dev: pd.DataFrame
    """Development period (``date < test_start``): hyperparameter tuning."""
    test: pd.DataFrame
    """Out-of-sample period (``date >= test_start``): walk-forward evaluation."""


def split_by_date(frame: pd.DataFrame, split: SplitSettings) -> Split:
    """Split the dataset at ``split.test_start`` (rows stay in chronological order)."""
    test_start = pd.Timestamp(split.test_start)
    is_test = frame["date"] >= test_start
    parts = Split(dev=frame[~is_test], test=frame[is_test])
    logger.info(
        "Development rows: %d (%s -> %s) | out-of-sample rows: %d (%s -> %s)",
        len(parts.dev),
        parts.dev["date"].min().date(),
        parts.dev["date"].max().date(),
        len(parts.test),
        parts.test["date"].min().date(),
        parts.test["date"].max().date(),
    )
    logger.info(
        "Development class balance | class 1: %.2f%% | class 0: %.2f%%",
        parts.dev["label"].mean() * 100,
        (1 - parts.dev["label"].mean()) * 100,
    )
    return parts
