"""Preprocessing pipeline for next-day Brent direction classification.

Execution order (it matters):

1. ``create_label``            -> label created BEFORE any filtering
2. ``engineer_features``       -> transformations and feature selection
3. ``handle_nulls``            -> drop rows with null features
4. ``filter_features_by_vif``  -> multicollinearity filter (train rows only)
5. ``split_temporal``          -> chronological train/val/test split
6. ``winsorize_features``      -> clip outliers (percentiles fitted on train)
7. ``scale_features``          -> standardisation (fitted on train only)

After ``load_oil_data`` the ``event_type`` / ``event_severity`` /
``event_description`` columns are the GEOPOLITICAL ones; the original oil
dataset columns live in ``oil_event_*``.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from brent_forecast._types import FloatArray
from brent_forecast.config import PreprocessingSettings, SplitSettings

logger = logging.getLogger(__name__)

SplitArrays = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]


# ──────────────────────────────────────────────
# 1. Target variable
# ──────────────────────────────────────────────


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target: ``label = 1`` if ``brent_return(t+1) > 0``, else 0.

    ``shift(-1)`` aligns every row with the next day's return. It must run
    BEFORE any filtering or reordering so that features and label stay
    aligned; the alignment is asserted on the first 100 rows.

    Parameters
    ----------
    df
        Merged dataset with ``date`` and ``brent_return``.

    Returns
    -------
    pandas.DataFrame
        Dataset sorted by date with a float ``label`` column.

    Notes
    -----
    Known issue (kept on purpose in this refactoring phase, as fixing it
    changes the metrics): ``NaN > 0`` evaluates to ``False``, so the last row
    gets ``label = 0`` instead of ``NaN`` and is not dropped by ``dropna``.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["label"] = (df["brent_return"].shift(-1) > 0).astype(float)

    returns = df["brent_return"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=float)
    for i in range(min(100, len(df) - 1)):
        next_return = returns[i + 1]
        label = labels[i]
        assert (next_return > 0) == (label == 1.0), (
            f"Misaligned label at row {i}: next return={next_return:.4f}, label={label}"
        )

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    logger.info("Class distribution:\n%s", df["label"].value_counts(normalize=True).round(3))
    return df


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
    - Seasonality: ``day_of_week`` and ``month``.

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

    # Seasonality
    df["day_of_week"] = df["date"].dt.dayofweek  # 0 = Monday ... 4 = Friday
    df["month"] = df["date"].dt.month  # 1-12

    # Columns EXCLUDED from the feature set:
    # - date / label: metadata
    # - brent_return: source of the label (leakage)
    # - brent_price / wti_price: non-stationary price levels; the predictive
    #   signal is kept through lag_ret_n (stationary log returns)
    # - event_*: textual geopolitical columns, captured by the binary flags
    # - oil_event_*: leftovers of the rename in data.load
    # - event_flag: implied by event_flag_binary; avoids duplicated signal
    exclude = {
        "date",
        "label",
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

    Lags create nulls at the start (``lag_ret_7``: 7 rows) and ``gpr_change``
    creates 21, so at most ~21 rows out of ~4,000 are dropped (< 0.6%).
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
# 4. Multicollinearity filter (VIF)
# ──────────────────────────────────────────────


def filter_features_by_vif(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end_date: pd.Timestamp,
    vif_threshold: float,
) -> list[str]:
    """Drop the highest-VIF feature, repeatedly, until all VIFs are below the threshold.

    The Variance Inflation Factor measures how much the variance of a
    coefficient is inflated by the other features; VIF > 10 means the feature
    is almost a linear combination of the others.

    The VIF is computed ONLY on training rows (``date < train_end_date``) so
    that validation and test data do not influence which features survive.

    Parameters
    ----------
    df
        Dataset with ``date`` and the candidate features.
    feature_cols
        Candidate feature names.
    train_end_date
        First date that is no longer part of the training set.
    vif_threshold
        Features are dropped while the maximum VIF is above this value.

    Returns
    -------
    list of str
        Surviving features, in their original order.
    """
    train_mask = df["date"] < train_end_date
    remaining = list(feature_cols)
    logger.info(
        "VIF filtering | initial features: %d | threshold VIF > %s", len(remaining), vif_threshold
    )

    while len(remaining) > 1:
        X = df.loc[train_mask, remaining].to_numpy().astype(float)

        vifs = []
        for i in range(X.shape[1]):
            try:
                v = float(variance_inflation_factor(X, i))
            except Exception:
                v = float("inf")
            vifs.append(v)

        max_vif = max(vifs)
        if max_vif <= vif_threshold:
            break

        worst_idx = vifs.index(max_vif)
        worst_name = remaining[worst_idx]
        vif_str = "inf" if not np.isfinite(max_vif) else f"{max_vif:.2f}"
        logger.info("VIF dropped: %-25s VIF = %s", worst_name, vif_str)
        remaining.pop(worst_idx)

    # Final report of the surviving features
    X_final = df.loc[train_mask, remaining].to_numpy().astype(float)
    logger.info("VIF final features (%d):", len(remaining))
    for i, name in enumerate(remaining):
        try:
            v = float(variance_inflation_factor(X_final, i))
            v_str = "inf" if not np.isfinite(v) else f"{v:.2f}"
        except Exception:
            v_str = "n/a"
        logger.info("  %-25s VIF = %s", name, v_str)

    return remaining


# ──────────────────────────────────────────────
# 5. Strict chronological split
# ──────────────────────────────────────────────


def split_temporal(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> SplitArrays:
    """Split the dataset in strict chronological order.

    - Train: ``date < train_end`` (2010-2021 by default)
    - Validation: ``train_end <= date < val_end`` (2022-2023 by default)
    - Test: ``date >= val_end`` (2024-2026 by default)

    Returns
    -------
    tuple of numpy.ndarray
        ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    train_mask = df["date"] < train_end
    val_mask = (df["date"] >= train_end) & (df["date"] < val_end)
    test_mask = df["date"] >= val_end

    X_train = df.loc[train_mask, feature_cols].to_numpy()
    X_val = df.loc[val_mask, feature_cols].to_numpy()
    X_test = df.loc[test_mask, feature_cols].to_numpy()

    y_train = df.loc[train_mask, "label"].to_numpy()
    y_val = df.loc[val_mask, "label"].to_numpy()
    y_test = df.loc[test_mask, "label"].to_numpy()

    logger.info(
        "Split sizes | train: %d | val: %d | test: %d",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
    )
    logger.info(
        "Train class balance | class 1: %.2f%% | class 0: %.2f%%",
        y_train.mean() * 100,
        (1 - y_train.mean()) * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# 6. Winsorisation
# ──────────────────────────────────────────────


def winsorize_features(
    X_train: FloatArray,
    X_val: FloatArray,
    X_test: FloatArray,
    feature_cols: list[str],
    lower: float,
    upper: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Clip the return features to the ``[lower, upper]`` percentiles of train.

    Motivation: the COVID crash (March 2020) produced returns > 10 that distort
    LogReg, MLP and SVM. Percentiles are computed ONLY on train to avoid
    leakage. The arrays are modified in place and also returned.
    """
    return_idx = [i for i, c in enumerate(feature_cols) if "ret" in c or "return" in c]

    for idx in return_idx:
        lo = np.percentile(X_train[:, idx], lower * 100)
        hi = np.percentile(X_train[:, idx], upper * 100)
        X_train[:, idx] = np.clip(X_train[:, idx], lo, hi)
        X_val[:, idx] = np.clip(X_val[:, idx], lo, hi)
        X_test[:, idx] = np.clip(X_test[:, idx], lo, hi)

    logger.info("Winsorized return features: %d", len(return_idx))
    return X_train, X_val, X_test


# ──────────────────────────────────────────────
# 7. Scaling
# ──────────────────────────────────────────────


def scale_features(
    X_train: FloatArray,
    X_val: FloatArray,
    X_test: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, StandardScaler]:
    """Standardise features to zero mean and unit variance.

    ``fit`` is called ONLY on train; validation and test only get ``transform``.
    """
    scaler = StandardScaler()
    X_train_sc: FloatArray = scaler.fit_transform(X_train)
    X_val_sc: FloatArray = scaler.transform(X_val)
    X_test_sc: FloatArray = scaler.transform(X_test)

    logger.debug("Scaled train mean (first 3): %s", X_train_sc.mean(axis=0)[:3].round(4))
    logger.debug("Scaled train std (first 3): %s", X_train_sc.std(axis=0)[:3].round(4))
    return X_train_sc, X_val_sc, X_test_sc, scaler


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────


def preprocess(
    df: pd.DataFrame,
    split: SplitSettings,
    preprocessing: PreprocessingSettings,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    StandardScaler,
    list[str],
]:
    """Run the full preprocessing: from the merged DataFrame to model-ready arrays.

    Parameters
    ----------
    df
        Output of ``load_oil_data``.
    split
        Chronological split boundaries.
    preprocessing
        VIF and winsorisation thresholds.

    Returns
    -------
    X_train, X_val, X_test : numpy.ndarray
        Scaled feature matrices.
    y_train, y_val, y_test : numpy.ndarray
        Float labels.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on train.
    feature_cols : list of str
        Feature names, in column order.
    """
    logger.info("Preprocessing pipeline started")

    df = create_label(df)
    df, feature_cols = engineer_features(df)
    df = handle_nulls(df, feature_cols)
    train_end = pd.Timestamp(split.train_end)
    val_end = pd.Timestamp(split.val_end)
    feature_cols = filter_features_by_vif(df, feature_cols, train_end, preprocessing.vif_threshold)

    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(
        df, feature_cols, train_end, val_end
    )
    X_train, X_val, X_test = winsorize_features(
        X_train,
        X_val,
        X_test,
        feature_cols,
        preprocessing.winsor_lower,
        preprocessing.winsor_upper,
    )
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    logger.info(
        "Preprocessing done | X_train %s | X_val %s | X_test %s",
        X_train_sc.shape,
        X_val_sc.shape,
        X_test_sc.shape,
    )

    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler, feature_cols
