"""Shared fixtures: synthetic datasets with the Kaggle schema and isolated settings.

No real data is ever used: every frame is generated from a seeded NumPy
generator, so the tests are deterministic and run offline.
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import DEFAULT_CONFIG_PATH, Settings, load_settings
from brent_forecast.data.load import DateBounds

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / DEFAULT_CONFIG_PATH

OIL_FILENAME = "oil_geopolitics_dataset_2010_2026.csv"
EVENTS_FILENAME = "geopolitical_events_timeline.csv"
EVENT_TYPES = ("war", "sanctions", "opec", "conflict", "disaster", "blockade")
DATE_BOUNDS = DateBounds(date(2009, 1, 1), date(2011, 12, 31), date(2025, 1, 1))

EXPECTED_FEATURES = [  # engineer_features output, in order
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
    "rsi_14",
    "macd",
    "macd_hist",
    "bb_pct_b",
    "bb_width",
    "mom_21",
    "mom_63",
    "vix_chg_1",
    "vix_chg_5",
    "dxy_ret_1",
    "dxy_ret_5",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def make_oil_frame(
    seed: int = 0, start: str = "2010-02-17", end: str = "2026-03-12", step: int = 1
) -> pd.DataFrame:
    """Oil dataset with the full Kaggle schema on every ``step``-th business day."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)[::step]
    n = len(dates)
    brent = 75 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    wti = brent * np.exp(rng.normal(-0.04, 0.01, n))
    df = pd.DataFrame({"date": dates})
    df["brent_price"] = brent
    df["wti_price"] = wti
    df["dxy_index"] = 90 + np.cumsum(rng.normal(0, 0.3, n))
    df["vix"] = np.abs(18 + np.cumsum(rng.normal(0, 0.1, n)))
    df.loc[rng.choice(n, 20, replace=False), "vix"] = np.nan
    df["gpr_index"] = np.repeat(rng.normal(100, 20, n // 5 + 1), 5)[:n]
    df["brent_return"] = df["brent_price"].pct_change() * 100
    df["wti_return"] = df["wti_price"].pct_change() * 100
    for k in (1, 3, 7):
        df[f"brent_lag_{k}"] = df["brent_price"].shift(k)
    for k in (1, 3, 7):
        df[f"wti_lag_{k}"] = df["wti_price"].shift(k)
    for name in ("brent", "wti"):
        df[f"{name}_volatility_7d"] = df[f"{name}_return"].rolling(7).std()
        df[f"{name}_volatility_30d"] = df[f"{name}_return"].rolling(30).std()
    df["brent_wti_spread"] = df["brent_price"] - df["wti_price"]
    df["event_flag"] = 0
    df["event_type"] = "none"
    df["event_description"] = "none"
    df["event_severity"] = 0
    flagged = rng.choice(np.arange(40, n), 25, replace=False)
    df.loc[flagged, "event_flag"] = 1
    df.loc[flagged, "event_type"] = rng.choice(EVENT_TYPES, 25)
    df.loc[flagged, "event_description"] = "oil dataset event"
    df.loc[flagged, "event_severity"] = rng.integers(1, 11, 25)
    return df


def make_events_frame(oil: pd.DataFrame, seed: int = 1, n_events: int = 35) -> pd.DataFrame:
    """Geopolitical events timeline on dates that exist in ``oil``."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(40, len(oil)), n_events, replace=False))
    return pd.DataFrame(
        {
            "date": oil["date"].iloc[idx].to_numpy(),
            "event_type": rng.choice(EVENT_TYPES, n_events),
            "event_description": [f"synthetic event {i}" for i in range(n_events)],
            "event_severity": rng.integers(1, 11, n_events),
        }
    )


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` with ISO dates, as the Kaggle files are."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop BRENT_* variables so the developer's environment never leaks into tests."""
    import os

    for key in list(os.environ):
        if key.startswith("BRENT_"):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Undo ``setup_logging`` calls (CLI callback, logging tests) after every test."""
    import logging

    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        if handler not in handlers:
            handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    logging.captureWarnings(False)


@pytest.fixture(scope="session")
def oil_frame() -> pd.DataFrame:
    """Synthetic oil dataset (read-only: copy before mutating)."""
    return make_oil_frame()


@pytest.fixture(scope="session")
def events_frame(oil_frame: pd.DataFrame) -> pd.DataFrame:
    """Synthetic geopolitical events (read-only: copy before mutating)."""
    return make_events_frame(oil_frame)


@pytest.fixture
def raw_data_dir(tmp_path: Path, oil_frame: pd.DataFrame, events_frame: pd.DataFrame) -> Path:
    """Directory with both synthetic CSVs (full daily calendar) under their canonical names."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    write_csv(oil_frame, data_dir / OIL_FILENAME)
    write_csv(events_frame, data_dir / EVENTS_FILENAME)
    return data_dir


@pytest.fixture
def small_raw_data_dir(tmp_path: Path) -> Path:
    """Like ``raw_data_dir`` but with one row per week (~840 rows) for fast end-to-end runs."""
    oil = make_oil_frame(step=5)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    write_csv(oil, data_dir / OIL_FILENAME)
    write_csv(make_events_frame(oil), data_dir / EVENTS_FILENAME)
    return data_dir


@pytest.fixture(scope="session")
def merged_frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Output of ``load_oil_data`` on the synthetic CSVs (read-only)."""
    from brent_forecast.data.load import load_oil_data

    oil = make_oil_frame()
    data_dir = tmp_path_factory.mktemp("merged")
    return load_oil_data(
        write_csv(oil, data_dir / OIL_FILENAME),
        write_csv(make_events_frame(oil), data_dir / EVENTS_FILENAME),
        DATE_BOUNDS,
    )


def _paths(tmp_path: Path) -> dict[str, Any]:
    return {
        "data_dir": tmp_path / "data" / "raw",
        "results_dir": tmp_path / "results",
        "checksums_file": tmp_path / "configs" / "data_checksums.json",
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings from configs/default.yaml with every path under ``tmp_path``."""
    return load_settings(CONFIG_PATH, paths=_paths(tmp_path))


# Settings sources are deep-merged, so every grid key of configs/default.yaml must be
# overridden here; otherwise the default (slow) values of the missing keys survive.
FAST_MODELS: dict[str, Any] = {
    "logistic_regression": {
        "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 200},
        "grid": {"C": [0.1, 1.0]},
    },
    "svm": {
        "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "grid": {"C": [1.0], "gamma": ["scale"]},
    },
    "random_forest": {
        "params": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 20, "n_jobs": 1},
        "grid": {"n_estimators": [10], "max_depth": [3], "min_samples_leaf": [20]},
    },
    "mlp": {
        "params": {
            "hidden_1": 8,
            "hidden_2": 4,
            "dropout_p": 0.2,
            "learning_rate": 0.01,
            "batch_size": 64,
            "max_epochs": 3,
            "patience": 2,
        },
        "grid": {"hidden_pair": [[8, 4]], "learning_rate": [0.01]},
    },
}


@pytest.fixture
def fast_settings(tmp_path: Path) -> Settings:
    """Settings with tiny grids and few epochs, for end-to-end tests."""
    return load_settings(
        CONFIG_PATH,
        paths=_paths(tmp_path),
        validation={
            "purge": 1,
            "embargo": 5,
            "tuning_splits": 2,
            "scoring": "roc_auc",
            "mode": "expanding",
            "test_window": 63,
            "rolling_train_size": 1260,
        },
        evaluation={"learning_curve_train_sizes": [0.5, 1.0], "bootstrap_resamples": 200},
        models=FAST_MODELS,
    )


@pytest.fixture
def toy_xy() -> tuple[np.ndarray, np.ndarray]:
    """Small, weakly separable binary problem (300 x 5)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(float)
    return X, y


@pytest.fixture
def plots_dir(tmp_path: Path) -> Iterator[Path]:
    """Empty plots directory."""
    path = tmp_path / "plots"
    yield path
