"""Synthetic datasets with the Kaggle schema, for demos and tests.

The prices are a geometric random walk, so by construction there is nothing
to predict: a run on this data exercises the whole system (training, the
statistical report, the registry, the API and the dashboard) and must
conclude that no model beats the baselines. Every frame comes from a seeded
NumPy generator, so the files are byte-for-byte reproducible.

The checksums file is never updated: a synthetic run keeps the provenance
``matches_recorded_checksums = None`` and the dashboard labels it synthetic.
This module lives outside ``brent_forecast.data`` on purpose: that package is a
dependency of the DVC stages, and the demo generator is not part of the pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from brent_forecast.config import DataSettings

logger = logging.getLogger(__name__)

EVENT_TYPES = ("war", "sanctions", "opec", "conflict", "disaster", "blockade")


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


def write_synthetic_dataset(
    data_dir: Path, data: DataSettings, seed: int = 0, overwrite: bool = False
) -> list[Path]:
    """Write both synthetic CSVs to ``data_dir`` under their canonical names.

    Parameters
    ----------
    data_dir
        Destination directory (created if missing).
    data
        Dataset settings; only the canonical file names are used.
    seed
        Seed of the prices; the events use ``seed + 1``.
    overwrite
        Replace existing files. Without it, existing data (possibly the real
        dataset) is never touched.

    Returns
    -------
    list of Path
        The oil file and the events file.

    Raises
    ------
    FileExistsError
        If a canonical file already exists and ``overwrite`` is false.
    """
    targets = [data_dir / name for name in data.filenames]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing data: {names}")
    data_dir.mkdir(parents=True, exist_ok=True)
    oil = make_oil_frame(seed=seed)
    events = make_events_frame(oil, seed=seed + 1)
    write_csv(oil, targets[0])
    write_csv(events, targets[1])
    logger.info("Wrote a synthetic dataset (random walk, seed %d) to %s", seed, data_dir)
    return targets
