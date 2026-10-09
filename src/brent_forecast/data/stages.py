"""Data stages of the reproducible pipeline (``dvc.yaml``): validate and featurize.

``brent data validate`` checks the raw files and writes a small JSON summary
(a DVC metric); ``brent featurize`` writes the model-ready dataset to Parquet so
that ``brent train --features`` trains on exactly the versioned snapshot.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from brent_forecast.config import Settings
from brent_forecast.data.download import load_checksums, sha256sum
from brent_forecast.data.load import DateBounds, load_oil_data
from brent_forecast.data.schemas import FEATURES_SCHEMA, validate
from brent_forecast.features.preprocessing import Dataset, build_dataset

logger = logging.getLogger(__name__)

# Columns of the featurized dataset that are not features.
META_COLUMNS = ("date", "label", "next_return")


def load_raw(settings: Settings) -> pd.DataFrame:
    """Load and validate the two raw CSVs (see :func:`brent_forecast.data.load.load_oil_data`)."""
    paths, data = settings.paths, settings.data
    return load_oil_data(
        paths.data_dir / data.oil_filename,
        paths.data_dir / data.events_filename,
        DateBounds(data.expected_start_min, data.expected_start_max, data.expected_end_min),
    )


def validate_raw(settings: Settings) -> dict[str, Any]:
    """Validate the raw files and write ``validation.json``; return the summary.

    Raises
    ------
    FileNotFoundError, ValueError
        If a file is missing or fails validation (the stage fails fast).
    """
    paths, data = settings.paths, settings.data
    df = load_raw(settings)
    recorded = load_checksums(paths.checksums_file, data)
    actual = {name: sha256sum(paths.data_dir / name) for name in data.filenames}
    summary = {
        "rows": len(df),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "sha256": actual,
        "matches_recorded_checksums": None if None in recorded.values() else recorded == actual,
    }
    paths.validation_file.parent.mkdir(parents=True, exist_ok=True)
    paths.validation_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Raw data valid: %d rows, %s -> %s", summary["rows"], summary["start"], summary["end"]
    )
    return summary


def featurize(settings: Settings, output: Path | None = None) -> Path:
    """Build the dataset from the raw files and write it to Parquet; return its path."""
    output = output or settings.paths.dataset_file
    dataset = build_dataset(load_raw(settings))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(output, index=False)
    logger.info(
        "Wrote %d rows x %d features to %s", len(dataset.frame), len(dataset.feature_cols), output
    )
    return output


def read_features(path: Path) -> Dataset:
    """Load a dataset written by :func:`featurize`.

    Raises
    ------
    ValueError
        If the file lacks the date, label or next-return columns.
    DataValidationError
        If it breaks the feature schema (e.g. a tampered or stale file).
    """
    frame = pd.read_parquet(path)
    missing = [c for c in META_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a featurized dataset: missing {missing}")
    feature_cols = [str(c) for c in frame.columns if c not in META_COLUMNS]
    return Dataset(frame=validate(frame, FEATURES_SCHEMA), feature_cols=feature_cols)
