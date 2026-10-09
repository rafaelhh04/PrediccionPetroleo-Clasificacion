"""Tests for brent_forecast.data.stages (validate / featurize) and `train --features`."""

import json
from pathlib import Path

import pandas as pd
import pytest

from brent_forecast.config import Settings
from brent_forecast.data.download import save_checksums, sha256sum
from brent_forecast.data.stages import META_COLUMNS, featurize, read_features, validate_raw
from brent_forecast.pipeline import prepare_data
from conftest import EVENTS_FILENAME, EXPECTED_FEATURES, OIL_FILENAME


def test_validate_writes_a_summary(settings: Settings, raw_data_dir: Path) -> None:
    summary = validate_raw(settings)

    written = json.loads(settings.paths.validation_file.read_text())
    assert written == summary
    assert summary["rows"] > 4000
    assert summary["start"] < summary["end"]
    assert set(summary["sha256"]) == {OIL_FILENAME, EVENTS_FILENAME}
    assert summary["matches_recorded_checksums"] is None  # nothing recorded


def test_validate_reports_matching_checksums(settings: Settings, raw_data_dir: Path) -> None:
    digests = {name: sha256sum(raw_data_dir / name) for name in (OIL_FILENAME, EVENTS_FILENAME)}
    save_checksums(settings.paths.checksums_file, digests, settings.data)

    assert validate_raw(settings)["matches_recorded_checksums"] is True


def test_validate_fails_fast_on_missing_files(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        validate_raw(settings)


def test_featurize_round_trip_matches_the_in_memory_dataset(
    settings: Settings, raw_data_dir: Path
) -> None:
    path = featurize(settings)

    from_disk = read_features(path)
    in_memory, _ = prepare_data(settings)

    assert path == settings.paths.dataset_file
    assert from_disk.feature_cols == EXPECTED_FEATURES
    pd.testing.assert_frame_equal(from_disk.frame, in_memory.frame)


def test_prepare_data_from_features_gives_the_same_split(
    settings: Settings, raw_data_dir: Path, tmp_path: Path
) -> None:
    path = featurize(settings, tmp_path / "custom.parquet")

    _, from_file = prepare_data(settings, path)
    _, from_raw = prepare_data(settings)

    pd.testing.assert_frame_equal(from_file.dev, from_raw.dev)
    pd.testing.assert_frame_equal(from_file.test, from_raw.test)


def test_read_features_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "x": [1.0]}).to_parquet(path)

    with pytest.raises(ValueError, match="not a featurized dataset"):
        read_features(path)
    assert META_COLUMNS == ("date", "label", "next_return")
