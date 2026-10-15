"""Tests for the synthetic dataset used by the demo (`brent data synthetic`)."""

from pathlib import Path

import pytest

from brent_forecast.config import Settings
from brent_forecast.data.load import DateBounds, load_oil_data
from brent_forecast.synthetic import write_synthetic_dataset


def test_the_synthetic_dataset_passes_the_data_contracts(
    settings: Settings, tmp_path: Path
) -> None:
    oil_path, events_path = write_synthetic_dataset(tmp_path / "raw", settings.data)

    d = settings.data
    bounds = DateBounds(d.expected_start_min, d.expected_start_max, d.expected_end_min)

    merged = load_oil_data(oil_path, events_path, bounds)  # Pandera-validated, real date bounds

    assert merged["date"].is_monotonic_increasing
    assert merged["date"].min().year == 2010
    assert merged["date"].max().year == 2026
    assert merged["date"].max().date() >= settings.split.test_start


def test_the_synthetic_dataset_is_reproducible(settings: Settings, tmp_path: Path) -> None:
    first = write_synthetic_dataset(tmp_path / "a", settings.data, seed=5)
    second = write_synthetic_dataset(tmp_path / "b", settings.data, seed=5)
    other = write_synthetic_dataset(tmp_path / "c", settings.data, seed=6)

    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]
    assert first[0].read_bytes() != other[0].read_bytes()


def test_existing_files_are_kept_unless_overwrite(settings: Settings, tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    events = data_dir / settings.data.events_filename
    events.write_text("keep me")

    with pytest.raises(FileExistsError, match=settings.data.events_filename):
        write_synthetic_dataset(data_dir, settings.data)
    assert events.read_text() == "keep me"
    assert not (data_dir / settings.data.oil_filename).exists()

    write_synthetic_dataset(data_dir, settings.data, overwrite=True)
    assert events.read_text().startswith("date,event_type")
