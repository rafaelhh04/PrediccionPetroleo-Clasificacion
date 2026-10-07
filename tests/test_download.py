"""Tests for the Kaggle download logic, using a fake downloader (no network)."""

import json
from pathlib import Path

import pytest

from brent_forecast.config import DataSettings
from brent_forecast.data.download import (
    ChecksumMismatchError,
    DataDownloadError,
    download_dataset,
    sha256sum,
    verify_checksums,
)

KAGGLE_DATASET = "owner/dataset"
OIL_FILENAME = "oil.csv"
EVENTS_FILENAME = "events.csv"
DATA = DataSettings(
    kaggle_dataset=KAGGLE_DATASET, oil_filename=OIL_FILENAME, events_filename=EVENTS_FILENAME
)

OIL_CSV = (
    "date,brent_price,wti_price,brent_return,event_type,event_description,event_severity\n"
    "2010-02-17,75.0,72.0,0.1,none,none,0\n"
)
EVENTS_CSV = "date,event_type,event_description,event_severity\n2011-02-15,war,Libya,8\n"


@pytest.fixture
def kaggle_cache(tmp_path: Path) -> Path:
    """Simulate the kagglehub cache with upstream (non-canonical) file names."""
    cache = tmp_path / "cache" / "versions" / "1"
    cache.mkdir(parents=True)
    (cache / "upstream_prices.csv").write_text(OIL_CSV)
    (cache / "events_upstream.csv").write_text(EVENTS_CSV)
    (cache / "README.txt").write_text("not a csv")
    return cache


def _fake_downloader(cache: Path):
    calls: list[tuple[str, bool]] = []

    def downloader(handle: str, *, force_download: bool = False) -> str:
        calls.append((handle, force_download))
        return str(cache)

    downloader.calls = calls  # type: ignore[attr-defined]
    return downloader


def test_download_copies_canonical_files_and_records_checksums(
    tmp_path: Path, kaggle_cache: Path
) -> None:
    data_dir = tmp_path / "data" / "raw"
    checksums = tmp_path / "checksums.json"
    downloader = _fake_downloader(kaggle_cache)

    files = download_dataset(data_dir, checksums, DATA, force=True, downloader=downloader)

    assert downloader.calls == [(KAGGLE_DATASET, True)]  # type: ignore[attr-defined]
    assert set(files) == {OIL_FILENAME, EVENTS_FILENAME}
    assert (data_dir / OIL_FILENAME).read_text() == OIL_CSV
    assert (data_dir / EVENTS_FILENAME).read_text() == EVENTS_CSV
    recorded = json.loads(checksums.read_text())["files"]
    assert recorded[OIL_FILENAME] == sha256sum(data_dir / OIL_FILENAME)
    assert recorded[EVENTS_FILENAME] == sha256sum(data_dir / EVENTS_FILENAME)


def test_download_is_idempotent_with_recorded_checksums(tmp_path: Path, kaggle_cache: Path) -> None:
    data_dir = tmp_path / "raw"
    checksums = tmp_path / "checksums.json"
    download_dataset(data_dir, checksums, DATA, downloader=_fake_downloader(kaggle_cache))
    before = checksums.read_text()

    download_dataset(data_dir, checksums, DATA, downloader=_fake_downloader(kaggle_cache))

    assert checksums.read_text() == before


def test_download_rejects_tampered_data(tmp_path: Path, kaggle_cache: Path) -> None:
    data_dir = tmp_path / "raw"
    checksums = tmp_path / "checksums.json"
    download_dataset(data_dir, checksums, DATA, downloader=_fake_downloader(kaggle_cache))
    (kaggle_cache / "upstream_prices.csv").write_text(OIL_CSV + "2010-02-18,1,1,1,none,none,0\n")

    with pytest.raises(ChecksumMismatchError, match=OIL_FILENAME):
        download_dataset(data_dir, checksums, DATA, downloader=_fake_downloader(kaggle_cache))
    assert not (data_dir / OIL_FILENAME).exists()


def test_update_checksums_accepts_new_data(tmp_path: Path, kaggle_cache: Path) -> None:
    data_dir = tmp_path / "raw"
    checksums = tmp_path / "checksums.json"
    download_dataset(data_dir, checksums, DATA, downloader=_fake_downloader(kaggle_cache))
    (kaggle_cache / "upstream_prices.csv").write_text(OIL_CSV + "2010-02-18,1,1,1,none,none,0\n")

    download_dataset(
        data_dir, checksums, DATA, update_checksums=True, downloader=_fake_downloader(kaggle_cache)
    )

    recorded = json.loads(checksums.read_text())["files"]
    assert recorded[OIL_FILENAME] == sha256sum(data_dir / OIL_FILENAME)


def test_download_fails_when_a_dataset_is_missing(tmp_path: Path, kaggle_cache: Path) -> None:
    (kaggle_cache / "events_upstream.csv").unlink()

    with pytest.raises(DataDownloadError, match=EVENTS_FILENAME):
        download_dataset(
            tmp_path / "raw", tmp_path / "c.json", DATA, downloader=_fake_downloader(kaggle_cache)
        )


def test_download_fails_on_ambiguous_sources(tmp_path: Path, kaggle_cache: Path) -> None:
    (kaggle_cache / "events_copy.csv").write_text(EVENTS_CSV)

    with pytest.raises(DataDownloadError, match="Ambiguous"):
        download_dataset(
            tmp_path / "raw", tmp_path / "c.json", DATA, downloader=_fake_downloader(kaggle_cache)
        )


def test_download_wraps_downloader_errors(tmp_path: Path) -> None:
    def failing(handle: str, *, force_download: bool = False) -> str:
        raise ConnectionError("403 Forbidden")

    with pytest.raises(DataDownloadError, match="manual download"):
        download_dataset(tmp_path / "raw", tmp_path / "c.json", DATA, downloader=failing)


def test_verify_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_checksums(tmp_path, tmp_path / "c.json", DATA)
