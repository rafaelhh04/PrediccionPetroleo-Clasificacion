"""Download the raw datasets from Kaggle and verify their integrity.

The Kaggle dataset ``kavyadhyani/global-oil-prices-andgeopolitical-events`` is
fetched with :mod:`kagglehub`, which reads the ``KAGGLE_USERNAME`` and
``KAGGLE_KEY`` environment variables (or ``~/.kaggle/kaggle.json``).

Upstream file names are not relied upon: each CSV is identified by its header
and copied to the data directory under the canonical name expected by
:mod:`brent_forecast.data.load`. Every copied file is then checked against the
SHA-256 digests stored in the checksums file.
"""

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from brent_forecast.data.load import DATA_DIR as _DATA_DIR
from brent_forecast.data.load import EVENTS_FILENAME, OIL_FILENAME

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "kavyadhyani/global-oil-prices-andgeopolitical-events"
CANONICAL_FILENAMES = (OIL_FILENAME, EVENTS_FILENAME)

DATA_DIR = Path(_DATA_DIR)
CHECKSUMS_PATH = Path("configs/data_checksums.json")

# Columns that identify each dataset. The events columns are also present in
# the oil dataset, so the oil signature is checked first.
_OIL_SIGNATURE = frozenset({"date", "brent_price", "wti_price", "brent_return"})
_EVENTS_SIGNATURE = frozenset({"date", "event_type", "event_description", "event_severity"})

Downloader = Callable[..., str]


class DataDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded or mapped to canonical files."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a data file does not match its recorded SHA-256 digest."""


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def has_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are available via env vars or kaggle.json."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").is_file()


def load_checksums(path: Path) -> dict[str, str | None]:
    """Load the expected digests; unknown files map to ``None``."""
    if not path.is_file():
        return dict.fromkeys(CANONICAL_FILENAMES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, str | None] = payload.get("files", {})
    return {name: files.get(name) for name in CANONICAL_FILENAMES}


def save_checksums(path: Path, checksums: dict[str, str | None]) -> None:
    """Write the digests file in a stable, diff-friendly format."""
    payload = {
        "dataset": KAGGLE_DATASET,
        "algorithm": "sha256",
        "files": {name: checksums.get(name) for name in CANONICAL_FILENAMES},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def identify_csv(path: Path) -> str | None:
    """Return the canonical file name matching the CSV header, or None."""
    columns = {str(c).strip() for c in pd.read_csv(path, nrows=0).columns}
    if _OIL_SIGNATURE <= columns:
        return OIL_FILENAME
    if _EVENTS_SIGNATURE <= columns:
        return EVENTS_FILENAME
    return None


def map_to_canonical(source_dir: Path) -> dict[str, Path]:
    """Map every canonical file name to exactly one CSV found under ``source_dir``."""
    found: dict[str, list[Path]] = {name: [] for name in CANONICAL_FILENAMES}
    for csv in sorted(source_dir.rglob("*.csv")):
        name = identify_csv(csv)
        if name is not None:
            found[name].append(csv)

    mapping: dict[str, Path] = {}
    for name, candidates in found.items():
        if not candidates:
            raise DataDownloadError(f"No CSV in {source_dir} matches the schema of {name}.")
        if len(candidates) > 1:
            listed = ", ".join(c.name for c in candidates)
            raise DataDownloadError(f"Ambiguous source for {name}: {listed}.")
        mapping[name] = candidates[0]
    return mapping


def verify_checksums(
    data_dir: Path,
    checksums_path: Path,
    *,
    update: bool = False,
) -> dict[str, str]:
    """Check the canonical files in ``data_dir`` against the recorded digests.

    Digests that are not recorded yet (``null`` in the checksums file) are
    filled in on first use and written back, so they can be committed. With
    ``update=True`` all recorded digests are overwritten.

    Returns
    -------
    dict[str, str]
        Actual digest of each canonical file.

    Raises
    ------
    FileNotFoundError
        If a canonical file is missing from ``data_dir``.
    ChecksumMismatchError
        If a file does not match its recorded digest.
    """
    expected = load_checksums(checksums_path)
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    recorded_new = False

    for name in CANONICAL_FILENAMES:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing data file: {path}")
        digest = sha256sum(path)
        actual[name] = digest

        if update or expected[name] is None:
            if expected[name] != digest:
                logger.warning("Recording SHA-256 for %s: %s", name, digest)
                expected[name] = digest
                recorded_new = True
        elif expected[name] != digest:
            mismatches.append(f"{name}: expected {expected[name]}, got {digest}")
        else:
            logger.info("Checksum OK: %s", name)

    if mismatches:
        raise ChecksumMismatchError("Checksum mismatch:\n  " + "\n  ".join(mismatches))
    if recorded_new:
        save_checksums(checksums_path, expected)
        logger.warning("Checksums written to %s; commit this file.", checksums_path)
    return actual


def download_dataset(
    data_dir: Path = DATA_DIR,
    checksums_path: Path = CHECKSUMS_PATH,
    *,
    force: bool = False,
    update_checksums: bool = False,
    downloader: Downloader | None = None,
) -> dict[str, Path]:
    """Download the Kaggle dataset, copy it under canonical names and verify it.

    Parameters
    ----------
    data_dir
        Destination directory for the canonical CSV files.
    checksums_path
        JSON file with the expected SHA-256 digests.
    force
        Bypass the kagglehub cache and download again.
    update_checksums
        Overwrite the recorded digests with the downloaded ones.
    downloader
        Injected download function (defaults to ``kagglehub.dataset_download``);
        it receives the dataset handle and returns the local directory.

    Returns
    -------
    dict[str, Path]
        Path of each canonical file inside ``data_dir``.
    """
    if downloader is None:
        import kagglehub

        downloader = kagglehub.dataset_download
        if not has_kaggle_credentials():
            logger.warning(
                "No Kaggle credentials found (KAGGLE_USERNAME/KAGGLE_KEY or "
                "~/.kaggle/kaggle.json); trying an anonymous download."
            )

    logger.info("Downloading Kaggle dataset %s", KAGGLE_DATASET)
    try:
        source_dir = Path(downloader(KAGGLE_DATASET, force_download=force))
    except Exception as exc:
        raise DataDownloadError(
            f"Could not download {KAGGLE_DATASET}: {exc}. Check your Kaggle "
            "credentials or follow the manual download steps in the README."
        ) from exc

    mapping = map_to_canonical(source_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    targets: dict[str, Path] = {}
    for name, source in mapping.items():
        target = data_dir / name
        shutil.copy2(source, target)
        logger.info("Copied %s -> %s", source.name, target)
        targets[name] = target

    try:
        verify_checksums(data_dir, checksums_path, update=update_checksums)
    except ChecksumMismatchError:
        for target in targets.values():
            target.unlink(missing_ok=True)
        raise
    return targets
