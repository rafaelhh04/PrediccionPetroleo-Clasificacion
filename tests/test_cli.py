"""Tests for the ``brent`` command-line interface (typer.testing.CliRunner)."""

import json
import re
import runpy
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml
from typer.testing import CliRunner

from brent_forecast.cli import app
from brent_forecast.config import Settings
from brent_forecast.data.download import save_checksums, sha256sum
from conftest import CONFIG_PATH, EVENTS_FILENAME, OIL_FILENAME

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI styling (rich colours output when FORCE_COLOR is set, e.g. in CI)."""
    return ANSI.sub("", text)


class Result(NamedTuple):
    exit_code: int
    output: str


def invoke(args: list[str]) -> Result:
    """Run the CLI and return its exit code and colour-free output."""
    result = runner.invoke(app, args)
    return Result(result.exit_code, plain(result.output))


def _write_config(settings: Settings, path: Path) -> Path:
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False))
    return path


@pytest.fixture
def config_file(settings: Settings, tmp_path: Path) -> Path:
    """YAML equal to configs/default.yaml but with every path under tmp_path."""
    return _write_config(settings, tmp_path / "config.yaml")


# ── global options ────────────────────────────────────────────


def test_help_lists_commands() -> None:
    result = invoke(["--help"])

    assert result.exit_code == 0
    for command in ("train", "report", "data", "config"):
        assert command in result.output


def test_no_arguments_shows_help() -> None:
    result = invoke([])

    assert "Usage" in result.output


def test_invalid_log_level_is_rejected() -> None:
    result = invoke(["--log-level", "verbose", "config", "show"])

    assert result.exit_code == 2
    assert "choose from DEBUG" in result.output


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["brent", "--help"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("brent_forecast", run_name="__main__")

    assert exc.value.code == 0
    assert "Usage: brent" in plain(capsys.readouterr().out)


# ── config show ───────────────────────────────────────────────


def test_config_show_prints_resolved_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_SEED", "7")

    result = invoke(["--config", str(CONFIG_PATH), "config", "show"])

    assert result.exit_code == 0
    config = json.loads(result.output)
    assert config["seed"] == 7
    assert config["split"] == {"test_start": "2024-01-01"}
    assert config["validation"]["purge"] == 1


def test_missing_config_file(tmp_path: Path) -> None:
    result = invoke(["--config", str(tmp_path / "nope.yaml"), "config", "show"])

    assert result.exit_code == 2
    assert "Invalid configuration" in result.output


def test_invalid_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_VALIDATION__TUNING_SPLITS", "1")

    result = invoke(["--config", str(CONFIG_PATH), "config", "show"])

    assert result.exit_code == 2
    assert "tuning_splits" in result.output


# ── data verify ───────────────────────────────────────────────


def test_data_verify_records_then_accepts_checksums(
    config_file: Path, settings: Settings, raw_data_dir: Path
) -> None:
    first = invoke(["--config", str(config_file), "data", "verify"])
    second = invoke(["--config", str(config_file), "data", "verify"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "All data files verified." in second.output
    recorded = json.loads(settings.paths.checksums_file.read_text())["files"]
    assert recorded[OIL_FILENAME] == sha256sum(raw_data_dir / OIL_FILENAME)


def test_data_verify_detects_mismatch(
    config_file: Path, settings: Settings, raw_data_dir: Path
) -> None:
    save_checksums(
        settings.paths.checksums_file,
        {OIL_FILENAME: "0" * 64, EVENTS_FILENAME: sha256sum(raw_data_dir / EVENTS_FILENAME)},
        settings.data,
    )

    result = invoke(["--config", str(config_file), "data", "verify"])

    assert result.exit_code == 1
    assert "Checksum mismatch" in result.output


def test_data_verify_reports_missing_files(config_file: Path) -> None:
    result = invoke(["--config", str(config_file), "data", "verify"])

    assert result.exit_code == 1
    assert "Missing data file" in result.output


# ── data download (kagglehub is faked; no network) ────────────


@pytest.fixture
def fake_kagglehub(monkeypatch: pytest.MonkeyPatch, raw_data_dir: Path, tmp_path: Path) -> Path:
    """Serve the synthetic CSVs from a fake kagglehub cache with upstream names."""
    import kagglehub

    cache = tmp_path / "kaggle_cache"
    cache.mkdir()
    (cache / "prices.csv").write_bytes((raw_data_dir / OIL_FILENAME).read_bytes())
    (cache / "events.csv").write_bytes((raw_data_dir / EVENTS_FILENAME).read_bytes())
    for f in raw_data_dir.iterdir():
        f.unlink()

    def dataset_download(handle: str, *, force_download: bool = False) -> str:
        return str(cache)

    monkeypatch.setattr(kagglehub, "dataset_download", dataset_download)
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    return cache


def test_data_download(config_file: Path, settings: Settings, fake_kagglehub: Path) -> None:
    result = invoke(["--config", str(config_file), "data", "download", "--force"])

    assert result.exit_code == 0, result.output
    assert (settings.paths.data_dir / OIL_FILENAME).is_file()
    assert (settings.paths.data_dir / EVENTS_FILENAME).is_file()
    assert result.output.count("Ready:") == 2


def test_data_download_failure_exits_with_error(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kagglehub

    def broken(handle: str, *, force_download: bool = False) -> str:
        raise ConnectionError("403 Forbidden")

    monkeypatch.setattr(kagglehub, "dataset_download", broken)

    result = invoke(["--config", str(config_file), "data", "download"])

    assert result.exit_code == 1
    assert "manual download" in result.output


# ── train ─────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_train_end_to_end(fast_settings: Settings, raw_data_dir: Path, tmp_path: Path) -> None:
    config = _write_config(fast_settings, tmp_path / "fast.yaml")

    result = invoke(["--config", str(config), "--log-level", "WARNING", "train"])

    assert result.exit_code == 0, result.output
    assert fast_settings.paths.metrics_file.is_file()
    assert fast_settings.paths.log_file.is_file()
    assert f"Log written to {fast_settings.paths.log_file}" in result.output

    report = invoke(["--config", str(config), "--log-level", "WARNING", "report"])

    assert report.exit_code == 0, report.output
    assert f"Report written to {fast_settings.paths.report_file}" in report.output


def test_report_without_a_training_run_fails(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "report"])

    assert result.exit_code == 1
    assert "run `brent train` first" in result.output
