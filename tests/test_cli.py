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
    for command in ("train", "report", "explain", "data", "config"):
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
    tracking = fast_settings.tracking.model_copy(update={"enabled": True})
    settings = fast_settings.model_copy(update={"tracking": tracking})
    config = _write_config(settings, tmp_path / "fast.yaml")

    result = invoke(["--config", str(config), "--log-level", "WARNING", "train"])

    assert result.exit_code == 0, result.output
    assert "MLflow run" in result.output
    assert fast_settings.paths.metrics_file.is_file()
    assert fast_settings.paths.log_file.is_file()
    assert f"Log written to {fast_settings.paths.log_file}" in result.output

    report = invoke(["--config", str(config), "--log-level", "WARNING", "report"])

    assert report.exit_code == 0, report.output
    assert f"Report written to {fast_settings.paths.report_file}" in report.output
    assert "champion = " in report.output

    registry = invoke(["--config", str(config), "registry", "show"])

    assert registry.exit_code == 0
    assert set(json.loads(registry.output)) == {"champion", "challenger"}

    prediction = invoke(["--config", str(config), "predict", "--snapshot"])

    assert prediction.exit_code == 0, prediction.output
    assert json.loads(prediction.output)["alias"] == "champion"

    explain = invoke(["--config", str(config), "--log-level", "WARNING", "explain"])

    assert explain.exit_code == 0, explain.output
    assert "Explanations written to" in explain.output


def test_explain_without_a_training_run_fails(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "explain"])

    assert result.exit_code == 1
    assert "run `brent train`" in result.output


def test_registry_show_before_any_promotion(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "registry", "show"])

    assert result.exit_code == 0
    assert "No model registered" in result.output


def test_registry_show_lists_aliases(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")
    aliases = {"champion": {"version": "3", "candidate": "persistence"}}
    monkeypatch.setattr("brent_forecast.tracking.registry_aliases", lambda s: aliases)

    result = invoke(["--config", str(config), "registry", "show"])

    assert result.exit_code == 0
    assert json.loads(result.output) == aliases


def test_report_warns_when_the_training_run_was_not_tracked(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking = settings.tracking.model_copy(update={"enabled": True})
    config = _write_config(settings.model_copy(update={"tracking": tracking}), tmp_path / "c.yaml")
    report = settings.paths.report_file
    monkeypatch.setattr("brent_forecast.evaluation.report.generate_report", lambda s: report)

    result = invoke(["--config", str(config), "report"])

    assert result.exit_code == 0
    assert "tracking enabled" in result.output


def test_report_without_a_training_run_fails(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "report"])

    assert result.exit_code == 1
    assert "run `brent train` first" in result.output


# ── DVC stages ────────────────────────────────────────────────


def test_data_validate_and_featurize(
    config_file: Path, settings: Settings, raw_data_dir: Path
) -> None:
    validate = invoke(["--config", str(config_file), "data", "validate"])
    featurize = invoke(["--config", str(config_file), "featurize"])

    assert validate.exit_code == 0, validate.output
    assert "Valid:" in validate.output
    assert settings.paths.validation_file.is_file()
    assert featurize.exit_code == 0, featurize.output
    assert settings.paths.dataset_file.is_file()


def test_data_validate_and_featurize_fail_without_data(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    assert invoke(["--config", str(config), "data", "validate"]).exit_code == 1
    assert invoke(["--config", str(config), "featurize"]).exit_code == 1


def test_train_rejects_a_missing_features_file(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "train", "--features", str(tmp_path / "x.parquet")])

    assert result.exit_code == 1
    assert "brent featurize" in result.output


# ── live data and prediction ──────────────────────────────────


def test_predict_without_a_registered_model_fails(settings: Settings, tmp_path: Path) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "predict"])

    assert result.exit_code == 1
    assert "No model with alias 'champion'" in result.output


def test_predict_reports_invalid_data(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(settings, tmp_path / "config.yaml")
    monkeypatch.setattr("brent_forecast.tracking.load_model", lambda s, alias: object())

    result = invoke(["--config", str(config), "predict"])  # no data files at all

    assert result.exit_code == 1
    assert "Prediction failed" in result.output


def test_predict_prints_and_saves_the_prediction(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_data_dir: Path
) -> None:
    from brent_forecast.models.baselines import PersistenceClassifier
    from brent_forecast.pipeline import prepare_data

    dataset, _ = prepare_data(settings)
    model = PersistenceClassifier().fit(dataset.features, dataset.target)
    config = _write_config(settings, tmp_path / "config.yaml")
    monkeypatch.setattr("brent_forecast.tracking.load_model", lambda s, alias: model)
    monkeypatch.setattr(
        "brent_forecast.tracking.registry_aliases",
        lambda s: {"challenger": {"version": "7", "candidate": "persistence"}},
    )

    result = invoke(
        ["--config", str(config), "-l", "WARNING", "predict", "--alias", "challenger", "--snapshot"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["version"] == "7"
    assert payload["candidate"] == "persistence"
    assert json.loads(settings.paths.prediction_file.read_text()) == payload


def test_data_ingest(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import EVENTS_FILENAME, OIL_FILENAME, make_events_frame, make_oil_frame, write_csv
    from test_ingest import HOLD_OUT, FakeMarket

    full = make_oil_frame()
    settings.paths.data_dir.mkdir(parents=True)
    write_csv(full.iloc[:-HOLD_OUT], settings.paths.data_dir / OIL_FILENAME)
    write_csv(make_events_frame(full), settings.paths.data_dir / EVENTS_FILENAME)
    fake = FakeMarket(full)
    monkeypatch.setattr("brent_forecast.data.ingest.http_get", lambda url, timeout: fake(url))
    config = _write_config(settings, tmp_path / "config.yaml")
    last = full["date"].iloc[-1].date().isoformat()

    result = invoke(["--config", str(config), "data", "ingest", "--end", last])

    assert result.exit_code == 0, result.output
    assert f"{HOLD_OUT} new day(s); data up to {last}" in result.output


def test_data_ingest_reports_failures(
    settings: Settings, tmp_path: Path, raw_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brent_forecast.data.ingest import IngestionError

    def blocked(url: str, timeout: float) -> str:
        raise IngestionError("network blocked")

    monkeypatch.setattr("brent_forecast.data.ingest.http_get", blocked)
    config = _write_config(settings, tmp_path / "config.yaml")

    result = invoke(["--config", str(config), "data", "ingest", "--end", "2026-04-30"])

    assert result.exit_code == 1
    assert "Ingestion failed: network blocked" in result.output
