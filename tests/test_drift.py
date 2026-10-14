"""Tests for brent_forecast.monitoring.drift (Evidently drift reports, no network)."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from brent_forecast.api.service import ServedModel
from brent_forecast.cli import app
from brent_forecast.config import Settings
from brent_forecast.models.baselines import PersistenceClassifier
from brent_forecast.monitoring import drift
from brent_forecast.monitoring.drift import (
    PREDICTION,
    ColumnDrift,
    DriftSummary,
    drift_dir,
    feature_history,
    monitor_drift,
    parse_column_drift,
    run_and_notify,
    split_windows,
)
from brent_forecast.pipeline import prepare_data

pytestmark = pytest.mark.filterwarnings(
    "ignore::RuntimeWarning",  # Evidently divides by zero on constant event flags
    "ignore::DeprecationWarning",
    "ignore::FutureWarning",
)


@pytest.fixture
def loader(settings: Settings, raw_data_dir: Path) -> Any:
    dataset, _ = prepare_data(settings)
    model = PersistenceClassifier().fit(dataset.features, dataset.target)
    served = ServedModel("champion", model, None, {"version": "4", "candidate": "persistence"})
    return lambda s, alias: served


def _no_model(settings: Settings, alias: str) -> ServedModel:
    raise RuntimeError("registry empty")


# ── real Evidently runs ───────────────────────────────────────


def test_a_drift_run_writes_the_report_summary_and_history(settings: Settings, loader: Any) -> None:
    summary = monitor_drift(settings, loader=loader)
    monitor_drift(settings, loader=loader)  # a second day

    folder = drift_dir(settings)
    assert "<html" in (folder / "report.html").read_text().lower()
    saved = json.loads((folder / "summary.json").read_text())
    assert saved["alert"] == summary.alert
    assert len((folder / "history.jsonl").read_text().splitlines()) == 2

    assert summary.source == "snapshot"
    assert summary.current.rows == settings.monitoring.current_days
    assert summary.reference.rows == settings.monitoring.reference_days
    assert summary.reference.end < summary.current.start
    tested = {c.column for c in summary.columns}
    assert PREDICTION in tested  # prediction drift
    assert not tested & set(settings.monitoring.exclude)  # no calendar features
    assert summary.model == {"alias": "champion", "version": "4", "candidate": "persistence"}
    assert summary.prediction_drift is not None
    expected_share = len(summary.drifted_features) / (len(tested) - 1)
    assert summary.drifted_share == pytest.approx(expected_share)
    assert summary.alert == (
        summary.drifted_share >= settings.monitoring.drift_share or bool(summary.prediction_drift)
    )


def test_without_a_model_data_drift_is_still_reported(
    settings: Settings, raw_data_dir: Path
) -> None:
    summary = monitor_drift(settings, loader=_no_model)

    assert summary.prediction_drift is None
    assert summary.model == {}
    assert any("prediction drift skipped" in reason for reason in summary.reasons)
    assert PREDICTION not in {c.column for c in summary.columns}
    assert (drift_dir(settings) / "report.html").is_file()


# ── alert rules (Evidently stubbed) ───────────────────────────


class _Snapshot:
    def save_html(self, path: str) -> None:
        Path(path).write_text("<html></html>")


def _stub_evidently(monkeypatch: pytest.MonkeyPatch, drifted: set[str]) -> None:
    def fake(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], share: float):  # type: ignore[no-untyped-def]
        tests = [
            ColumnDrift(c, "K-S p_value", 0.05, 0.01 if c in drifted else 0.5, c in drifted)
            for c in columns
        ]
        return _Snapshot(), tests

    monkeypatch.setattr(drift, "run_evidently", fake)


@pytest.mark.parametrize(
    ("drifted", "on_prediction", "alert"),
    [
        (set(), True, False),
        ({PREDICTION}, True, True),
        ({PREDICTION}, False, False),
        ({"vix", "dxy_index"}, True, False),
    ],
)
def test_alert_rules(
    settings: Settings,
    loader: Any,
    monkeypatch: pytest.MonkeyPatch,
    drifted: set[str],
    on_prediction: bool,
    alert: bool,
) -> None:
    _stub_evidently(monkeypatch, drifted)
    monitoring = settings.monitoring.model_copy(update={"alert_on_prediction_drift": on_prediction})

    summary = monitor_drift(settings.model_copy(update={"monitoring": monitoring}), loader=loader)

    assert summary.alert is alert
    assert summary.prediction_drift is (PREDICTION in drifted)


def test_most_features_drifting_raises_an_alert(
    settings: Settings, loader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, features, _ = feature_history(settings)
    kept = [f for f in features if f not in settings.monitoring.exclude]
    _stub_evidently(monkeypatch, set(kept[: len(kept) // 2 + 1]))

    summary = monitor_drift(settings, loader=loader)

    assert summary.alert
    assert summary.drifted_share > 0.5
    assert summary.reasons[0].startswith("ALERT:")


def test_live_data_is_monitored_once_ingested(
    settings: Settings, loader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_evidently(monkeypatch, set())
    oil = next(settings.paths.data_dir.glob("oil_*.csv"))
    live = settings.paths.live_dir / oil.name
    live.parent.mkdir(parents=True)
    live.write_text(oil.read_text())

    assert monitor_drift(settings, loader=loader, predictions=False).source == "live"


def test_notifications(settings: Settings, loader: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []
    _stub_evidently(monkeypatch, {PREDICTION})
    run_and_notify(settings, notify=lambda t, lines: sent.append(t), loader=loader)
    _stub_evidently(monkeypatch, set())
    run_and_notify(settings, notify=lambda t, lines: sent.append(t), loader=loader)
    run_and_notify(settings, notify=lambda t, lines: sent.append(t), loader=loader, always=True)

    assert sent == ["Brent drift ALERT", "Brent drift: no alert"]


def test_the_default_notifier_reads_the_webhook_setting(
    settings: Settings, loader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[Any] = []
    _stub_evidently(monkeypatch, {PREDICTION})
    monkeypatch.setattr(
        "brent_forecast.notify.notify", lambda url, title, lines, timeout: posted.append(url)
    )

    run_and_notify(settings, loader=loader)

    assert posted == [None]  # no webhook configured: logged only


# ── helpers ───────────────────────────────────────────────────


def test_parse_column_drift_handles_p_values_and_distances() -> None:
    payload = {
        "metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {}},
            {
                "config": {
                    "type": "x:ValueDrift",
                    "column": "a",
                    "method": "K-S p_value",
                    "threshold": 0.05,
                },
                "value": 0.01,
            },
            {
                "config": {
                    "type": "x:ValueDrift",
                    "column": "b",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.05,
            },
            {
                "config": {
                    "type": "x:ValueDrift",
                    "column": "c",
                    "method": "Jensen-Shannon distance",
                    "threshold": 0.1,
                },
                "value": 0.2,
            },
        ]
    }

    tests = {t.column: t.drifted for t in parse_column_drift(payload)}

    assert tests == {"a": True, "b": False, "c": True}


def test_windows() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "x": range(100)})

    reference, current = split_windows(frame, current_days=20, reference_days=None)
    assert (len(reference), len(current)) == (80, 20)
    assert current["x"].iloc[0] == 80
    reference, _ = split_windows(frame, current_days=20, reference_days=50)
    assert reference["x"].tolist() == list(range(30, 80))
    with pytest.raises(ValueError, match="at least 120"):
        split_windows(frame, current_days=40, reference_days=None)


def test_summary_lines() -> None:
    summary = DriftSummary(
        source="live",
        reference=drift.Window("2023-01-02", "2025-12-31", 700),
        current=drift.Window("2026-01-02", "2026-03-31", 60),
        columns=[],
        drifted_features=["vix"],
        drifted_share=0.04,
        prediction_drift=False,
        alert=False,
        reasons=[],
    )

    text = "\n".join(summary.lines())

    assert "60 days, live" in text
    assert "drifted features: 1 (4%): vix" in text
    assert "prediction drift: no" in text


# ── CLI ───────────────────────────────────────────────────────


def _config(settings: Settings, path: Path) -> Path:
    import yaml

    payload = settings.model_dump(mode="json")
    payload["notify"]["webhook_url"] = None
    path.write_text(yaml.safe_dump(payload))
    return path


def test_cli_prints_the_summary_and_can_fail_on_alerts(
    settings: Settings, raw_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_evidently(monkeypatch, {"vix"} | set(feature_history(settings)[1]))
    config = _config(settings, tmp_path / "c.yaml")
    base = ["--config", str(config), "--log-level", "WARNING", "monitor", "drift"]

    ok = CliRunner().invoke(app, [*base, "--no-predictions"])
    failing = CliRunner().invoke(app, [*base, "--no-predictions", "--fail-on-alert"])

    assert ok.exit_code == 0, ok.output
    assert json.loads(ok.output)["alert"] is True
    assert failing.exit_code == 2


def test_cli_reports_missing_data(settings: Settings, tmp_path: Path) -> None:
    config = _config(settings, tmp_path / "c.yaml")

    result = CliRunner().invoke(app, ["--config", str(config), "monitor", "drift"])

    assert result.exit_code == 1
    assert "Drift monitoring failed" in result.output
