"""Tests for brent_forecast.retrain (scheduled retraining) with injected stages.

The stages (ingest, train, evaluate, explain) are fakes that record their calls,
so the orchestration, idempotency and notifications are tested in milliseconds;
one slow test runs the real stages end to end.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from brent_forecast.cli import app
from brent_forecast.config import Settings
from brent_forecast.data.ingest import IngestionError
from brent_forecast.retrain import (
    Outcome,
    Steps,
    fingerprint,
    input_files,
    outcome_file,
    retrain,
    retrain_dir,
    run_and_notify,
    staged_store,
    state_file,
)
from brent_forecast.tracking import Decision


@pytest.fixture
def tracked(settings: Settings, raw_data_dir: Path) -> Settings:
    tracking = settings.tracking.model_copy(update={"enabled": True})
    return settings.model_copy(update={"tracking": tracking})


class FakeRegistry:
    """Fake stages: training registers a new version; the decision is configurable."""

    def __init__(self, decides: str = "persistence") -> None:
        self.decides = decides
        self.calls: list[str] = []
        self.champion_info: dict[str, str] = {}
        self.version = 0
        self.ingest_error: Exception | None = None
        self.train_error: Exception | None = None
        self.uris: list[str] = []

    def steps(self) -> Steps:
        return Steps(
            ingest=self.ingest,
            train=self.train,
            evaluate=self.evaluate,
            explain=self.explain,
            champion=lambda s: dict(self.champion_info),
        )

    def ingest(self, settings: Settings) -> int:
        self.calls.append("ingest")
        if self.ingest_error:
            raise self.ingest_error
        return 2

    def train(self, settings: Settings, dataset: Path) -> None:
        self.calls.append("train")
        self.uris.append(settings.tracking.uri)
        assert dataset.is_file()
        if self.train_error:
            raise self.train_error

    def evaluate(self, settings: Settings) -> Decision:
        self.calls.append("evaluate")
        self.version += 1
        self.champion_info = {"version": str(self.version), "candidate": self.decides}
        return Decision(champion=self.decides, challenger="other", reason=f"{self.decides} wins.")

    def explain(self, settings: Settings) -> None:
        self.calls.append("explain")


def test_a_first_run_trains_promotes_and_records_its_fingerprint(tracked: Settings) -> None:
    registry = FakeRegistry()

    outcome = retrain(tracked, steps=registry.steps())

    assert registry.calls == ["ingest", "train", "evaluate", "explain"]
    assert outcome.status == "promoted"  # no champion before
    assert outcome.champion_after == {"version": "1", "candidate": "persistence"}
    assert outcome.new_rows == 2
    assert outcome.data_source == "snapshot"
    assert outcome.data_end is not None
    assert (retrain_dir(tracked) / "dataset.parquet").is_file()
    state = json.loads(state_file(tracked).read_text())
    assert state["fingerprint"] == outcome.fingerprint
    assert json.loads(outcome_file(tracked).read_text())["status"] == "promoted"


def test_a_rerun_without_changes_is_skipped(tracked: Settings) -> None:
    registry = FakeRegistry()
    retrain(tracked, steps=registry.steps())
    registry.calls.clear()

    outcome = retrain(tracked, steps=registry.steps())

    assert registry.calls == ["ingest"]  # nothing trained, nothing registered
    assert outcome.status == "skipped"
    expected = {"version": "1", "candidate": "persistence"}
    assert outcome.champion_before == expected
    assert outcome.champion_after == expected
    assert "skipped" in json.loads(outcome_file(tracked).read_text())["status"]


def test_force_retrains_and_keeps_the_same_champion_as_a_refresh(tracked: Settings) -> None:
    registry = FakeRegistry()
    retrain(tracked, steps=registry.steps())

    outcome = retrain(tracked, steps=registry.steps(), force=True, ingest=False, explain=False)

    assert outcome.status == "refreshed"
    assert outcome.champion_before["version"] == "1"
    assert outcome.champion_after["version"] == "2"
    assert outcome.new_rows is None


def test_a_significant_model_is_reported_as_a_new_champion(tracked: Settings) -> None:
    registry = FakeRegistry()
    retrain(tracked, steps=registry.steps())
    registry.decides = "lightgbm"

    outcome = retrain(tracked, steps=registry.steps(), force=True)

    assert outcome.status == "promoted"
    assert outcome.champion_after["candidate"] == "lightgbm"
    assert any("persistence v1 -> lightgbm v2" in line for line in outcome.lines())


def test_new_data_or_a_new_configuration_trigger_a_retraining(tracked: Settings) -> None:
    oil, events, _ = input_files(tracked)
    base = fingerprint(tracked, (oil, events))

    reseeded = tracked.model_copy(update={"seed": 7})
    assert fingerprint(reseeded, (oil, events)) != base
    serving = tracked.serving.model_copy(update={"alias": "challenger"})
    assert fingerprint(tracked.model_copy(update={"serving": serving}), (oil, events)) == base

    live = tracked.paths.live_dir / oil.name
    live.parent.mkdir(parents=True)
    live.write_text(oil.read_text())  # `brent data ingest` creates the live copy
    assert input_files(tracked)[2] == "live"
    assert fingerprint(tracked, (live, events)) == base  # same content
    live.write_text(oil.read_text() + "\n")
    assert fingerprint(tracked, (live, events)) != base


def test_live_data_is_used_once_ingested(tracked: Settings) -> None:
    oil, _, _ = input_files(tracked)
    live = tracked.paths.live_dir / oil.name
    live.parent.mkdir(parents=True)
    live.write_text(oil.read_text())

    outcome = retrain(tracked, steps=FakeRegistry().steps(), ingest=False)

    assert outcome.data_source == "live"


def test_a_failed_ingestion_retrains_on_the_existing_data(tracked: Settings) -> None:
    registry = FakeRegistry()
    registry.ingest_error = IngestionError("Yahoo is down")

    outcome = retrain(tracked, steps=registry.steps())

    assert outcome.status == "promoted"
    assert outcome.new_rows is None
    assert outcome.warnings == ["ingestion failed, using the existing data (Yahoo is down)"]
    assert "warning: ingestion failed" in "\n".join(outcome.lines())


def test_retraining_needs_tracking(settings: Settings, raw_data_dir: Path) -> None:
    with pytest.raises(ValueError, match="tracking.enabled"):
        retrain(settings, steps=FakeRegistry().steps())


def test_outcomes_are_notified(tracked: Settings) -> None:
    sent: list[tuple[str, list[str]]] = []
    registry = FakeRegistry()

    run_and_notify(tracked, notify=lambda t, lines: sent.append((t, lines)), steps=registry.steps())
    run_and_notify(tracked, notify=lambda t, lines: sent.append((t, lines)), steps=registry.steps())

    assert [title for title, _ in sent] == [
        "Brent retraining: NEW champion promoted",
        "Brent retraining: skipped (nothing changed)",
    ]
    assert "status: promoted" in sent[0][1]


def test_a_failure_is_notified_recorded_and_raised(tracked: Settings) -> None:
    sent: list[tuple[str, list[str]]] = []
    registry = FakeRegistry()
    retrain(tracked, steps=registry.steps())
    previous_state = state_file(tracked).read_text()
    registry.train_error = MemoryError("out of memory")

    with pytest.raises(MemoryError):
        run_and_notify(
            tracked,
            notify=lambda t, lines: sent.append((t, lines)),
            steps=registry.steps(),
            force=True,
        )

    assert sent[0][0] == "Brent retraining FAILED"
    assert "MemoryError: out of memory" in sent[0][1]
    assert json.loads(outcome_file(tracked).read_text())["status"] == "failed"
    assert state_file(tracked).read_text() == previous_state  # last success kept


def test_the_default_notifier_uses_the_configured_webhook(
    tracked: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import SecretStr

    posted: list[tuple[str | None, str]] = []
    monkeypatch.setattr(
        "brent_forecast.notify.notify",
        lambda url, title, lines, timeout: posted.append((url, title)),
    )
    notify_cfg = tracked.notify.model_copy(update={"webhook_url": SecretStr("https://hook")})

    run_and_notify(tracked.model_copy(update={"notify": notify_cfg}), steps=FakeRegistry().steps())

    assert posted == [("https://hook", "Brent retraining: NEW champion promoted")]


# ── staged SQLite store ───────────────────────────────────────


def _db_with(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
        db.execute("INSERT INTO t VALUES (?)", (value,))


def _values(path: Path) -> list[str]:
    with sqlite3.connect(path) as db:
        return [row[0] for row in db.execute("SELECT v FROM t")]


def test_a_staged_store_is_published_only_on_success(tracked: Settings) -> None:
    database = Path(tracked.tracking.uri.removeprefix("sqlite:///"))
    _db_with(database, "before")

    with staged_store(tracked) as staged:
        local = Path(staged.tracking.uri.removeprefix("sqlite:///"))
        assert local != database and _values(local) == ["before"]
        _db_with(local, "run")
        assert _values(database) == ["before"]  # untouched while the run is going on
    assert _values(database) == ["before", "run"]

    with pytest.raises(RuntimeError), staged_store(tracked) as staged:
        _db_with(Path(staged.tracking.uri.removeprefix("sqlite:///")), "failed run")
        raise RuntimeError("training crashed")
    assert _values(database) == ["before", "run"]


def test_open_connections_see_the_published_store(tracked: Settings) -> None:
    # MLflow keeps one engine per URI for the whole process: publishing must not swap the
    # file under it (a rename would leave it reading the deleted, old database).
    database = Path(tracked.tracking.uri.removeprefix("sqlite:///"))
    _db_with(database, "before")
    held = sqlite3.connect(database)
    try:
        with staged_store(tracked) as staged:
            _db_with(Path(staged.tracking.uri.removeprefix("sqlite:///")), "run")
        assert [row[0] for row in held.execute("SELECT v FROM t")] == ["before", "run"]
    finally:
        held.close()


def test_a_first_staged_run_creates_the_store(tracked: Settings) -> None:
    database = Path(tracked.tracking.uri.removeprefix("sqlite:///"))

    with staged_store(tracked) as staged:
        _db_with(Path(staged.tracking.uri.removeprefix("sqlite:///")), "first")

    assert _values(database) == ["first"]


def test_a_server_store_is_not_staged(tracked: Settings) -> None:
    tracking = tracked.tracking.model_copy(update={"uri": "http://mlflow:5000"})
    server = tracked.model_copy(update={"tracking": tracking})

    with staged_store(server) as staged:
        assert staged is server


def test_retrain_trains_against_the_staged_store(tracked: Settings) -> None:
    registry = FakeRegistry()

    retrain(tracked, steps=registry.steps(), stage=True)

    assert registry.uris and registry.uris[0] != tracked.tracking.uri


# ── CLI ───────────────────────────────────────────────────────


def _config(settings: Settings, path: Path) -> Path:
    import yaml

    payload = settings.model_dump(mode="json")
    payload["notify"]["webhook_url"] = None
    path.write_text(yaml.safe_dump(payload))
    return path


def test_cli_prints_the_outcome(
    tracked: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake(settings: Settings, **options: Any) -> Outcome:
        seen.update(options)
        return Outcome(status="skipped", fingerprint="abc")

    monkeypatch.setattr("brent_forecast.retrain.run_and_notify", fake)
    config = _config(tracked, tmp_path / "c.yaml")

    result = CliRunner().invoke(
        app,
        ["--config", str(config), "--log-level", "WARNING", "retrain", "--no-ingest", "--force"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "skipped"
    assert seen == {"ingest": False, "explain": True, "force": True, "stage": False}


def test_cli_exits_with_1_on_failure(
    tracked: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(settings: Settings, **options: Any) -> Outcome:
        raise RuntimeError("registry locked")

    monkeypatch.setattr("brent_forecast.retrain.run_and_notify", broken)
    config = _config(tracked, tmp_path / "c.yaml")

    result = CliRunner().invoke(app, ["--config", str(config), "retrain", "--stage-store"])

    assert result.exit_code == 1
    assert "registry locked" in result.output


# ── end to end (real stages) ──────────────────────────────────


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
def test_retrain_end_to_end(fast_settings: Settings, raw_data_dir: Path) -> None:
    tracking = fast_settings.tracking.model_copy(update={"enabled": True})
    settings = fast_settings.model_copy(update={"tracking": tracking})
    sent: list[str] = []

    first = run_and_notify(
        settings, notify=lambda t, lines: sent.append(t), ingest=False, explain=False, stage=True
    )
    second = run_and_notify(
        settings, notify=lambda t, lines: sent.append(t), ingest=False, explain=False, stage=True
    )

    assert first.status == "promoted" and first.champion_after["version"] == "1"
    assert second.status == "skipped"
    from brent_forecast.tracking import registry_aliases

    assert set(registry_aliases(settings)) == {"champion", "challenger"}
    assert len(sent) == 2
