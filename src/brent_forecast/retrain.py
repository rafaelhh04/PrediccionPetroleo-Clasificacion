"""Scheduled retraining (``brent retrain``): ingest, train, evaluate and promote, explain.

One run of the scheduled job (Cloud Run Job + Cloud Scheduler, see ``infra/``):

1. **ingest** the days since the last available date (``brent data ingest``);
   if a source is down the run goes on with the data it has;
2. build the dataset from the **live** prices file when it exists (snapshot plus
   ingested days), otherwise from the raw snapshot, and write it to
   ``results/retrain/dataset.parquet``;
3. **skip** if the data, the training configuration and the code version are
   those of the last successful run (the fingerprint in ``retrain_state.json``):
   re-running the job, or running it on a day without new data, changes nothing;
4. **train** on that dataset and log the run to MLflow (``brent train``);
5. **report and promote** (``brent report``): the promotion rule of
   :func:`~brent_forecast.tracking.select_champion` decides. A model replaces the
   baseline as ``champion`` only if it beats it significantly; otherwise the
   refitted baseline stays champion. Either way the alias moves to the version
   trained on the new data and the API picks it up;
6. **explain** the models (``brent explain``);
7. write ``retrain.json`` and send a notification (:mod:`brent_forecast.notify`).

On a SQLite tracking store the run can work on a local copy of the database
(``stage_store``) that replaces the original only after a successful run: on
Cloud Storage FUSE every SQLite write would rewrite the whole object, and a
failed run leaves the registry untouched.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from brent_forecast.config import Settings
from brent_forecast.data.download import sha256sum
from brent_forecast.tracking import CHAMPION, Decision

logger = logging.getLogger(__name__)

Status = Literal["promoted", "refreshed", "skipped", "failed"]

# Sections of the configuration that change what a training run produces.
TRAINING_SECTIONS = frozenset(
    {"seed", "split", "preprocessing", "validation", "evaluation", "backtest", "tuning", "models"}
)


@dataclass(frozen=True)
class Outcome:
    """Result of one retraining run (written to ``retrain.json`` and notified)."""

    status: Status
    """``promoted`` (new champion candidate), ``refreshed`` (same candidate, new
    version), ``skipped`` (nothing changed) or ``failed``."""
    fingerprint: str | None
    data_source: str | None = None
    data_end: str | None = None
    new_rows: int | None = None
    champion_before: dict[str, str] = field(default_factory=dict)
    champion_after: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    finished_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def lines(self) -> list[str]:
        """Human-readable summary, one line per fact (for the notification)."""
        before = _describe(self.champion_before)
        after = _describe(self.champion_after)
        lines = [f"status: {self.status}"]
        if self.data_end:
            lines.append(f"data: {self.data_source} up to {self.data_end}")
        if self.new_rows is not None:
            lines.append(f"ingested: {self.new_rows} new day(s)")
        if self.status in ("promoted", "refreshed"):
            lines.append(f"champion: {before} -> {after}")
        elif self.champion_before:
            lines.append(f"champion: {before}")
        if self.reason:
            lines.append(self.reason)
        lines += [f"warning: {w}" for w in self.warnings]
        lines.append(f"duration: {self.seconds / 60:.1f} min")
        return lines


def _describe(info: dict[str, str]) -> str:
    if not info:
        return "none"
    return f"{info.get('candidate', '?')} v{info.get('version', '?')}"


@dataclass(frozen=True)
class Steps:
    """The stages of a run (injectable, so tests do not train for minutes)."""

    ingest: Callable[[Settings], int]
    """Fetch recent data; return the number of new days."""
    train: Callable[[Settings, Path], None]
    evaluate: Callable[[Settings], Decision]
    explain: Callable[[Settings], Any]
    champion: Callable[[Settings], dict[str, str]]


def _ingest(settings: Settings) -> int:
    from brent_forecast.data.ingest import ingest

    return ingest(settings).new_rows


def _train(settings: Settings, dataset: Path) -> None:
    from brent_forecast.pipeline import run
    from brent_forecast.tracking import log_training

    log_training(settings, run(settings, dataset))


def _evaluate(settings: Settings) -> Decision:
    from brent_forecast.evaluation.report import generate_report
    from brent_forecast.tracking import log_evaluation

    generate_report(settings)
    return log_evaluation(settings)


def _explain(settings: Settings) -> Path:
    from brent_forecast.evaluation.explain import generate_explanations

    return generate_explanations(settings)


def _champion(settings: Settings) -> dict[str, str]:
    from brent_forecast.tracking import registry_aliases

    return registry_aliases(settings).get(CHAMPION, {})


DEFAULT_STEPS = Steps(
    ingest=_ingest, train=_train, evaluate=_evaluate, explain=_explain, champion=_champion
)


# ── files ─────────────────────────────────────────────────────


def retrain_dir(settings: Settings) -> Path:
    """Working directory of the retraining job (dataset)."""
    return settings.paths.results_dir / "retrain"


def state_file(settings: Settings) -> Path:
    """Fingerprint and summary of the last *successful* run (the idempotency key)."""
    return settings.paths.results_dir / "retrain_state.json"


def outcome_file(settings: Settings) -> Path:
    """Summary of the last run, whatever its status."""
    return settings.paths.results_dir / "retrain.json"


def input_files(settings: Settings) -> tuple[Path, Path, str]:
    """Prices file (live copy if any), events file and the name of the source."""
    paths, data = settings.paths, settings.data
    live = paths.live_dir / data.oil_filename
    oil, source = (
        (live, "live") if live.is_file() else (paths.data_dir / data.oil_filename, "snapshot")
    )
    return oil, paths.data_dir / data.events_filename, source


def fingerprint(settings: Settings, files: tuple[Path, ...]) -> str:
    """SHA-256 over the input files, the training configuration and the package version."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.name}={sha256sum(path)}\n".encode())
    config = settings.model_dump(mode="json", include=set(TRAINING_SECTIONS))
    digest.update(json.dumps(config, sort_keys=True).encode())
    digest.update(version("brent-forecast").encode())
    return digest.hexdigest()


def build_dataset(settings: Settings, oil: Path, events: Path, output: Path) -> str:
    """Featurize ``oil`` + ``events`` like ``brent featurize``; return the last date."""
    from brent_forecast.data.load import DateBounds, load_oil_data
    from brent_forecast.features.preprocessing import build_dataset as features

    data = settings.data
    frame = load_oil_data(
        oil,
        events,
        DateBounds(data.expected_start_min, data.expected_start_max, data.expected_end_min),
    )
    dataset = features(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(output, index=False)
    return str(dataset.frame["date"].max().date())


@contextmanager
def staged_store(settings: Settings) -> Iterator[Settings]:
    """Work on a local copy of a SQLite tracking store; publish it only on success.

    Other stores (a tracking server, PostgreSQL) are used as they are.
    """
    uri = settings.tracking.uri
    if not uri.startswith("sqlite:///"):
        yield settings
        return
    database = Path(uri.removeprefix("sqlite:///"))
    with tempfile.TemporaryDirectory(prefix="brent-mlflow-") as tmp:
        local = Path(tmp) / database.name
        if database.is_file():
            shutil.copyfile(database, local)
        tracking = settings.tracking.model_copy(update={"uri": f"sqlite:///{local}"})
        yield settings.model_copy(update={"tracking": tracking})
        # Reached only when the run succeeded: replace the store in one rename.
        if not local.is_file():  # nothing was tracked
            return
        database.parent.mkdir(parents=True, exist_ok=True)
        partial = database.with_name(f"{database.name}.partial")
        shutil.copyfile(local, partial)
        os.replace(partial, database)
        logger.info("Published the tracking store to %s", database)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ── the run ───────────────────────────────────────────────────


def retrain(
    settings: Settings,
    *,
    ingest: bool = True,
    explain: bool = True,
    force: bool = False,
    stage: bool = False,
    steps: Steps = DEFAULT_STEPS,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Run one retraining (see the module docstring); return its outcome.

    Raises
    ------
    ValueError
        If experiment tracking is disabled (there would be no registry to promote in).
    Exception
        Whatever a stage raises; :func:`run_and_notify` turns it into a ``failed`` outcome.
    """
    if not settings.tracking.enabled:
        raise ValueError("Retraining needs tracking.enabled = true (the registry decides).")
    start, warnings = clock(), []

    new_rows = None
    if ingest:
        from brent_forecast.data.ingest import IngestionError

        try:
            new_rows = steps.ingest(settings)
        except IngestionError as exc:  # a source is down: retrain on the data we have
            warnings.append(f"ingestion failed, using the existing data ({exc})")
            logger.warning("Ingestion failed: %s", exc)

    oil, events, source = input_files(settings)
    key = fingerprint(settings, (oil, events))
    previous = _read(state_file(settings))
    before = steps.champion(settings)
    if previous.get("fingerprint") == key and before and not force:
        outcome = Outcome(
            status="skipped",
            fingerprint=key,
            data_source=source,
            data_end=previous.get("data_end"),
            new_rows=new_rows,
            champion_before=before,
            champion_after=before,
            reason="Same data, configuration and code as the last successful run.",
            warnings=warnings,
            seconds=clock() - start,
        )
        _write(outcome_file(settings), asdict(outcome))
        logger.info("Retraining skipped: nothing changed since %s", previous.get("finished_at"))
        return outcome

    dataset = retrain_dir(settings) / "dataset.parquet"
    data_end = build_dataset(settings, oil, events, dataset)
    logger.info("Retraining on %s data up to %s", source, data_end)
    with staged_store(settings) if stage else _as_is(settings) as run_settings:
        steps.train(run_settings, dataset)
        decision = steps.evaluate(run_settings)
        after = steps.champion(run_settings)
    if explain:
        steps.explain(settings)

    status: Status = "promoted" if before.get("candidate") != decision.champion else "refreshed"
    outcome = Outcome(
        status=status,
        fingerprint=key,
        data_source=source,
        data_end=data_end,
        new_rows=new_rows,
        champion_before=before,
        champion_after=after,
        reason=decision.reason,
        warnings=warnings,
        seconds=clock() - start,
    )
    _write(outcome_file(settings), asdict(outcome))
    _write(state_file(settings), asdict(outcome))
    return outcome


@contextmanager
def _as_is(settings: Settings) -> Iterator[Settings]:
    yield settings


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_and_notify(
    settings: Settings,
    *,
    notify: Callable[[str, list[str]], Any] | None = None,
    **options: Any,
) -> Outcome:
    """:func:`retrain`, then notify the outcome; a failure is notified and re-raised."""
    if notify is None:
        from brent_forecast.notify import notify as send

        url = settings.notify.webhook_url
        secret = url.get_secret_value() if url else None

        def notify(title: str, lines: list[str]) -> Any:
            return send(secret, title, lines, timeout=settings.notify.timeout_seconds)

    try:
        outcome = retrain(settings, **options)
    except Exception as exc:
        failed = Outcome(status="failed", fingerprint=None, reason=f"{type(exc).__name__}: {exc}")
        _write(outcome_file(settings), asdict(failed))
        notify("Brent retraining FAILED", failed.lines())
        raise
    titles = {
        "promoted": "Brent retraining: NEW champion promoted",
        "refreshed": "Brent retraining: champion refreshed on new data",
        "skipped": "Brent retraining: skipped (nothing changed)",
    }
    notify(titles.get(outcome.status, "Brent retraining"), outcome.lines())
    return outcome
