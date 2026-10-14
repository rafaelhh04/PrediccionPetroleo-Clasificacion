"""Data and prediction drift monitoring with Evidently (``brent monitor drift``).

The model is used on the most recent trading days; drift monitoring asks
whether those inputs, and the model's outputs on them, still look like the data
it was fitted on:

- **current**: the feature rows of the last ``monitoring.current_days`` days of
  the series the API predicts from (the live copy when ``brent data ingest`` has
  run, the snapshot otherwise);
- **reference**: the ``monitoring.reference_days`` rows before them (the
  training data of the served model, which is refitted on everything labelled);
- **prediction**: ``P(up)`` of the served model (``serving.alias``) on both
  windows, compared like one more column (prediction drift).

Evidently's ``DataDriftPreset`` picks a test per column (Kolmogorov-Smirnov
p-value for these sample sizes) and counts the drifted columns. An **alert** is
raised when the share of drifted features reaches ``monitoring.drift_share``
or, with ``monitoring.alert_on_prediction_drift``, when the predictions
drifted. The HTML report, a JSON summary and a history line go to
``results/drift/``; an alert is sent to the notification webhook.

Drift is a reason to look, not a verdict: with ~60 current days some features
(volatility regimes, the GPR index carried forward on live data) drift often.
"""

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from brent_forecast.config import Settings

logger = logging.getLogger(__name__)

PREDICTION = "prediction"
"""Column of the served model's P(up) in the drift report."""


@dataclass(frozen=True)
class ColumnDrift:
    """Drift test of one column (as computed by Evidently)."""

    column: str
    method: str
    threshold: float
    score: float
    drifted: bool


@dataclass(frozen=True)
class Window:
    """Date range and size of one side of the comparison."""

    start: str
    end: str
    rows: int


@dataclass(frozen=True)
class DriftSummary:
    """Machine-readable outcome of a drift run (``results/drift/summary.json``)."""

    source: str
    reference: Window
    current: Window
    columns: list[ColumnDrift]
    drifted_features: list[str]
    drifted_share: float
    prediction_drift: bool | None
    """``None`` when no model could be loaded (data drift only)."""
    alert: bool
    reasons: list[str]
    model: dict[str, str] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def lines(self) -> list[str]:
        """Human-readable summary for the notification."""
        lines = [
            f"current: {self.current.start} -> {self.current.end} ({self.current.rows} days, "
            f"{self.source})",
            f"reference: {self.reference.start} -> {self.reference.end} "
            f"({self.reference.rows} days)",
            f"drifted features: {len(self.drifted_features)} "
            f"({self.drifted_share:.0%}): {', '.join(self.drifted_features) or 'none'}",
        ]
        if self.prediction_drift is not None:
            lines.append(f"prediction drift: {'yes' if self.prediction_drift else 'no'}")
        return [*lines, *self.reasons]


ModelLoader = Callable[[Settings, str], Any]
"""``(settings, alias) -> ServedModel``, e.g. ``api.service.load_from_registry``."""


def drift_dir(settings: Settings) -> Path:
    """Where the report, the summary and the history are written."""
    return settings.paths.results_dir / "drift"


def feature_history(settings: Settings) -> tuple[pd.DataFrame, list[str], str]:
    """Dated feature rows of the series the API predicts from, the feature names and the source.

    Built like :func:`brent_forecast.predict.latest_features` (no label: the last
    days are included), from the live prices when they exist.
    """
    from brent_forecast.data.load import DateBounds, load_oil_data
    from brent_forecast.features.preprocessing import engineer_features, handle_nulls

    paths, data = settings.paths, settings.data
    live = paths.live_dir / data.oil_filename
    if live.is_file():
        oil, source = live, "live"
    else:
        oil, source = paths.data_dir / data.oil_filename, "snapshot"
    frame = load_oil_data(
        oil,
        paths.data_dir / data.events_filename,
        DateBounds(data.expected_start_min, data.expected_start_max, data.expected_end_min),
    )
    frame, columns = engineer_features(frame)
    frame = handle_nulls(frame, columns)
    return frame[["date", *columns]].reset_index(drop=True), columns, source


def split_windows(
    frame: pd.DataFrame, current_days: int, reference_days: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into reference rows and current rows (the last ``current_days``).

    Raises
    ------
    ValueError
        If there are not enough rows for both windows.
    """
    if len(frame) < current_days + 2 * current_days:
        raise ValueError(
            f"Need at least {3 * current_days} feature rows for a {current_days}-day current "
            f"window and a reference twice as long; got {len(frame)}."
        )
    current = frame.iloc[-current_days:]
    reference = frame.iloc[:-current_days]
    if reference_days is not None:
        reference = reference.iloc[-reference_days:]
    return reference.reset_index(drop=True), current.reset_index(drop=True)


def _window(frame: pd.DataFrame) -> Window:
    dates = frame["date"]
    return Window(str(dates.min().date()), str(dates.max().date()), len(frame))


def run_evidently(
    reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str], drift_share: float
) -> tuple[Any, list[ColumnDrift]]:
    """Run Evidently's ``DataDriftPreset`` on ``columns``; return the snapshot and the tests."""
    # Evidently's telemetry (UI service) must never phone home from a job or a test.
    os.environ.setdefault("ITERATIVE_DO_NOT_TRACK", "do-not-track")
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=list(columns))
    report = Report([DataDriftPreset(drift_share=drift_share)])
    snapshot = report.run(
        Dataset.from_pandas(current[list(columns)], data_definition=definition),
        Dataset.from_pandas(reference[list(columns)], data_definition=definition),
    )
    return snapshot, parse_column_drift(snapshot.dict())


def parse_column_drift(payload: dict[str, Any]) -> list[ColumnDrift]:
    """Per-column tests (``ValueDrift`` metrics) of an Evidently snapshot dictionary.

    A p-value method drifts when the p-value is *below* the threshold; a
    distance (Wasserstein, Jensen-Shannon...) when the distance reaches it.
    """
    tests = []
    for metric in payload.get("metrics", []):
        config = metric.get("config", {})
        if not str(config.get("type", "")).endswith("ValueDrift"):
            continue
        method, threshold = str(config["method"]), float(config["threshold"])
        score = float(metric["value"])
        drifted = score < threshold if "p_value" in method else score >= threshold
        tests.append(ColumnDrift(str(config["column"]), method, threshold, score, drifted))
    return tests


def _predict(served: Any, frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    names = list(served.columns) if served.columns is not None else list(columns)
    return pd.Series(served.model.predict_proba(frame[names])[:, 1], index=frame.index)


def _default_loader(settings: Settings, alias: str) -> Any:
    from brent_forecast.api.service import load_from_registry

    return load_from_registry(settings, alias)


def monitor_drift(
    settings: Settings,
    *,
    predictions: bool = True,
    loader: ModelLoader = _default_loader,
) -> DriftSummary:
    """Compare the current window with the reference; write the report; return the summary.

    The served model (``serving.alias``) is loaded read-only from the registry for
    prediction drift; if none is available the run reports data drift only.

    Raises
    ------
    FileNotFoundError, ValueError
        If the data files are missing, invalid or too short for both windows.
    """
    cfg = settings.monitoring
    frame, inputs, source = feature_history(settings)
    features = [c for c in inputs if c not in cfg.exclude]
    reference, current = split_windows(frame, cfg.current_days, cfg.reference_days)

    columns, reasons, model = list(features), [], {}
    if predictions:
        try:
            served = loader(settings, settings.serving.alias)
        except Exception as exc:  # no registry yet: data drift is still worth reporting
            reasons.append(f"prediction drift skipped: no model ({type(exc).__name__}: {exc})")
            logger.warning("Prediction drift skipped: %s", exc)
        else:
            reference[PREDICTION] = _predict(served, reference, inputs)
            current[PREDICTION] = _predict(served, current, inputs)
            columns.append(PREDICTION)
            model = {
                "alias": served.alias,
                **{k: served.info.get(k, "") for k in ("version", "candidate")},
            }

    snapshot, tests = run_evidently(reference, current, columns, cfg.drift_share)
    by_column = {t.column: t for t in tests}
    drifted = [c for c in features if c in by_column and by_column[c].drifted]
    share = len(drifted) / len(features) if features else 0.0
    prediction_drift = by_column[PREDICTION].drifted if PREDICTION in by_column else None

    if share >= cfg.drift_share:
        reasons.insert(
            0, f"ALERT: {share:.0%} of the features drifted (threshold {cfg.drift_share:.0%})"
        )
    if prediction_drift and cfg.alert_on_prediction_drift:
        reasons.insert(0, "ALERT: the distribution of the model's predictions drifted")
    alert = share >= cfg.drift_share or bool(prediction_drift and cfg.alert_on_prediction_drift)

    summary = DriftSummary(
        source=source,
        reference=_window(reference),
        current=_window(current),
        columns=tests,
        drifted_features=drifted,
        drifted_share=share,
        prediction_drift=prediction_drift,
        alert=alert,
        reasons=reasons,
        model=model,
    )
    write_outputs(settings, summary, snapshot)
    return summary


def write_outputs(settings: Settings, summary: DriftSummary, snapshot: Any) -> Path:
    """Write ``report.html`` and ``summary.json``, append to ``history.jsonl``; return the dir."""
    folder = drift_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(folder / "report.html"))
    payload = asdict(summary)
    (folder / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    line = {k: payload[k] for k in ("generated_at", "drifted_share", "prediction_drift", "alert")}
    line["current_end"] = summary.current.end
    with (folder / "history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(line) + "\n")
    logger.info("Drift report written to %s (alert: %s)", folder, summary.alert)
    return folder


def run_and_notify(
    settings: Settings,
    *,
    notify: Callable[[str, list[str]], Any] | None = None,
    always: bool = False,
    **options: Any,
) -> DriftSummary:
    """:func:`monitor_drift`, then notify an alert (or every run with ``always``)."""
    summary = monitor_drift(settings, **options)
    if notify is None:
        from brent_forecast.notify import notify as send

        url = settings.notify.webhook_url
        secret = url.get_secret_value() if url else None

        def notify(title: str, lines: list[str]) -> Any:
            return send(secret, title, lines, timeout=settings.notify.timeout_seconds)

    if summary.alert:
        notify("Brent drift ALERT", summary.lines())
    elif always:
        notify("Brent drift: no alert", summary.lines())
    return summary
