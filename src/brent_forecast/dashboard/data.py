"""Data behind the dashboard: provenance, prices, the API prediction and the reports.

Every loader returns ``None`` (or an empty frame) when its artefact does not
exist yet, so the page can say which command produces it instead of failing.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from brent_forecast.config import Settings

DataKind = Literal["real", "synthetic", "mismatch", "unknown"]


@dataclass(frozen=True)
class Provenance:
    """Whether the results come from the published Kaggle dataset."""

    kind: DataKind
    message: str

    @property
    def is_real(self) -> bool:
        """True only when the data files match the recorded Kaggle checksums."""
        return self.kind == "real"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def provenance(settings: Settings) -> Provenance:
    """Classify the data of the last training run (``metrics.json``).

    ``brent train`` records the SHA-256 of the input files and whether they match the
    checksums committed for the Kaggle dataset: ``True`` is real data, ``None`` means
    that no checksum is recorded (the synthetic development dataset), ``False`` a
    different file.
    """
    metrics = _read_json(settings.paths.metrics_file)
    if metrics is None:
        return Provenance("unknown", "No training run yet: run `brent train` and `brent report`.")
    matches = metrics.get("data", {}).get("matches_recorded_checksums")
    if matches is True:
        return Provenance(
            "real", "Real data: the files match the recorded checksums of the Kaggle dataset."
        )
    if matches is False:
        return Provenance(
            "mismatch",
            "The data files do NOT match the recorded Kaggle checksums: results are not "
            "attributable to the published dataset.",
        )
    return Provenance(
        "synthetic",
        "SYNTHETIC data: no Kaggle checksum is recorded, so these results come from the "
        "synthetic development dataset (a random walk with the Kaggle schema). They "
        "demonstrate the methodology, not the Brent market.",
    )


def price_history(settings: Settings) -> pd.DataFrame:
    """Daily Brent and WTI closes (ingested data when available, else the raw snapshot)."""
    paths, data = settings.paths, settings.data
    for path in (paths.live_dir / data.oil_filename, paths.data_dir / data.oil_filename):
        if path.is_file():
            frame = pd.read_csv(path, usecols=["date", "brent_price", "wti_price"])
            frame["date"] = pd.to_datetime(frame["date"])
            return frame.sort_values("date").set_index("date")
    return pd.DataFrame(columns=["brent_price", "wti_price"])


class ApiError(RuntimeError):
    """The inference API is unreachable or answered with an error."""


@dataclass(frozen=True)
class ApiClient:
    """Minimal client of the inference API (standard library only)."""

    base_url: str
    timeout: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "ApiClient":
        """Client for ``dashboard.api_url``."""
        cfg = settings.dashboard
        return cls(cfg.api_url.rstrip("/"), cfg.timeout_seconds)

    def _call(self, method: str, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            data=b"{}" if method == "POST" else None,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload: dict[str, Any] = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"{method} {path} -> {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ApiError(f"{method} {path} failed: {exc}") from exc
        return payload

    def predict_latest(self) -> dict[str, Any]:
        """``POST /predict`` with an empty body: the next day after the latest data."""
        return self._call("POST", "/predict")

    def model_info(self) -> dict[str, Any]:
        """``GET /model-info``."""
        return self._call("GET", "/model-info")


def load_report(settings: Settings) -> dict[str, Any] | None:
    """Machine-readable statistical report (``brent report``)."""
    report: dict[str, Any] | None = _read_json(settings.paths.report_data_file)
    return report


def candidates_table(report: dict[str, Any]) -> pd.DataFrame:
    """One row per candidate: out-of-sample AUC with its CI and the comparison to the baseline."""
    rows = []
    for name, c in report["candidates"].items():
        comparison = report["comparisons"].get(name, {})
        rows.append(
            {
                "candidate": name,
                "kind": c["kind"],
                "auc": c["auc"]["estimate"],
                "auc_low": c["auc"]["low"],
                "auc_high": c["auc"]["high"],
                "delta_auc_vs_baseline": comparison.get("delta_auc", {}).get("estimate"),
                "p_holm": comparison.get("p_holm"),
                "significant": comparison.get("significant"),
                "brier": c.get("brier"),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


def backtest_table(report: dict[str, Any]) -> pd.DataFrame:
    """One row per strategy of the economic backtest (after costs)."""
    rows = [
        {
            "strategy": name,
            "kind": s["kind"],
            "sharpe": s["sharpe"],
            "cagr": s["cagr"],
            "max_drawdown": s["max_drawdown"],
            "hit_ratio": s["hit_ratio"],
            "n_trades": s["n_trades"],
        }
        for name, s in report["backtest"]["strategies"].items()
    ]
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False, ignore_index=True)


def shap_importances(settings: Settings) -> dict[str, pd.DataFrame]:
    """Model key -> mean |SHAP| per feature (``brent explain``), most important first."""
    explain_dir = settings.paths.explain_dir
    return {
        path.stem.removeprefix("shap_"): pd.read_csv(path)
        for path in sorted(explain_dir.glob("shap_*.csv"))
    }


def plot(settings: Settings, name: str) -> Path | None:
    """Path of a figure in ``results/plots`` if it exists."""
    path = settings.paths.plots_dir / name
    return path if path.is_file() else None


def drift_summary(settings: Settings) -> dict[str, Any] | None:
    """Last drift summary (``brent monitor drift``)."""
    from brent_forecast.monitoring.drift import drift_dir  # Evidently is imported lazily

    summary: dict[str, Any] | None = _read_json(drift_dir(settings) / "summary.json")
    return summary


def drift_history(settings: Settings) -> pd.DataFrame:
    """One row per drift run: date of the current window, drifted share and alert."""
    from brent_forecast.monitoring.drift import drift_dir

    path = drift_dir(settings) / "history.jsonl"
    if not path.is_file():
        return pd.DataFrame(columns=["current_end", "drifted_share", "alert"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    history = pd.DataFrame(rows)
    history["current_end"] = pd.to_datetime(history["current_end"])
    return history.drop_duplicates("current_end", keep="last").sort_values("current_end")


def drift_report(settings: Settings) -> Path | None:
    """Evidently HTML report of the last drift run, if any."""
    from brent_forecast.monitoring.drift import drift_dir

    path = drift_dir(settings) / "report.html"
    return path if path.is_file() else None
