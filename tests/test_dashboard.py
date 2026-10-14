"""Tests for the Streamlit demo: data loaders without Streamlit, the page with AppTest."""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from brent_forecast.config import Settings
from brent_forecast.dashboard import data
from brent_forecast.dashboard.data import ApiClient, ApiError

REPORT: dict[str, Any] = {
    "period": ["2024-01-01", "2026-03-11"],
    "n_days": 573,
    "alpha": 0.05,
    "reference_baseline": "Persistence",
    "conclusion": ["No model beats the baseline.", "Conclusion: no skill."],
    "candidates": {
        "Logistic Regression": {
            "kind": "model",
            "auc": {"estimate": 0.53, "low": 0.48, "high": 0.58},
            "brier": 0.2495,
        },
        "Persistence": {
            "kind": "baseline",
            "auc": {"estimate": 0.51, "low": 0.47, "high": 0.54},
            "brier": 0.49,
        },
    },
    "comparisons": {
        "Logistic Regression": {
            "delta_auc": {"estimate": 0.02},
            "p_holm": 1.0,
            "significant": False,
        }
    },
    "backtest": {
        "settings": {"threshold": 0.5, "allow_short": False, "cost_bps": 5.0},
        "strategies": {
            "Buy & hold": {
                "kind": "baseline",
                "sharpe": 1.06,
                "cagr": 0.33,
                "max_drawdown": -0.34,
                "hit_ratio": 0.48,
                "n_trades": 1,
            },
            "Logistic Regression": {
                "kind": "model",
                "sharpe": 1.16,
                "cagr": 0.25,
                "max_drawdown": -0.23,
                "hit_ratio": 0.52,
                "n_trades": 122,
            },
        },
    },
}
PREDICTION = {
    "as_of": "2026-03-12",
    "proba_up": 1.0,
    "direction": "up",
    "threshold": 0.5,
    "source": "snapshot",
}
INFO = {
    "alias": "champion",
    "registered_model": "brent-direction-classifier",
    "version": "1",
    "candidate": "persistence",
    "kind": "baseline",
    "promotion_reason": "No model beats the best baseline.",
}


def _metrics(settings: Settings, matches: bool | None) -> None:
    path = settings.paths.metrics_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": {"matches_recorded_checksums": matches}}))


def _artefacts(settings: Settings) -> None:
    _metrics(settings, None)
    settings.paths.report_data_file.write_text(json.dumps(REPORT))
    settings.paths.explain_dir.mkdir(parents=True)
    pd.DataFrame({"feature": ["vix", "mom_63"], "mean_abs_shap": [0.02, 0.07]}).to_csv(
        settings.paths.explain_dir / "shap_lightgbm.csv", index=False
    )
    _drift(settings)


DRIFT = {
    "source": "live",
    "reference": {"start": "2023-01-02", "end": "2025-12-31", "rows": 756},
    "current": {"start": "2026-01-02", "end": "2026-03-31", "rows": 60},
    "columns": [
        {
            "column": "vix",
            "method": "K-S p_value",
            "threshold": 0.05,
            "score": 0.001,
            "drifted": True,
        },
        {
            "column": "prediction",
            "method": "K-S p_value",
            "threshold": 0.05,
            "score": 0.4,
            "drifted": False,
        },
    ],
    "drifted_features": ["vix"],
    "drifted_share": 0.6,
    "prediction_drift": False,
    "alert": True,
    "reasons": ["ALERT: 60% of the features drifted (threshold 50%)"],
    "model": {},
    "generated_at": "2026-04-01T23:00:00+00:00",
}


def _drift(settings: Settings) -> None:
    folder = settings.paths.results_dir / "drift"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(DRIFT))
    (folder / "report.html").write_text("<html></html>")
    lines = [
        {"current_end": "2026-03-30", "drifted_share": 0.3, "alert": False},
        {"current_end": "2026-03-31", "drifted_share": 0.5, "alert": False},
        {"current_end": "2026-03-31", "drifted_share": 0.6, "alert": True},  # re-run, same day
    ]
    (folder / "history.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))


# ── provenance ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("matches", "kind"), [(True, "real"), (None, "synthetic"), (False, "mismatch")]
)
def test_provenance_follows_the_recorded_checksums(
    settings: Settings, matches: bool | None, kind: str
) -> None:
    _metrics(settings, matches)

    prov = data.provenance(settings)

    assert prov.kind == kind
    assert prov.is_real is (kind == "real")


def test_provenance_without_a_training_run(settings: Settings) -> None:
    assert data.provenance(settings).kind == "unknown"


# ── loaders ───────────────────────────────────────────────────


def test_price_history_prefers_live_data(settings: Settings, raw_data_dir: Path) -> None:
    snapshot = data.price_history(settings)
    settings.paths.live_dir.mkdir(parents=True)
    live_file = settings.paths.live_dir / settings.data.oil_filename
    shutil.copy(raw_data_dir / settings.data.oil_filename, live_file)
    frame = pd.read_csv(live_file)
    frame.tail(10).to_csv(live_file, index=False)

    live = data.price_history(settings)

    assert list(snapshot.columns) == ["brent_price", "wti_price"]
    assert snapshot.index.is_monotonic_increasing
    assert len(live) == 10


def test_price_history_without_data_is_empty(settings: Settings) -> None:
    assert data.price_history(settings).empty


def test_report_tables(settings: Settings) -> None:
    candidates = data.candidates_table(REPORT)
    backtest = data.backtest_table(REPORT)

    assert candidates["candidate"].tolist() == ["Logistic Regression", "Persistence"]
    assert candidates.loc[1, "p_holm"] is None or pd.isna(candidates.loc[1, "p_holm"])
    assert backtest["strategy"].tolist() == ["Logistic Regression", "Buy & hold"]
    assert data.load_report(settings) is None


def test_shap_importances_and_plots(settings: Settings) -> None:
    assert data.shap_importances(settings) == {}
    _artefacts(settings)

    importances = data.shap_importances(settings)

    assert list(importances) == ["lightgbm"]
    assert data.plot(settings, "missing.png") is None


# ── API client ────────────────────────────────────────────────


def test_api_client_against_the_real_api(settings: Settings, raw_data_dir: Path) -> None:
    """The client and the server agree on the contract (served in-process with uvicorn)."""
    import threading
    import time

    import uvicorn

    from brent_forecast.api import create_app
    from brent_forecast.api.service import ServedModel
    from brent_forecast.models.baselines import PersistenceClassifier

    model = PersistenceClassifier().fit(pd.DataFrame({"lag_ret_1": [0.1, -0.1]}), [0, 1])
    served = ServedModel("champion", model, None, {"version": "2", "candidate": "persistence"})
    config = uvicorn.Config(
        create_app(settings, loader=lambda s, a: served), host="127.0.0.1", port=0, log_config=None
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        client = ApiClient(f"http://127.0.0.1:{port}", timeout=10)
        assert client.predict_latest()["source"] == "snapshot"
        assert client.model_info()["version"] == "2"
        with pytest.raises(ApiError, match="404"):
            client._call("GET", "/missing")
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_api_client_reports_connection_errors(settings: Settings) -> None:
    client = ApiClient.from_settings(
        settings.model_copy(
            update={
                "dashboard": settings.dashboard.model_copy(
                    update={"api_url": "http://127.0.0.1:9/", "timeout_seconds": 1}
                )
            }
        )
    )

    assert client.base_url == "http://127.0.0.1:9"
    with pytest.raises(ApiError, match="failed"):
        client.predict_latest()


def test_drift_loaders(settings: Settings) -> None:
    assert data.drift_summary(settings) is None
    assert data.drift_history(settings).empty
    assert data.drift_report(settings) is None
    _drift(settings)

    assert data.drift_summary(settings)["alert"] is True  # type: ignore[index]
    history = data.drift_history(settings)
    assert history["drifted_share"].tolist() == [0.3, 0.6]  # last run of each day
    assert data.drift_report(settings) is not None


# ── page ──────────────────────────────────────────────────────


def _page(settings):  # unannotated: AppTest runs a copy of this source without the imports
    from brent_forecast.dashboard.data import ApiClient
    from brent_forecast.dashboard.page import render

    render(settings, ApiClient("http://api.invalid", 1))


def _run(settings: Settings) -> AppTest:
    app = AppTest.from_function(_page, args=(settings,), default_timeout=60)
    return app.run()


def test_the_page_renders_every_section(
    settings: Settings, raw_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artefacts(settings)
    monkeypatch.setattr(ApiClient, "predict_latest", lambda self: PREDICTION)
    monkeypatch.setattr(ApiClient, "model_info", lambda self: INFO)

    app = _run(settings)

    assert not app.exception
    assert "SYNTHETIC data" in app.warning[0].value
    assert [m.value for m in app.metric] == [
        *("▲ up", "100.0%", "2026-03-12", "persistence (v1)"),  # prediction
        *("60%", "no", "2026-01-02 → 2026-03-31"),  # monitoring
    ]
    assert "naive baseline" in app.info[0].value
    assert [t.label for t in app.tabs] == [
        "Prices",
        "Statistical report",
        "Backtest",
        "Explanations",
        "Monitoring",
    ]
    assert len(app.dataframe) == 3  # candidates, backtest and drift tests
    assert "60% of the features drifted" in app.error[0].value
    assert app.selectbox[0].value == "lightgbm"
    assert any("not investment advice" in c.value for c in app.caption)


def test_the_page_degrades_without_artefacts_or_api(settings: Settings) -> None:
    app = _run(settings)

    assert not app.exception
    assert "No training run yet" in app.warning[0].value
    assert "inference API is not available" in app.error[0].value
    infos = " ".join(i.value for i in app.info)
    for command in ("brent data download", "brent report", "brent explain", "brent monitor drift"):
        assert command in infos


def test_real_and_mismatched_data_banners(settings: Settings) -> None:
    _metrics(settings, True)
    assert "Real data" in _run(settings).success[0].value

    _metrics(settings, False)
    assert "do NOT match" in _run(settings).error[0].value


def test_the_entry_point_loads_the_default_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import CONFIG_PATH, REPO_ROOT

    monkeypatch.chdir(REPO_ROOT)  # configs/default.yaml is resolved against the working directory
    monkeypatch.setenv("BRENT_PATHS__RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BRENT_PATHS__DATA_DIR", str(tmp_path / "data" / "raw"))
    monkeypatch.setenv("BRENT_DASHBOARD__API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BRENT_DASHBOARD__TIMEOUT_SECONDS", "1")
    script = REPO_ROOT / "src" / "brent_forecast" / "dashboard" / "app.py"
    assert CONFIG_PATH.is_file()

    app = AppTest.from_file(str(script), default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "Brent next-day direction"
    assert "127.0.0.1:9" in app.error[0].value
