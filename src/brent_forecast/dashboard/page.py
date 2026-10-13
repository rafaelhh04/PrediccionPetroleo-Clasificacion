"""Layout of the Streamlit demo; all the data comes from :mod:`.data`."""

import pandas as pd
import streamlit as st

from brent_forecast.config import Settings
from brent_forecast.dashboard.data import (
    ApiClient,
    ApiError,
    Provenance,
    backtest_table,
    candidates_table,
    load_report,
    plot,
    price_history,
    provenance,
    shap_importances,
)
from brent_forecast.models.registry import MODEL_NAMES

DISCLAIMER = (
    "Educational project, **not investment advice**. The statistical evaluation found no "
    "next-day directional skill: the served model is whatever the promotion rule allows, "
    "possibly a naive baseline."
)
PERIODS = {"1 year": 1, "3 years": 3, "5 years": 5, "All": None}


def _banner(prov: Provenance) -> None:
    if prov.is_real:
        st.success(prov.message, icon="✅")
    elif prov.kind == "mismatch":
        st.error(prov.message, icon="🚫")
    else:
        st.warning(prov.message, icon="⚠️")


def _prediction(client: ApiClient) -> None:
    st.subheader("Prediction for the next trading day")
    try:
        prediction = client.predict_latest()
        info = client.model_info()
    except ApiError as exc:
        st.error(f"The inference API is not available ({client.base_url}): {exc}")
        st.caption("Start it with `brent serve` (or `docker compose up`).")
        return
    up = prediction["direction"] == "up"
    cols = st.columns(4)
    cols[0].metric("Direction", "▲ up" if up else "▼ down/flat")
    cols[1].metric("P(up)", f"{prediction['proba_up']:.1%}")
    cols[2].metric("Last observed day", prediction["as_of"] or "n/a")
    cols[3].metric("Model", f"{info['candidate']} (v{info['version']})")
    st.caption(
        f"Alias `{info['alias']}` of `{info['registered_model']}` · features from "
        f"`{prediction['source']}` · threshold {prediction['threshold']}"
    )
    if info.get("kind") == "baseline":
        st.info(
            f"The served model is a **naive baseline**. {info.get('promotion_reason') or ''}",
            icon="ℹ️",
        )


def _prices(settings: Settings) -> None:
    prices = price_history(settings)
    if prices.empty:
        st.info("No price data: run `brent data download`.")
        return
    period = st.radio("Period", list(PERIODS), index=1, horizontal=True)
    years = PERIODS[period]
    if years is not None:
        prices = prices[prices.index >= prices.index.max() - pd.DateOffset(years=years)]
    st.line_chart(prices, y=["brent_price", "wti_price"], y_label="USD per barrel")
    st.caption(f"{len(prices)} trading days up to {prices.index.max().date()}.")


def _report(settings: Settings) -> None:
    report = load_report(settings)
    if report is None:
        st.info("No statistical report yet: run `brent report`.")
        return
    start, end = report["period"]
    st.markdown(
        f"Out of sample **{start} → {end}** ({report['n_days']} days, walk-forward), reference "
        f"baseline **{report['reference_baseline']}**, Holm-adjusted DeLong at "
        f"α = {report['alpha']}."
    )
    for line in report["conclusion"]:
        st.markdown(f"- {line}")
    st.dataframe(
        candidates_table(report),
        hide_index=True,
        column_config={
            "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
            "auc_low": st.column_config.NumberColumn("CI low", format="%.3f"),
            "auc_high": st.column_config.NumberColumn("CI high", format="%.3f"),
            "delta_auc_vs_baseline": st.column_config.NumberColumn("ΔAUC", format="%+.3f"),
            "p_holm": st.column_config.NumberColumn("p (Holm)", format="%.3f"),
            "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
        },
    )
    figures = [
        p
        for p in (
            plot(settings, "roc_oos_comparison.png"),
            plot(settings, "reliability_diagram.png"),
        )
        if p
    ]
    for column, figure in zip(st.columns(max(len(figures), 1)), figures, strict=False):
        column.image(str(figure))


def _backtest(settings: Settings) -> None:
    report = load_report(settings)
    if report is None:
        st.info("No backtest yet: run `brent report`.")
        return
    cfg = report["backtest"]["settings"]
    st.markdown(
        f"Long when P(up) ≥ {cfg['threshold']}, {'short' if cfg['allow_short'] else 'flat'} "
        f"otherwise, {cfg['cost_bps']} bps per unit of turnover."
    )
    st.dataframe(
        backtest_table(report),
        hide_index=True,
        column_config={
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "cagr": st.column_config.NumberColumn("CAGR", format="percent"),
            "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
            "hit_ratio": st.column_config.NumberColumn("Hit ratio", format="percent"),
            "n_trades": st.column_config.NumberColumn("Trades", format="%d"),
        },
    )
    equity = plot(settings, "equity_curves.png")
    if equity:
        st.image(str(equity))


def _explanations(settings: Settings) -> None:
    importances = shap_importances(settings)
    if not importances:
        st.info("No explanations yet: run `brent explain`.")
        return
    key = st.selectbox("Model", list(importances), format_func=lambda k: MODEL_NAMES.get(k, k))
    top = importances[key].head(15)
    st.bar_chart(
        top,
        x="feature",
        y="mean_abs_shap",
        horizontal=True,
        sort="-mean_abs_shap",
        x_label="",
        y_label="mean |SHAP| (top 15 features)",
    )
    cols = st.columns(2)
    for column, kind, caption in (
        (cols[0], "beeswarm", "Global: SHAP values of every out-of-sample day"),
        (cols[1], "waterfall", "Local: the most recent explained day"),
    ):
        figure = plot(settings, f"shap_{kind}_{key}.png")
        if figure:
            column.image(str(figure), caption=caption)


def render(settings: Settings, client: ApiClient) -> None:
    """Draw the whole page."""
    st.set_page_config(page_title="Brent direction demo", page_icon="🛢️", layout="wide")
    st.title("Brent next-day direction")
    _banner(provenance(settings))
    _prediction(client)
    tabs = st.tabs(["Prices", "Statistical report", "Backtest", "Explanations"])
    with tabs[0]:
        _prices(settings)
    with tabs[1]:
        _report(settings)
    with tabs[2]:
        _backtest(settings)
    with tabs[3]:
        _explanations(settings)
    st.divider()
    st.caption(DISCLAIMER)
