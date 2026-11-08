"""Dashboard: Marketing Budget Optimizer + Big Data benchmark.

    streamlit run app.py
It only reads the small result files in outputs/ (created by `python -m adbudget.run_all`), so it opens instantly.
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from adbudget import budget_recommender as BR
from adbudget import settings as P

O = P.OUT_DIR
PLAT_COLORS = {"Facebook": "#3b5998", "Instagram": "#c13584", "Google": "#34a853", "TikTok": "#555555", "YouTube": "#e53935"}
ENGINE_COLORS = {"Pandas": "#3b6ea5", "MapReduce": "#d98324", "Spark": "#2e7d32"}
st.set_page_config(page_title="Ad Budget Optimizer", page_icon="📈", layout="wide")


@st.cache_data
def read_csv(name: str, mtime: float, **kw) -> pd.DataFrame:
    return pd.read_csv(O / name, **kw)


@st.cache_data
def read_json(name: str, mtime: float) -> dict:
    return json.loads((O / name).read_text(encoding="utf-8"))


def load_csv(name, **kw):
    f = O / name
    return read_csv(name, f.stat().st_mtime, **kw) if f.exists() else None


def load_json(name):
    f = O / name
    return read_json(name, f.stat().st_mtime) if f.exists() else None


def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    return fig


st.title("📈 Digital Advertising Budget Optimizer")
st.caption("Simulated advertising history for the sunscreen brand · KPIs · Pandas vs MapReduce vs Spark · budget recommendation engine")

rc, summary = load_json("response_curves.json"), load_json("analysis_summary.json")
plat, camp = load_csv("platform_kpis.csv"), load_csv("campaign_kpis.csv")
if rc is None or summary is None or plat is None:
    st.warning("No results yet. Run `python -m adbudget.run_all --quick` (about a minute) from the project folder, then reload this page.")
    st.stop()
curves = rc["curves"]
hist_share = dict(zip(plat["platform"], plat["spend_share"]))

tabs = st.tabs(["Budget recommender", "Campaign performance", "Big Data benchmark", "Can I trust it?"])

# ------------------------------------------------------------------ Budget recommender
with tabs[0]:
    st.subheader("Where should we spend the next budget?")
    c = st.columns(5)
    budget = c[0].number_input("Budget (Rs)", 100_000, 100_000_000, int(rc["budget"]), step=100_000)
    weeks = c[1].slider("Spent over (weeks)", 1, 12, int(rc["weeks"]))
    start = c[2].date_input("Plan starts", date.fromisoformat(rc["plan_start"]), help="Sunscreen sells best in summer, so the start date changes the forecast.")
    lo, hi = c[3].slider("Min / max share per platform", 0, 100, (int(P.MIN_SHARE * 100), int(P.MAX_SHARE * 100)), step=5, format="%d%%")
    cap = c[4].checkbox("Stay near historical range", True, help="Do not plan a platform above 125% of its highest weekly spend in history.")
    try:
        rec = BR.allocate(curves, budget, weeks, start, lo / 100, hi / 100, P.HISTORY_CAP if cap else None)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    rec["historical_share_pct"] = rec["platform"].map(lambda p: round(hist_share[p] * 100, 1))
    cmp_ = BR.compare_plans(curves, rec, hist_share, budget, weeks, start)

    k = st.columns(4)
    k[0].metric("Predicted revenue", f"Rs {rec['expected_revenue'].sum():,.0f}")
    k[1].metric("Predicted ROAS", f"{rec['expected_revenue'].sum() / budget:.2f}")
    k[2].metric("vs equal split", f"{cmp_.loc[1, 'recommended_is_better_by_pct']:+.1f}%", help="How much better the recommended split is, as predicted by the model")
    k[3].metric("vs last year's split", f"{cmp_.loc[2, 'recommended_is_better_by_pct']:+.1f}%")

    c1, c2 = st.columns([3, 2])
    long = rec.melt(id_vars="platform", value_vars=["historical_share_pct", "share_pct"], var_name="plan", value_name="share")
    long["plan"] = long["plan"].map({"historical_share_pct": "Same as last year", "share_pct": "Recommended"})
    fig = px.bar(long, x="platform", y="share", color="plan", barmode="group", text="share", title="Share of budget (%)",
                 color_discrete_map={"Same as last year": "#b0b6bf", "Recommended": "#2f5d8a"},
                 category_orders={"platform": list(rec["platform"])})
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    c1.plotly_chart(style(fig), width="stretch")
    c2.markdown("**Recommended allocation**")
    c2.dataframe(rec[["platform", "share_pct", "spend", "expected_revenue", "expected_roas"]].rename(columns={
        "share_pct": "share %", "spend": "spend (Rs)", "expected_revenue": "pred. revenue", "expected_roas": "pred. ROAS"}), hide_index=True)

    detail = rec[["platform", "weekly_spend", "marginal_roas", "curve_exponent_b", "limit", "within_history"]].rename(columns={
        "weekly_spend": "weekly spend (Rs)", "marginal_roas": "marginal ROAS", "curve_exponent_b": "curve b", "within_history": "inside history range"})
    with st.expander("Why this split? (marginal ROAS, limits, extrapolation)"):
        st.markdown("At the best split, the **marginal ROAS** (revenue from the next rupee) is equal on every platform, except platforms held at a limit. "
                    "A platform at its minimum with marginal ROAS below 1 is kept only because of the minimum-share rule.")
        st.dataframe(detail, hide_index=True)
        out_of_range = rec.loc[~rec["within_history"], "platform"].tolist()
        if out_of_range:
            st.warning("Planned outside the weekly spend range seen in history (extrapolation): " + ", ".join(out_of_range) + ". Move in steps and re-measure.")
    st.dataframe(cmp_, hide_index=True)
    st.download_button("Download recommendation (CSV)", rec.to_csv(index=False).encode("utf-8-sig"), "budget_recommendation.csv", "text/csv")
    unc = load_csv("recommendation_uncertainty.csv")
    if unc is not None:
        with st.expander("How much could the shares move? (bootstrap, default settings)"):
            f = go.Figure()
            for r in unc.itertuples():
                f.add_trace(go.Scatter(x=[r.p05, r.p95], y=[r.platform] * 2, mode="lines", line=dict(width=8, color=PLAT_COLORS[r.platform]), showlegend=False))
                f.add_trace(go.Scatter(x=[r.median], y=[r.platform], mode="markers", marker=dict(size=11, color="white", line=dict(width=2, color="black")), showlegend=False))
            f.update_layout(xaxis_title="recommended share (%): 5th to 95th percentile of 200 refits")
            st.plotly_chart(style(f, 300), width="stretch")

# ------------------------------------------------------------------ Campaign performance
with tabs[1]:
    st.subheader(f"History: {summary['rows']:,} simulated records, {summary['date_from']} to {summary['date_to']}")
    k = st.columns(6)
    for col, (lab, val) in zip(k, [("Spend", f"Rs {summary['spend'] / 1e6:.2f}M"), ("Revenue", f"Rs {summary['revenue'] / 1e6:.2f}M"),
                                   ("ROAS", f"{summary['roas']:.2f}"), ("CTR", f"{summary['ctr']:.2%}"),
                                   ("CPC", f"Rs {summary['cpc']:.1f}"), ("CPA", f"Rs {summary['cpa']:,.0f}")]):
        col.metric(lab, val)
    c1, c2 = st.columns(2)
    d = plat.sort_values("roas", ascending=False)
    c1.plotly_chart(style(px.bar(d, x="platform", y="roas", color="platform", color_discrete_map=PLAT_COLORS, title="ROAS by platform (break-even = 1)")
                          .add_hline(y=1, line_dash="dash")), width="stretch")
    c2.plotly_chart(style(px.bar(d, x="platform", y="cpa", color="platform", color_discrete_map=PLAT_COLORS, title="Cost per conversion (Rs)")), width="stretch")
    st.markdown("**Platforms**")
    show = d[["platform", "spend", "revenue", "ctr", "cpc", "cpa", "conversion_rate", "roas"]].copy()
    show["ctr"], show["conversion_rate"] = show["ctr"] * 100, show["conversion_rate"] * 100
    st.dataframe(show, hide_index=True, column_config={
        "spend": st.column_config.NumberColumn(format="%.0f"), "revenue": st.column_config.NumberColumn(format="%.0f"),
        "ctr": st.column_config.NumberColumn("CTR %", format="%.2f"), "conversion_rate": st.column_config.NumberColumn("conv. rate %", format="%.1f"),
        "cpc": st.column_config.NumberColumn(format="%.1f"), "cpa": st.column_config.NumberColumn(format="%.0f"), "roas": st.column_config.NumberColumn(format="%.2f")})
    if camp is not None:
        st.markdown("**Campaigns** (verdict by ROAS: ≥3 scale up, 2–3 maintain, 1–2 optimise, <1 cut / rework)")
        st.dataframe(camp[["campaign", "spend", "revenue", "roas", "cpa", "conversion_rate", "verdict"]], hide_index=True)
    month = load_csv("monthly_platform_kpis.csv")
    if month is not None:
        metric = st.selectbox("Monthly trend", ["revenue", "spend", "roas", "conversions"])
        st.plotly_chart(style(px.line(month, x="month", y=metric, color="platform", markers=True, color_discrete_map=PLAT_COLORS), 360), width="stretch")
    seg = load_csv("segment_kpis.csv")
    if seg is not None:
        pick = st.radio("Audience segment", ["device", "age_group", "gender", "city"], horizontal=True)
        s = seg[seg["segment"] == pick].sort_values("roas", ascending=False)
        st.plotly_chart(style(px.bar(s, x="value", y="roas", title=f"ROAS by {pick.replace('_', ' ')}", color_discrete_sequence=["#2f5d8a"]), 300), width="stretch")

# ------------------------------------------------------------------ Benchmark
with tabs[2]:
    bm, meta = load_csv("benchmark_results.csv"), load_json("benchmark_meta.json")
    if bm is None:
        st.info("No benchmark yet. Run `python -m adbudget.run_all` (or `python -m adbudget.benchmark`).")
    else:
        st.subheader("Pandas vs MapReduce vs Spark")
        if meta:
            st.caption(f"{meta['cpu_cores']} CPU cores · Python {meta['python']} · pandas {meta['pandas']}"
                       + (f" · PySpark {meta.get('pyspark', '')}" if meta.get("pyspark") else "")
                       + (f" · Spark start-up {meta['spark_startup_seconds']} s (excluded from timings)" if meta.get("spark_startup_seconds") else ""))
        names = {"ads": "Job A: advertising KPIs (sum per platform and campaign)", "nlp": "Job B: classify sunscreen comments (TF-IDF + Logistic Regression)"}
        for job in ("ads", "nlp"):
            g = bm[bm["job"] == job]
            if g.empty:
                continue
            st.markdown(f"**{names[job]}**")
            c1, c2 = st.columns([3, 2])
            fig = px.line(g.sort_values("rows"), x="rows", y="seconds", color="engine", markers=True, log_x=True, log_y=True,
                          color_discrete_map=ENGINE_COLORS, labels={"rows": "records", "seconds": "seconds (log)"})
            c1.plotly_chart(style(fig), width="stretch")
            piv = g.pivot(index="rows", columns="engine", values="seconds")[["Pandas", "MapReduce", "Spark"]].round(2)
            piv["all answers match"] = g.groupby("rows")["same_answer_as_pandas"].all().map({True: "yes", False: "NO"})
            c2.dataframe(piv)
        st.info("**Reading the chart:** Pandas has almost no overhead and wins while data fits in memory; Spark pays a fixed start-up/planning cost and "
                "is built for data or compute that outgrows one machine; record-by-record MapReduce in Python is slow for simple sums but shows the "
                "map → shuffle → reduce model. All engines share the same few cores here, so this does not show cluster scaling.")
    stab = load_csv("scale_stability.csv")
    if stab is not None:
        st.markdown("**Does the recommendation change with dataset size?**")
        st.dataframe(stab, hide_index=True)

# ------------------------------------------------------------------ Trust
with tabs[3]:
    weekly = load_csv("weekly_platform.csv", parse_dates=["week_start"])
    st.subheader("Response curves: revenue vs weekly spend")
    pick = st.selectbox("Platform", list(curves))
    cv = curves[pick]
    g = weekly[weekly["platform"] == pick]
    xs = np.linspace(g["spend"].min() * 0.8, g["spend"].max() * 1.3, 80)
    f = go.Figure()
    f.add_trace(go.Scatter(x=g["spend"], y=g["revenue"], mode="markers", name="weeks", marker=dict(color=PLAT_COLORS[pick])))
    f.add_trace(go.Scatter(x=xs, y=np.exp(cv["alpha"] + cv["b_used"] * np.log(xs) + cv["sigma2"] / 2), name="fitted curve (average season, campaigns share spend equally)", line=dict(color="black")))
    f.add_trace(go.Scatter(x=xs, y=xs, name="break-even (ROAS = 1)", line=dict(dash="dash", color="grey")))
    f.update_layout(xaxis_title="weekly spend (Rs)", yaxis_title="weekly revenue (Rs)")
    st.plotly_chart(style(f, 380), width="stretch")
    st.caption(f"Fitted exponent b = {cv['b']:.2f}" + (f", kept at {cv['b_used']:.2f} when optimising" if cv["b"] != cv["b_used"] else "") +
               f" (b < 1 = diminishing returns). In this simulation the true value is {P.PLATFORM_PARAMS[pick]['b']}. R² on log scale = {cv['r2_log']:.2f}.")
    h = rc["holdout"]
    st.markdown(f"**Held-out test:** curves fitted without the last {h['holdout_weeks']} weeks predicted those weeks' revenue "
                f"with a mean error of {h['mean_mape_pct']}% per platform-week and {h['total_revenue_error_pct']:+.1f}% on total revenue.")
    st.dataframe(pd.DataFrame(h["per_platform"]).T.rename(columns={"mape_pct": "mean abs. error %", "total_error_pct": "total error %"}))
    st.warning("All advertising data is simulated. Predicted uplifts compare plans inside the fitted model; validate with a controlled budget test before a full rollout.")
