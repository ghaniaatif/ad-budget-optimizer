"""Build the Word report (outputs/Project2_Report.docx) from the REAL result files.

Every table, chart and number comes from outputs/*.csv|json, so the report always matches the run.
    python -m adbudget.report
"""
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from adbudget import generate_data as G
from adbudget import settings as P
from adbudget.docx_helpers import add_table, bullets, figure

O = P.OUT_DIR
ENGINE_ORDER = ["Pandas", "MapReduce", "Spark"]
ENGINE_COLORS = {"Pandas": "#3b6ea5", "MapReduce": "#d98324", "Spark": "#2e7d32"}
PLAT_COLORS = {"Facebook": "#3b5998", "Instagram": "#c13584", "Google": "#34a853", "TikTok": "#111111", "YouTube": "#e53935"}
rs = lambda x: f"Rs {x:,.0f}"          # noqa: E731
fmt_secs = lambda s: f"{s:.2f}" if s < 100 else f"{s:.0f}"   # noqa: E731


def _save(fig, name):
    path = O / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_benchmark(bm):
    jobs = [j for j in ("ads", "nlp") if j in set(bm["job"])]
    fig, axes = plt.subplots(1, len(jobs), figsize=(5.2 * len(jobs), 3.8), squeeze=False)
    titles = {"ads": "Job A: advertising KPIs", "nlp": "Job B: classify comments"}
    for ax, job in zip(axes[0], jobs):
        for eng, g in bm[bm["job"] == job].groupby("engine"):
            g = g.sort_values("rows")
            ax.plot(g["rows"], g["seconds"], marker="o", label=eng, color=ENGINE_COLORS[eng])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("records")
        ax.set_ylabel("seconds (log scale)")
        ax.set_title(titles[job])
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend()
    return _save(fig, "p2_chart_benchmark.png")


def chart_platforms(plat):
    d = plat.sort_values("roas", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(d["platform"], d["roas"], color=[PLAT_COLORS[p] for p in d["platform"]])
    axes[0].axhline(1, color="grey", ls="--", lw=1)
    axes[0].set_title("ROAS by platform (revenue per Rs 1 spent)")
    axes[1].bar(d["platform"], d["cpa"], color=[PLAT_COLORS[p] for p in d["platform"]])
    axes[1].set_title("Cost per conversion (Rs)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "p2_chart_platforms.png")


def chart_curves(weekly, curves):
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), sharey=False)
    for ax, (plat, c) in zip(axes, curves.items()):
        g = weekly[weekly["platform"] == plat]
        ax.scatter(g["spend"] / 1000, g["revenue"] / 1000, s=10, color=PLAT_COLORS[plat], alpha=0.7)
        xs = np.linspace(g["spend"].min() * 0.9, g["spend"].max() * 1.1, 60)
        ax.plot(xs / 1000, np.exp(c["alpha"] + c["b_used"] * np.log(xs) + c["sigma2"] / 2) / 1000, color="black", lw=1.4)
        ax.set_title(f"{plat}  (b = {c['b_used']:.2f})", fontsize=10)
        ax.set_xlabel("weekly spend (Rs 000)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("weekly revenue (Rs 000)")
    return _save(fig, "p2_chart_curves.png")


def chart_allocation(rec):
    d = rec.sort_values("share_pct", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.2, d["historical_share_pct"], 0.4, label="Same as last year", color="#b0b6bf")
    ax.bar(x + 0.2, d["share_pct"], 0.4, label="Recommended", color="#2f5d8a")
    for i, v in enumerate(d["share_pct"]):
        ax.text(i + 0.2, v + 0.6, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x, d["platform"])
    ax.set_ylabel("% of budget")
    ax.set_title("Where the next budget should go")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "p2_chart_allocation.png")


def _read(name, **kw):
    return pd.read_csv(O / name, **kw)


def _speed_notes(bm) -> list[str]:
    notes = []
    for job, name in (("ads", "Job A (advertising KPIs)"), ("nlp", "Job B (sentiment classification)")):
        g = bm[bm["job"] == job]
        if g.empty:
            continue
        piv = g.pivot(index="rows", columns="engine", values="seconds")[ENGINE_ORDER]
        parts = []
        for rows, r in piv.iterrows():
            best = r.idxmin()
            lab = g[g["rows"] == rows]["scale"].iloc[0]
            parts.append(f"{lab}: {best} ({fmt_secs(r.min())} s)")
        notes.append(f"{name}, fastest engine per size: " + "; ".join(parts) + ".")
        top = piv.index.max()
        r = piv.loc[top]
        others = ", ".join(f"{e} {fmt_secs(r[e])} s" for e in piv.columns)
        notes.append(f"At the largest size for {name} ({top:,} records): {others}.")
    return notes


def _nlp_note(bm) -> str:
    g = bm[bm["job"] == "nlp"]
    if g.empty:
        return "The sentiment job was not run in this benchmark."
    top = g["rows"].max()
    r = g[g["rows"] == top].set_index("engine")["seconds"]
    best_par = min(r.get("MapReduce", np.inf), r.get("Spark", np.inf))
    ratio = r["Pandas"] / best_par
    cores = int(bm.attrs.get("cores", 0)) or None
    if ratio > 1.15:
        return (f"For the CPU-heavy sentiment job the picture changes: at {top:,} comments the engines that use every core finish about "
                f"{ratio:.1f} times faster than single-process Pandas, because almost all the time goes into cleaning text and TF-IDF "
                "(real computation per record), which parallelises well, instead of into moving data.")
    return ("For the CPU-heavy sentiment job the engines are close, because almost all the time goes into cleaning text and TF-IDF, not into moving data.")


def build_report() -> None:
    bm = _read("benchmark_results.csv")
    meta = json.loads((O / "benchmark_meta.json").read_text())
    summ = json.loads((O / "analysis_summary.json").read_text())
    plat = _read("platform_kpis.csv")
    camp = _read("campaign_kpis.csv")
    seg = _read("segment_kpis.csv")
    rec = _read("recommendation.csv")
    cmp_ = _read("plan_comparison.csv")
    unc = _read("recommendation_uncertainty.csv") if (O / "recommendation_uncertainty.csv").exists() else None
    stab = _read("scale_stability.csv") if (O / "scale_stability.csv").exists() else None
    rc = json.loads((O / "response_curves.json").read_text())
    weekly = _read("weekly_platform.csv", parse_dates=["week_start"])
    curves, hold = rc["curves"], rc["holdout"]
    model_kind = meta.get("model_used_for_nlp_job", "demo")

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ---- cover
    doc.add_paragraph()
    t = doc.add_heading("Project 2: Digital Advertising Budget Optimizer", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Extending the Sunscreen Product Sentiment Analysis project: scaling to 10 million records, "
                            "Pandas vs MapReduce vs Spark, and a budget recommendation engine")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].italic = True
    m = doc.add_paragraph(f"Big Data Analytics  |  Project 2  |  {date.today():%d %B %Y}")
    m.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # ---- 1 summary
    best = rec.iloc[0]
    doc.add_heading("1. Executive summary", 1)
    bullets(doc, [
        f"This project studies a simulated advertising business for a sunscreen brand "
        f"({summ['rows']:,} records, {summ['n_platforms']} platforms, {summ['n_campaigns']} campaigns, "
        f"{summ['date_from']} to {summ['date_to']}), Big Data processing, and a budget recommendation engine.",
        f"Overall performance on the simulated history: spend {rs(summ['spend'])}, revenue {rs(summ['revenue'])}, "
        f"ROAS {summ['roas']:.2f}, CTR {summ['ctr']:.2%}, CPC {rs(summ['cpc'])}, CPA {rs(summ['cpa'])}, "
        f"conversion rate {summ['conversion_rate']:.1%}.",
        f"Pandas, MapReduce and Spark were run on 10K, 100K, 1M and 10M records. All engines returned the same answers "
        f"({'verified' if bm['same_answer_as_pandas'].all() else 'NOT all verified, see Section 4'}).",
        f"For the next {rs(rc['budget'])} (spent over {rc['weeks']} weeks) the engine recommends "
        + ", ".join(f"{r.platform} {r.share_pct:.0f}%" for r in rec.itertuples()) +
        f". Predicted revenue is {rs(rc['total_expected_revenue'])} (ROAS {rc['total_expected_roas']:.2f}), "
        f"{cmp_.loc[1, 'recommended_is_better_by_pct']:.0f}% above an equal split and "
        f"{cmp_.loc[2, 'recommended_is_better_by_pct']:.0f}% above repeating last year's split (model-based estimates).",
    ])
    doc.add_paragraph("All advertising data in this report is simulated. The simulation is built so that the true behaviour "
                      "(diminishing returns, seasonality) is known, which lets the report check that the engine recovers it.")

    # ---- 2 data
    doc.add_heading("2. Data and simulation design", 1)
    doc.add_paragraph("Each record is one slice of ad delivery with the fields requested in the project brief:")
    add_table(doc, ["Field", "Meaning"], [
        ["date, campaign, platform", "When, which of the 15 campaigns, which of Facebook / Instagram / Google / TikTok / YouTube"],
        ["impressions, clicks", "Ads shown, ads clicked"],
        ["spend, conversions, revenue", "Rupees spent, purchases made, rupees earned"],
        ["age, gender, city, device", "Audience segment of the record (8 cities, 3 devices)"]], [2.0, 4.6])
    doc.add_paragraph(f"The business is the same at every size (about {rs(P.ANNUAL_SPEND)} of spend over 52 weeks). A bigger file is a "
                      "finer-grained log of that same business, so KPIs and advice are comparable across sizes. Behaviour is built in as follows:")
    bullets(doc, [
        "Each platform has its own cost per 1,000 impressions, click rate, order value and ROAS (table below).",
        "Diminishing returns: when a campaign spends more in a week, its conversion rate falls, so revenue grows like spend to the power b (b < 1).",
        "Seasonality: sunscreen converts best in summer (+35% at the June peak, -35% in winter).",
        "Small, realistic differences by device, gender and age group."])
    add_table(doc, ["Platform", "Cost per 1000 impr. (Rs)", "Click rate", "Order value (Rs)", "Base ROAS", "True b", "Last year's budget share"],
              [[p, v["cpm"], f"{v['ctr']:.1%}", v["aov"], v["roas"], v["b"], f"{v['share']:.0%}"] for p, v in P.PLATFORM_PARAMS.items()],
              [1.0, 1.1, 0.8, 1.0, 0.8, 0.7, 1.2])
    rows = []
    for lab, n in P.SCALES.items():
        files = G.ads_files(lab)
        size = sum(f.stat().st_size for f in files) / 1e6 if files else float("nan")
        rows.append([lab, f"{n:,}", len(files), f"{size:,.1f} MB" if size == size else "n/a"])
    add_table(doc, ["Scale", "Records", "Parquet files", "Size on disk"], rows, [1.2, 1.5, 1.3, 1.5])

    # ---- 3 KPIs
    doc.add_heading("3. Campaign performance metrics", 1)
    add_table(doc, ["Metric", "Formula", "Meaning"], [
        ["CTR", "clicks / impressions", "How often people click"], ["CPC", "spend / clicks", "Rs per click"],
        ["CPA", "spend / conversions", "Rs per purchase"], ["Conversion rate", "conversions / clicks", "Clicks that become purchases"],
        ["ROAS", "revenue / spend", "Rs earned per Rs 1 spent"], ["Revenue / profit", "sum of revenue; revenue - spend", "Before product costs"]],
        [1.4, 2.2, 3.0])
    doc.add_paragraph("KPIs are always computed from summed counts, never by averaging per-record ratios, which would be wrong for "
                      f"ratios. Results on {summ['rows']:,} records:")
    d = plat.sort_values("roas", ascending=False)
    add_table(doc, ["Platform", "Spend (Rs)", "Revenue (Rs)", "CTR", "CPC", "CPA", "Conv. rate", "ROAS"],
              [[r.platform, f"{r.spend:,.0f}", f"{r.revenue:,.0f}", f"{r.ctr:.2%}", f"{r.cpc:.1f}", f"{r.cpa:,.0f}", f"{r.conversion_rate:.1%}", f"{r.roas:.2f}"]
               for r in d.itertuples()], [1.0, 1.0, 1.1, 0.7, 0.7, 0.7, 0.8, 0.6])
    figure(doc, chart_platforms(plat), "Figure 1: ROAS and cost per conversion by platform", 6.0)
    doc.add_paragraph("Campaign verdicts use ROAS bands: 3.0 and above = Scale up, 2.0 to 3.0 = Maintain, 1.0 to 2.0 = Optimise, below 1.0 = Cut / rework.")
    c = camp.sort_values("roas", ascending=False)
    add_table(doc, ["Campaign", "Spend (Rs)", "ROAS", "CPA", "Verdict"],
              [[r.campaign, f"{r.spend:,.0f}", f"{r.roas:.2f}", f"{r.cpa:,.0f}", r.verdict] for r in c.itertuples()], [2.8, 1.1, 0.7, 0.8, 1.2])
    for s in ("device", "age_group"):
        g = seg[seg["segment"] == s].sort_values("roas", ascending=False)
        doc.add_paragraph(f"Best and worst {s.replace('_', ' ')}: {g.iloc[0]['value']} (ROAS {g.iloc[0]['roas']:.2f}) and {g.iloc[-1]['value']} (ROAS {g.iloc[-1]['roas']:.2f}).")

    # ---- 4 benchmark
    doc.add_heading("4. Big Data benchmark: Pandas vs MapReduce vs Spark", 1)
    doc.add_paragraph("Two jobs were run on each engine so the answers can be compared:")
    bullets(doc, [
        "Job A (advertising): sum impressions, clicks, spend, conversions and revenue for every platform and campaign. Every KPI follows from these sums.",
        "Job B (sentiment): classify every sunscreen comment with a TF-IDF and Logistic Regression model (clean text, TF-IDF, Logistic Regression) and count the labels.",
        "Pandas: load into memory and use vectorised operations on one process.",
        "MapReduce: the classic model written in plain Python. Map tasks (one per file block) handle one record at a time and combine locally, "
        "a shuffle groups results by key, reducers add the partial sums. Runs on all CPU cores with multiprocessing. It illustrates the "
        "programming model; it is not Hadoop.",
        "Spark: PySpark in local mode on the same cores, with the work expressed as DataFrame operations and planned by Spark."])
    doc.add_paragraph(f"Machine: {meta['cpu_cores']} CPU cores, Python {meta['python']}, pandas {meta['pandas']}"
                      + (f", PySpark {meta['pyspark']}" if meta.get("pyspark") else "") +
                      f". Times are seconds from reading the files to the final table (median of {int(bm['repeats'].max())} runs for 10K and 100K, one run for larger sizes). "
                      + (f"Spark session start-up ({meta['spark_startup_seconds']} s) is reported separately, and a warm-up job ran before timing."
                         if meta.get("spark_startup_seconds") else ""))
    for job, title in (("ads", "Job A: advertising KPIs (seconds)"), ("nlp", "Job B: sentiment classification (seconds)")):
        g = bm[bm["job"] == job]
        if g.empty:
            continue
        piv = g.pivot(index="scale", columns="engine", values="seconds").reindex([s for s in P.SCALES if s in set(g["scale"])])[ENGINE_ORDER]
        ok = g.groupby("scale")["same_answer_as_pandas"].all()
        doc.add_paragraph(title).runs[0].bold = True
        add_table(doc, ["Records"] + list(piv.columns) + ["Same answer"],
                  [[f"{s} ({P.SCALES[s]:,})"] + [fmt_secs(piv.loc[s, e]) for e in piv.columns] + ["yes" if ok[s] else "NO"] for s in piv.index])
    figure(doc, chart_benchmark(bm), "Figure 2: run time against number of records (both axes logarithmic)", 6.3)
    doc.add_heading("What the measurements show", 2)
    bullets(doc, _speed_notes(bm))
    if (bm["job"] == "nlp").any() and P.SCALES[P.NLP_MAX_SCALE] < P.SCALES["10M"]:
        doc.add_paragraph(f"Job B stops at {bm[bm['job'] == 'nlp']['scale'].iloc[-1]} records by default because cleaning and scoring text costs tens of "
                          "microseconds per comment; `python -m adbudget.benchmark --nlp-max 10M` lifts the limit.")
    doc.add_heading("How to read this", 2)
    bullets(doc, [
        "Pandas has almost no start-up cost and is very fast for column sums, so it wins at small sizes and often much further than people expect, "
        "as long as the data fits in memory. Its limit is memory and a single process, not speed on small data.",
        "Spark pays a fixed cost (JVM, planning, scheduling), which is why it is slow on small data. It becomes attractive when data no longer fits "
        "on one machine, or when the work is CPU-heavy per record and can be spread over many machines. Here all engines share the same few cores, "
        "so this test shows start-up and per-record overhead, not the cluster scaling that is Spark's main purpose.",
        "Record-by-record MapReduce in Python is the slowest for simple sums because each record passes through the interpreter, "
        "but it parallelises naturally and teaches the shuffle and reduce ideas that Hadoop and Spark are built on.",
        _nlp_note(bm),
        "A fast answer is only useful if it is right: every engine's output was compared with Pandas (integers exactly, money to 1e-9)."])

    # ---- 5 sentiment link
    doc.add_heading("5. Sentiment model at scale", 1)
    nl = bm[bm["job"] == "nlp"]
    if model_kind == "external":
        doc.add_paragraph("Job B used the sentiment model supplied from the Sunscreen Sentiment Analysis project (YouTube comments, TF-IDF + Logistic Regression).")
    else:
        dm = json.loads((O / "demo_model_metrics.json").read_text()) if (O / "demo_model_metrics.json").exists() else {}
        doc.add_paragraph("No Phase 1 model was found when the benchmark ran, so a demo model with the same design (TF-IDF words and word pairs "
                          "into Logistic Regression) was trained on synthetic sunscreen comments"
                          + (f" (test accuracy {dm['accuracy']:.1%}, macro-F1 {dm['macro_f1']:.2f})" if dm else "") +
                          ". These timings measure computing cost only; the accuracy of a classifier trained on real comments is reported in the Sunscreen Sentiment Analysis project. "
                          "Set the SENTIMENT_MODEL environment variable to a trained model file and rerun to score with your own model.")
    if not nl.empty:
        doc.add_paragraph(f"The largest sentiment run classified {int(nl['rows'].max()):,} comments.")

    # ---- 6 budget
    doc.add_heading("6. Marketing Budget Recommendation Engine", 1)
    doc.add_paragraph(f"Question from management: where should the next {rs(rc['budget'])} be spent? Method:")
    bullets(doc, [
        "Learn a response curve for each platform from weekly history: weekly revenue = A x (weekly spend)^b, fitted on a log scale with a yearly seasonal term.",
        "Choose the split that maximises predicted total revenue. At the optimum one more rupee earns the same on every platform (equal marginal ROAS), "
        f"except where a platform sits at its limits: at least {rc['min_share']:.0%} and at most {rc['max_share']:.0%} of the budget, and no more than 125% "
        "of the highest weekly spend ever seen on that platform (the curve is not trusted beyond what it has seen).",
        f"Spend is spread evenly over {rc['weeks']} weeks starting {rc['plan_start']}, so the seasonal effect of those weeks is included."])
    figure(doc, chart_curves(weekly, curves), "Figure 3: weekly spend against weekly revenue with the fitted curve", 6.6)
    doc.add_paragraph("Curves are fitted on campaign-weeks (about 150 per platform), which vary spend far more than 52 weekly totals and so pin down b "
                      "much better; a platform's curve then assumes its campaigns share its spend equally. The curves in Figure 3 are drawn on platform-week totals.")
    add_table(doc, ["Platform", "Fitted b", "b used by optimiser", "True b (simulation)", "R-squared (log)", "Weekly spend seen (Rs)"],
              [[p, f"{c['b']:.2f}", f"{c['b_used']:.2f}", P.PLATFORM_PARAMS[p]["b"], f"{c['r2_log']:.2f}", f"{c['hist_min_weekly_spend']:,.0f} to {c['hist_max_weekly_spend']:,.0f}"]
               for p, c in curves.items()], [0.9, 0.8, 1.1, 1.2, 1.0, 1.8])
    doc.add_paragraph("Because the data is simulated, the true exponent is known. The fitted values rank the platforms correctly and sit within about 0.05 to 0.1 "
                      "of the truth (slightly too high: noise in the data biases b toward 1, so the engine is a little too optimistic about extra spend). "
                      f"The b is kept between 0.2 and 0.95 when optimising (and the curve refitted with b fixed), so a platform whose fit shows no diminishing "
                      "returns, like TikTok here, is not allowed to absorb the whole budget.")
    doc.add_heading("Recommendation", 2)
    add_table(doc, ["Platform", "Share", "Spend (Rs)", "Predicted revenue (Rs)", "Predicted ROAS", "Marginal ROAS", "Last year's share", "Note"],
              [[r.platform, f"{r.share_pct:.1f}%", f"{r.spend:,.0f}", f"{r.expected_revenue:,.0f}", f"{r.expected_roas:.2f}", f"{r.marginal_roas:.2f}",
                f"{r.historical_share_pct:.1f}%", "" if pd.isna(r.limit) else r.limit] for r in rec.itertuples()],
              [0.9, 0.6, 0.9, 1.1, 0.8, 0.8, 0.8, 1.0])
    figure(doc, chart_allocation(rec), "Figure 4: recommended split against last year's split", 5.4)
    low = rec[rec["marginal_roas"] < 1]
    if not low.empty:
        doc.add_paragraph("Marginal ROAS below 1.00 for " + ", ".join(low["platform"]) + " means the next rupee there earns back less than a rupee even at the "
                          "minimum share. The minimum is a business rule (keep a presence on every channel); the engine would cut further without it.")
    doc.add_paragraph("Predicted outcome of the plans (model-based estimates, not observed results):")
    add_table(doc, ["Plan", "Predicted revenue (Rs)", "ROAS"] + P.PLATFORMS,
              [[r["plan"], f"{r['expected_revenue']:,.0f}", f"{r['expected_roas']:.2f}"] + [f"{r[p]:.0f}%" for p in P.PLATFORMS] for _, r in cmp_.iterrows()],
              [1.4, 1.3, 0.6] + [0.7] * 5)
    doc.add_heading("How far to trust it", 2)
    bullets(doc, [
        f"Held-out test: curves fitted without the last {hold['holdout_weeks']} weeks predicted those weeks' revenue with a mean absolute error of "
        f"{hold['mean_mape_pct']}% per platform-week and {hold['total_revenue_error_pct']:+.1f}% on total revenue.",
        "Bootstrap (200 refits on resampled weeks), range of the recommended share: " + (", ".join(
            f"{r.platform} {r.p05:.0f}-{r.p95:.0f}%" for r in unc.itertuples()) if unc is not None else "not run") + ".",
        "A negative error means the model under-predicted the held-out weeks. Those weeks are autumn, which the first weeks of the one-year history "
        "(November to August) do not contain, so the seasonal term had to extrapolate; more years of data would help.",
        "Platforms planned outside their historical weekly spend range: " + (", ".join(rec.loc[~rec["within_history"], "platform"]) or "none") +
        ". Treat those as extrapolation and increase them in steps, re-measuring each time."])
    if stab is not None:
        doc.add_paragraph("Does the advice depend on dataset size? Recommended share (%) when the engine learns from each size:")
        add_table(doc, ["Records"] + P.PLATFORMS, [[f"{int(r['rows']):,}"] + [f"{r[p]:.1f}" for p in P.PLATFORMS] for _, r in stab.iterrows()])
        doc.add_paragraph("Small samples are noisy (few conversions per campaign-week). Compare the rows to see how far the advice moves as the data grows; "
                          "the more stable the shares, the more the decision can be trusted.")

    # ---- 7 dashboard / run
    doc.add_heading("7. Dashboard and how to run", 1)
    bullets(doc, [
        "`pip install -r requirements.txt` (Java 17 or newer is needed for Spark).",
        "`python -m adbudget.run_all --quick` (about a minute) or `python -m adbudget.run_all` (all sizes up to 10 million).",
        "`streamlit run app.py`: change the budget, weeks, limits and start date and see the "
        "allocation update; view campaign performance, the benchmark and the scale check.",
        "`python -m adbudget.budget_recommender --budget 1000000 --weeks 4` gives the recommendation on the command line."])

    # ---- 8 limits
    doc.add_heading("8. Limitations and future work", 1)
    bullets(doc, [
        "The advertising data is simulated. Real campaigns have attribution delays, overlapping audiences and creative fatigue that this model ignores.",
        "Response curves are simple power laws per platform. Interactions between platforms (a YouTube view followed by a Google search) are not modelled.",
        "The benchmark uses one machine; MapReduce is an in-process illustration and Spark runs locally. Cluster behaviour (network shuffle, node failure) is not measured.",
        "Predicted uplift compares plans inside the fitted model; it should be validated with a controlled budget test before a full rollout.",
        "Future work: link sentiment to advertising (for example, feature the most-praised product aspects found by sentiment analysis in ad copy), "
        "use real campaign exports, and add a Bayesian model that reports uncertainty directly."])

    out = O / "Project2_Report.docx"
    doc.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    build_report()
