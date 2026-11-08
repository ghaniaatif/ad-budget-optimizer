"""Run the whole project:  data -> benchmark -> KPIs -> budget recommendation -> scale-stability check -> Word report.

    python -m adbudget.run_all                    # everything, up to 10 million records (about 10-15 minutes)
    python -m adbudget.run_all --quick            # 10K and 100K only (about 1 minute), good for a first try
    python -m adbudget.run_all --skip-benchmark   # reuse the saved benchmark, redo the rest
"""
import argparse
import json
import logging

import pandas as pd

from adbudget import analysis, benchmark, budget_recommender
from adbudget import generate_data as G
from adbudget import settings as P

log = logging.getLogger("run_all")
SUM_AND_KEYS = ["date", "platform", "campaign", "impressions", "clicks", "spend", "conversions", "revenue"]


def scale_stability(scales: list[str], budget: float, weeks: int) -> pd.DataFrame:
    """Does the recommendation change with dataset size? (Same business, finer logs: it should settle as data grows.)"""
    rows = []
    for label in scales:
        weekly = analysis.weekly_campaign(analysis.load_ads(label, SUM_AND_KEYS))
        curves = budget_recommender.fit_curves(weekly)
        rec = budget_recommender.allocate(curves, budget, weeks).set_index("platform")["share_pct"]
        rows.append({"scale": label, "rows": P.SCALES[label], **rec.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(P.OUT_DIR / "scale_stability.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="only 10K and 100K records")
    ap.add_argument("--scales", nargs="+", choices=list(P.SCALES))
    ap.add_argument("--nlp-max", default=P.NLP_MAX_SCALE, choices=list(P.SCALES))
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--budget", type=float, default=P.BUDGET)
    ap.add_argument("--weeks", type=int, default=P.HORIZON_WEEKS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    scales = args.scales or (["10K", "100K"] if args.quick else list(P.SCALES))
    biggest = scales[-1] if args.scales or args.quick else "10M"

    log.info("1/5 Generating data: %s", ", ".join(scales))
    for s in scales:
        G.ensure_ads(s)
    if not args.skip_benchmark:
        log.info("2/5 Benchmark: Pandas vs MapReduce vs Spark")
        nlp_max = args.nlp_max if P.SCALES[args.nlp_max] <= P.SCALES[biggest] else biggest
        benchmark.run(scales, nlp_max, ["Pandas", "MapReduce", "Spark"])
    log.info("3/5 Campaign performance on %s records", biggest)
    analysis.analyse(biggest)
    log.info("4/5 Budget recommendation for Rs %s over %d weeks", f"{args.budget:,.0f}", args.weeks)
    weekly = pd.read_csv(P.OUT_DIR / "weekly_campaign.csv", parse_dates=["week_start"])
    res = budget_recommender.recommend(weekly, args.budget, args.weeks)
    print(res["recommendation"][["platform", "share_pct", "spend", "expected_revenue", "expected_roas"]].to_string(index=False))
    log.info("5/5 Scale stability of the recommendation")
    scale_stability(scales, args.budget, args.weeks)
    (P.OUT_DIR / "run_info.json").write_text(json.dumps({"scales": scales, "analysis_scale": biggest, "budget": args.budget, "weeks": args.weeks}, indent=2))
    if not args.no_report:
        from adbudget.report import build_report
        build_report()
        log.info("Report: %s", P.OUT_DIR / "Project2_Report.docx")
    log.info("Done. Dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
