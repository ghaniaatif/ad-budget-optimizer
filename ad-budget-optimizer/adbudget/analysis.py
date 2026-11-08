"""Campaign performance tables (CTR, CPC, CPA, conversion rate, ROAS, revenue) from the simulated records.

    python -m adbudget.analysis --scale 10M

Writes small CSV files to outputs/ that the dashboard, the budget engine and the report read, so the big
Parquet files are only needed once.
"""
import argparse
import json
import logging

import pandas as pd

from adbudget import generate_data as G
from adbudget import kpis
from adbudget import settings as P

log = logging.getLogger("analysis")
COLS = ["date", "campaign", "platform", "impressions", "clicks", "spend", "conversions", "revenue", "age", "gender", "city", "device"]


def load_ads(label: str, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or COLS
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in G.ensure_ads(label)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if "age" in df:
        df["age_group"] = pd.cut(df["age"], [17, 24, 34, 44, 64], labels=list(P.AGE_GROUPS)).astype(str)
    return df


def weekly_platform(df: pd.DataFrame) -> pd.DataFrame:
    """One row per platform and week: the table the budget engine learns from."""
    wk = ((df["date"] - pd.Timestamp(P.START_DATE)).dt.days // 7).rename("week")
    g = df.groupby(["platform", wk], observed=True)[kpis.SUM_COLS].sum().reset_index()
    g["week_start"] = pd.Timestamp(P.START_DATE) + pd.to_timedelta(g["week"] * 7, unit="D")
    g["platform"] = g["platform"].astype(str)
    return g


def weekly_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """One row per campaign and week: what the budget engine learns its response curves from."""
    wk = ((df["date"] - pd.Timestamp(P.START_DATE)).dt.days // 7).rename("week")
    g = df.groupby(["platform", "campaign", wk], observed=True)[kpis.SUM_COLS].sum().reset_index()
    g["week_start"] = pd.Timestamp(P.START_DATE) + pd.to_timedelta(g["week"] * 7, unit="D")
    g[["platform", "campaign"]] = g[["platform", "campaign"]].astype(str)
    return g


def analyse(label: str) -> dict:
    df = load_ads(label)
    out = P.OUT_DIR
    plat = kpis.rollup(df, ["platform"])
    plat["platform"] = plat["platform"].astype(str)
    plat["spend_share"] = plat["spend"] / plat["spend"].sum()
    plat.sort_values("roas", ascending=False).to_csv(out / "platform_kpis.csv", index=False)

    camp = kpis.rollup(df, ["platform", "campaign"])
    camp[["platform", "campaign"]] = camp[["platform", "campaign"]].astype(str)
    camp["verdict"] = camp["roas"].map(kpis.verdict)
    camp.sort_values("roas", ascending=False).to_csv(out / "campaign_kpis.csv", index=False)

    df["month"] = df["date"].dt.strftime("%Y-%m")
    month = kpis.rollup(df, ["month", "platform"])
    month["platform"] = month["platform"].astype(str)
    month.to_csv(out / "monthly_platform_kpis.csv", index=False)

    seg = [kpis.rollup(df, [c]).rename(columns={c: "value"}).assign(segment=c) for c in ("device", "gender", "age_group", "city")]
    pd.concat(seg).astype({"value": str})[["segment", "value"] + [c for c in seg[0].columns if c != "value"]].to_csv(out / "segment_kpis.csv", index=False)

    weekly_platform(df).to_csv(out / "weekly_platform.csv", index=False)
    weekly_campaign(df).to_csv(out / "weekly_campaign.csv", index=False)

    total = kpis.add_kpis(pd.DataFrame([df[kpis.SUM_COLS].sum()]))
    summary = {"scale": label, "rows": int(len(df)), "date_from": str(df["date"].min().date()), "date_to": str(df["date"].max().date()),
               **{k: float(total[k].iloc[0]) for k in kpis.SUM_COLS + ["ctr", "cpc", "cpa", "conversion_rate", "roas", "profit"]},
               "n_campaigns": int(df["campaign"].nunique()), "n_platforms": int(df["platform"].nunique())}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Analysed %s records: ROAS %.2f, spend Rs %.0f", f"{len(df):,}", summary["roas"], summary["spend"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", default="1M", choices=list(P.SCALES))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    analyse(args.scale)


if __name__ == "__main__":
    main()
