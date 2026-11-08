"""Advertising KPIs from summed metrics (always compute KPIs from SUMS, never average per-row ratios)."""
import numpy as np
import pandas as pd

from adbudget import settings as P

SUM_COLS = ["impressions", "clicks", "spend", "conversions", "revenue"]


def _div(a, b):
    b = pd.Series(b).replace(0, np.nan)
    return pd.Series(a) / b


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add CTR, CPC, CPA, conversion rate, ROAS (and profit) to a table that has the five summed columns."""
    out = df.copy()
    out["ctr"] = _div(out["clicks"], out["impressions"])            # clicks / impressions
    out["cpc"] = _div(out["spend"], out["clicks"])                  # Rs per click
    out["cpa"] = _div(out["spend"], out["conversions"])             # Rs per conversion (acquisition)
    out["conversion_rate"] = _div(out["conversions"], out["clicks"])
    out["roas"] = _div(out["revenue"], out["spend"])                # revenue per Rs spent
    out["profit"] = out["revenue"] - out["spend"]                   # revenue minus ad spend (before product cost)
    return out


def verdict(roas: float) -> str:
    if pd.isna(roas):
        return "No data"
    for floor, label in P.ROAS_BANDS:
        if roas >= floor:
            return label
    return P.ROAS_BANDS[-1][1]


def rollup(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum the metrics by the given columns and add KPIs."""
    g = df.groupby(by, observed=True)[SUM_COLS].sum().reset_index()
    g.insert(len(by), "rows", df.groupby(by, observed=True).size().values)
    return add_kpis(g)
