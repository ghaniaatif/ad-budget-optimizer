"""Marketing Budget Recommendation Engine: "Where should we spend the next Rs. 1 million?"

How it works (plain version)
1. From the weekly history of each platform's campaigns, learn a *response curve*:  weekly revenue = A * (weekly spend) ** b
   (fitted on a log scale, with a yearly seasonal term because sunscreen sells best in summer).
   b < 1 means diminishing returns: the more you spend on a platform in a week, the less each extra rupee earns.
2. Find the split of the budget that maximises total predicted revenue. At the best split, one more rupee earns the
   same on every platform (equal marginal ROAS), except where a platform hits its minimum / maximum share.
3. Check the engine honestly: fit on the first weeks only and predict the held-out last weeks; bootstrap the history
   to show how much each recommended share could move; flag plans that go beyond the spend range seen in history.

    python -m adbudget.budget_recommender --budget 1000000 --weeks 4
"""
import argparse
import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from adbudget import settings as P

log = logging.getLogger("budget")
B_MIN, B_MAX = 0.2, 0.95       # the exponent b is kept in this range when optimising (b >= 1 would mean "spend everything here")


# ------------------------------------------------------------------ learning the response curves
def _season(mid_dates) -> tuple[np.ndarray, np.ndarray]:
    doy = pd.DatetimeIndex(mid_dates).dayofyear.to_numpy()
    return np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


def _fit_platform(g: pd.DataFrame, n_campaigns: int) -> dict | None:
    """Fit log(revenue) = campaign level + b*log(spend) + seasonal terms on campaign-weeks of ONE platform.

    Campaign-weeks (about 150 per platform) vary their spend far more than platform totals do, so b is identified much
    better than from 52 weekly totals. If b falls outside [B_MIN, B_MAX] it is clipped AND the other terms are refitted
    with b held fixed (otherwise the level of the curve would not match the clipped b).
    """
    g = g[(g["spend"] > 0) & (g["revenue"] > 0)]
    if len(g) < 30:
        return None
    camps = sorted(g["campaign"].unique())
    C = (g["campaign"].to_numpy()[:, None] == np.array(camps)[None, :]).astype(float)
    sn, cs = _season(g["week_start"] + pd.Timedelta(days=3))
    ln_s, y = np.log(g["spend"].to_numpy()), np.log(g["revenue"].to_numpy())
    k = len(camps)
    coef, *_ = np.linalg.lstsq(np.column_stack([C, ln_s, sn, cs]), y, rcond=None)
    b_raw = float(coef[k])
    b_used = float(np.clip(b_raw, B_MIN, B_MAX))
    if b_used != b_raw:
        coef2, *_ = np.linalg.lstsq(np.column_stack([C, sn, cs]), y - b_used * ln_s, rcond=None)
        alphas, c_sin, c_cos = coef2[:k], float(coef2[k]), float(coef2[k + 1])
    else:
        alphas, c_sin, c_cos = coef[:k], float(coef[k + 1]), float(coef[k + 2])
    resid = y - (C @ alphas + b_used * ln_s + c_sin * sn + c_cos * cs)
    sigma2 = float(resid.var(ddof=k + 3))
    r2 = 1 - float(resid.var()) / float(y.var()) if y.var() > 0 else 0.0
    return {"alpha": float(np.log(np.exp(alphas).sum()) - b_used * np.log(n_campaigns)),   # platform level if its campaigns share spend equally
            "b": b_raw, "b_used": b_used, "c_sin": c_sin, "c_cos": c_cos, "sigma2": sigma2, "r2_log": round(r2, 3),
            "n_obs": int(len(g)), "campaign_alpha": {str(c): float(a) for c, a in zip(camps, alphas)}}


def fit_curves(weekly_campaign: pd.DataFrame) -> dict:
    """One response curve per platform from the campaign-by-week table (columns: platform, campaign, week, week_start, sums)."""
    plat_week = weekly_campaign.groupby(["platform", "week"], as_index=False)[["spend", "revenue", "conversions"]].sum()
    out = {}
    for plat, g in weekly_campaign.groupby("platform"):
        fit = _fit_platform(g, g["campaign"].nunique())
        if fit is None:
            continue
        pw = plat_week[plat_week["platform"] == plat]
        fit.update(hist_min_weekly_spend=float(pw["spend"].min()), hist_max_weekly_spend=float(pw["spend"].max()),
                   hist_mean_weekly_spend=float(pw["spend"].mean()),
                   revenue_per_conversion=float(pw["revenue"].sum() / max(pw["conversions"].sum(), 1)))
        out[str(plat)] = fit
    return out


def holdout_check(weekly_campaign: pd.DataFrame, holdout_weeks: int = P.HOLDOUT_WEEKS) -> dict:
    """Fit on all but the last weeks, then predict each held-out campaign-week from its real spend and add up per platform-week."""
    cut = weekly_campaign["week"].max() - holdout_weeks + 1
    curves = fit_curves(weekly_campaign[weekly_campaign["week"] < cut])
    test = weekly_campaign[(weekly_campaign["week"] >= cut) & weekly_campaign["platform"].isin(curves)].copy()
    sn, cs = _season(test["week_start"] + pd.Timedelta(days=3))
    pred = np.empty(len(test))
    for i, (plat, camp, spend) in enumerate(zip(test["platform"], test["campaign"], test["spend"])):
        c = curves[plat]
        pred[i] = np.exp(c["campaign_alpha"][camp] + c["b_used"] * np.log(max(spend, 1e-9)) + c["c_sin"] * sn[i] + c["c_cos"] * cs[i] + c["sigma2"] / 2)
    test["pred"] = pred
    pw = test.groupby(["platform", "week"], as_index=False)[["revenue", "pred"]].sum()
    rows = {}
    for plat, g in pw.groupby("platform"):
        rows[str(plat)] = {"mape_pct": round(float(np.mean(np.abs(g["pred"] - g["revenue"]) / np.maximum(g["revenue"], 1)) * 100), 1),
                           "total_error_pct": round(float((g["pred"].sum() - g["revenue"].sum()) / g["revenue"].sum() * 100), 1)}
    tot_a, tot_p = float(pw["revenue"].sum()), float(pw["pred"].sum())
    return {"holdout_weeks": holdout_weeks, "per_platform": rows,
            "total_revenue_error_pct": round((tot_p - tot_a) / tot_a * 100, 1) if tot_a else None,
            "mean_mape_pct": round(float(np.mean([r["mape_pct"] for r in rows.values()])), 1) if rows else None}


# ------------------------------------------------------------------ choosing the split
def _plan_terms(curve: dict, plan_start: date, weeks: int) -> float:
    """Average seasonal effect over the planned weeks (log scale) plus the log-normal bias correction."""
    mids = pd.date_range(pd.Timestamp(plan_start) + pd.Timedelta(days=3), periods=weeks, freq="7D")
    s, c = _season(mids)
    return float(np.mean(curve["c_sin"] * s + curve["c_cos"] * c) + curve["sigma2"] / 2)


def allocate(curves: dict, budget: float = P.BUDGET, weeks: int = P.HORIZON_WEEKS, plan_start: date | None = None,
             min_share: float = P.MIN_SHARE, max_share: float = P.MAX_SHARE, history_cap: float | None = P.HISTORY_CAP) -> pd.DataFrame:
    """Best split of `budget` (spent evenly over `weeks`) across the platforms in `curves`.

    history_cap: a platform is not planned above this multiple of its highest weekly spend in history, because the
    response curve is only trustworthy inside the range it was learned from. Use None to switch the safeguard off.
    """
    plan_start = plan_start or (P.START_DATE + timedelta(days=7 * P.N_WEEKS))
    plats = list(curves)
    n = len(plats)
    if not (n * min_share <= 1 + 1e-9 and n * max_share >= 1 - 1e-9):
        raise ValueError(f"Min/max share limits are impossible for {n} platforms (min x {n} must be <= 100% <= max x {n}).")
    W = budget / weeks
    A = np.array([np.exp(curves[p]["alpha"] + _plan_terms(curves[p], plan_start, weeks)) for p in plats])
    b = np.array([curves[p]["b_used"] for p in plats])
    lo, hi = np.full(n, min_share * W), np.full(n, max_share * W)
    cap = np.full(n, np.inf)
    if history_cap:
        cap = history_cap * np.array([curves[p]["hist_max_weekly_spend"] for p in plats])
        hi = np.minimum(hi, cap)
    lo = np.minimum(lo, hi)
    if hi.sum() < W * (1 - 1e-9):
        raise ValueError("This budget is larger than the platforms' historical spend range allows. Spend it over more weeks, "
                         "raise the max share, or switch off the history safeguard.")

    def spend_at(lam: float) -> np.ndarray:
        return np.clip((lam / (A * b)) ** (1.0 / (b - 1.0)), lo, hi)

    l_lo, l_hi = 1e-9, 1e9                        # bisection on the common marginal revenue per rupee (log scale)
    for _ in range(300):
        mid = np.sqrt(l_lo * l_hi)
        if spend_at(mid).sum() > W:
            l_lo = mid
        else:
            l_hi = mid
    w = spend_at(np.sqrt(l_lo * l_hi))
    w = w * (W / w.sum())                         # remove the tiny numerical gap
    return _table(curves, plats, A, b, w, weeks, lo, hi, cap, budget)


def _table(curves, plats, A, b, w, weeks, lo, hi, cap, budget) -> pd.DataFrame:
    spend = w * weeks
    rounded = np.round(spend / 1000) * 1000                      # whole thousands of rupees are easier to act on
    rounded[np.argmax(rounded)] += budget - rounded.sum()
    wr = rounded / weeks
    rev = weeks * A * wr ** b
    df = pd.DataFrame({
        "platform": plats, "share_pct": np.round(rounded / budget * 100, 1), "spend": rounded, "weekly_spend": np.round(wr),
        "expected_revenue": np.round(rev), "expected_roas": np.round(rev / rounded, 2),
        "expected_conversions": np.round(rev / np.array([curves[p]["revenue_per_conversion"] or np.nan for p in plats])),
        "marginal_roas": np.round(A * b * wr ** (b - 1), 2),
        "curve_exponent_b": np.round([curves[p]["b"] for p in plats], 2),
        "limit": np.where(np.isclose(w, hi, rtol=1e-6), np.where(np.isclose(hi, cap, rtol=1e-6), "at history cap", "at max share"),
                          np.where(np.isclose(w, lo, rtol=1e-6), "at min share", "")),
        "within_history": [bool(curves[p]["hist_min_weekly_spend"] * 0.95 <= x <= curves[p]["hist_max_weekly_spend"] * 1.05)
                           for p, x in zip(plats, wr)],
    })
    return df.sort_values("share_pct", ascending=False).reset_index(drop=True)


def expected_revenue(curves: dict, spend_by_platform: dict, weeks: int, plan_start: date | None = None) -> float:
    plan_start = plan_start or (P.START_DATE + timedelta(days=7 * P.N_WEEKS))
    total = 0.0
    for p, s in spend_by_platform.items():
        c = curves[p]
        total += weeks * np.exp(c["alpha"] + _plan_terms(c, plan_start, weeks)) * (max(s, 1e-9) / weeks) ** c["b_used"]
    return float(total)


def compare_plans(curves: dict, rec: pd.DataFrame, history_share: dict, budget: float, weeks: int, plan_start=None) -> pd.DataFrame:
    """Predicted result of the recommended split against 'equal split' and 'same as last year'."""
    plats = list(rec["platform"])
    plans = {"Recommended": dict(zip(rec["platform"], rec["spend"])),
             "Equal split": {p: budget / len(plats) for p in plats},
             "Same as last year": {p: budget * history_share[p] for p in plats}}
    rows = []
    for name, sp in plans.items():
        rev = expected_revenue(curves, sp, weeks, plan_start)
        rows.append({"plan": name, "expected_revenue": round(rev), "expected_roas": round(rev / budget, 2), **{p: round(sp[p] / budget * 100, 1) for p in plats}})
    out = pd.DataFrame(rows)
    out["uplift_vs_this_plan_pct"] = np.round((out.loc[0, "expected_revenue"] / out["expected_revenue"] - 1) * 100, 1)
    return out.rename(columns={"uplift_vs_this_plan_pct": "recommended_is_better_by_pct"})


def bootstrap_shares(weekly: pd.DataFrame, n: int = 200, seed: int = P.SEED, **kw) -> pd.DataFrame:
    """Refit on resampled weeks n times: how much could each recommended share move because of noisy history?"""
    rng = np.random.default_rng(seed)
    n_plat = weekly["platform"].nunique()
    weeks = np.sort(weekly["week"].unique())
    by_week = {w: g for w, g in weekly.groupby("week")}
    shares = []
    for _ in range(n):
        pick = rng.choice(weeks, size=len(weeks), replace=True)
        boot = pd.concat([by_week[w].assign(week=j) for j, w in enumerate(pick)], ignore_index=True)   # new id per draw, so a week drawn twice is not added up twice
        curves = fit_curves(boot)
        if len(curves) == n_plat:
            rec = allocate(curves, **kw)
            shares.append(rec.set_index("platform")["share_pct"])
    s = pd.DataFrame(shares)
    return pd.DataFrame({"p05": s.quantile(0.05), "median": s.median(), "p95": s.quantile(0.95)}).round(1).rename_axis("platform").reset_index()


# ------------------------------------------------------------------ run + save
def recommend(weekly: pd.DataFrame, budget: float = P.BUDGET, weeks: int = P.HORIZON_WEEKS, save: bool = True,
              bootstrap: int = 200) -> dict:
    curves = fit_curves(weekly)
    rec = allocate(curves, budget, weeks)
    hist = weekly.groupby("platform")["spend"].sum()
    hist_share = (hist / hist.sum()).to_dict()
    cmp_ = compare_plans(curves, rec, hist_share, budget, weeks)
    rec["historical_share_pct"] = rec["platform"].map(lambda p: round(hist_share[p] * 100, 1))
    result = {"budget": budget, "weeks": weeks, "min_share": P.MIN_SHARE, "max_share": P.MAX_SHARE,
              "plan_start": str(P.START_DATE + timedelta(days=7 * P.N_WEEKS)),
              "curves": curves, "holdout": holdout_check(weekly),
              "total_expected_revenue": float(rec["expected_revenue"].sum()),
              "total_expected_roas": round(float(rec["expected_revenue"].sum() / budget), 2)}
    boot = bootstrap_shares(weekly, bootstrap, budget=budget, weeks=weeks) if bootstrap else None
    if save:
        rec.to_csv(P.OUT_DIR / "recommendation.csv", index=False)
        cmp_.to_csv(P.OUT_DIR / "plan_comparison.csv", index=False)
        if boot is not None:
            boot.to_csv(P.OUT_DIR / "recommendation_uncertainty.csv", index=False)
        (P.OUT_DIR / "response_curves.json").write_text(json.dumps(result, indent=2))
    return {"recommendation": rec, "comparison": cmp_, "uncertainty": boot, **result}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=P.BUDGET)
    ap.add_argument("--weeks", type=int, default=P.HORIZON_WEEKS)
    ap.add_argument("--weekly-csv", default=str(P.OUT_DIR / "weekly_campaign.csv"), help="made by python -m adbudget.analysis")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    weekly = pd.read_csv(args.weekly_csv, parse_dates=["week_start"])
    res = recommend(weekly, args.budget, args.weeks)
    pd.set_option("display.width", 200)
    print(f"\nWhere to spend Rs {args.budget:,.0f} over {args.weeks} weeks\n")
    print(res["recommendation"].to_string(index=False))
    print("\n", res["comparison"].to_string(index=False))
    print("\nHoldout check:", res["holdout"])


if __name__ == "__main__":
    main()
