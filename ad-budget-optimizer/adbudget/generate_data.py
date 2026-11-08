"""Big Data simulation: generate advertising records (10K -> 100K -> 1M -> 10M) and sunscreen comments.

Ad records follow the project brief:
    date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device

Design choices (so results can be compared across scales):
* The *business* is the same at every scale (about Rs 12 million spent over 52 weeks, 15 campaigns, 5 platforms).
  A bigger dataset is a finer-grained log of that same business, so KPIs and budget advice should agree across scales.
* Every platform has "diminishing returns": revenue grows like spend**b with b < 1 (see settings.PLATFORM_PARAMS).
  A campaign that spends more in a week gets a worse conversion rate. This is what the budget engine must discover.
* Conversion rate follows the seasons (sunscreen sells best in summer) and differs a little by device, gender and age.

Files are written as Parquet in data/ads_<scale>/ (git-ignored).  Run:  python -m adbudget.generate_data --scales 10K 100K
"""
import argparse
import logging
import shutil
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from adbudget import settings as P

log = logging.getLogger("generate")
EPOCH = date(1970, 1, 1)
CAMPAIGNS = [(p, f"{p} | {t}") for p in P.PLATFORMS for t in P.THEMES]      # 15 campaigns
N_CAMP, N_CW = len(CAMPAIGNS), len(CAMPAIGNS) * P.N_WEEKS                  # N_CW = campaign-weeks (780)
_Z = np.random.default_rng(P.SEED).standard_normal((N_CAMP, P.N_WEEKS))      # fixed weekly budget shocks
_M = np.exp(P.BUDGET_NOISE * _Z - P.BUDGET_NOISE ** 2 / 2)                  # budget multiplier per campaign-week (mean 1)


def _dict_col(codes: np.ndarray, names: list[str]) -> pa.Array:
    return pa.DictionaryArray.from_arrays(pa.array(codes.astype(np.int8)), pa.array(names))


def _pick(rng, spec: dict, n: int):
    names = list(spec)
    p = np.array([v[0] if isinstance(v, tuple) else v for v in spec.values()], dtype=float)
    return names, rng.choice(len(names), size=n, p=p / p.sum())


# ------------------------------------------------------------------ advertising records
def ads_chunk(start: int, n: int, total: int, rng: np.random.Generator) -> pa.Table:
    """Rows [start, start+n) of a dataset that has `total` rows."""
    idx = np.arange(start, start + n, dtype=np.int64)
    cw = (idx * 7919) % N_CW                       # 7919 is coprime with 780, so rows spread evenly over campaign-weeks
    camp, week = cw % N_CAMP, cw // N_CAMP
    plat = camp // len(P.THEMES)
    prm = {k: np.array([P.PLATFORM_PARAMS[p][k] for p in P.PLATFORMS]) for k in ("cpm", "ctr", "aov", "roas", "b", "share")}
    m = _M[camp, week]
    weekly_budget = P.ANNUAL_SPEND * prm["share"][plat] / P.N_WEEKS / len(P.THEMES) * m      # Rs for this campaign-week
    imp = rng.poisson(weekly_budget / (prm["cpm"][plat] / 1000) / (total / N_CW))

    dev_names, dev = _pick(rng, P.DEVICES, n)
    gen_names, gen = _pick(rng, P.GENDERS, n)
    age_names, agg = _pick(rng, P.AGE_GROUPS, n)
    city_names, city = _pick(rng, P.CITIES, n)
    lo = np.array([P.AGE_GROUPS[a][0] for a in age_names])[agg]
    hi = np.array([P.AGE_GROUPS[a][1] for a in age_names])[agg]
    age = rng.integers(lo, hi + 1)
    ctr_mod = np.array([P.DEVICES[d][1] for d in dev_names])[dev] * np.array([P.AGE_GROUPS[a][3] for a in age_names])[agg]
    cvr_mod = (np.array([P.DEVICES[d][2] for d in dev_names])[dev] * np.array([P.GENDERS[g][2] for g in gen_names])[gen]
               * np.array([P.AGE_GROUPS[a][4] for a in age_names])[agg])

    day = week * 7 + rng.integers(0, 7, n)
    doy = ((date.fromordinal(P.START_DATE.toordinal()) - date(P.START_DATE.year, 1, 1)).days + day) % 365
    season = 1 + P.SEASON_AMPLITUDE * np.sin(2 * np.pi * (doy - 80) / 365)             # peak around 20 June
    ctr = prm["ctr"][plat] * ctr_mod
    cvr0 = prm["roas"][plat] * (prm["cpm"][plat] / 1000) / (prm["ctr"][plat] * prm["aov"][plat])
    cvr = np.clip(cvr0 * cvr_mod * season * m ** (-(1 - prm["b"][plat])), 0, 0.6)      # diminishing returns live here

    clicks = rng.binomial(imp, np.clip(ctr, 0, 1))
    conv = rng.binomial(clicks, cvr)
    spend = np.round(imp * prm["cpm"][plat] / 1000 * rng.lognormal(0, 0.03, n), 2)
    shape = np.maximum(conv, 1) * 4.0
    revenue = np.round(np.where(conv > 0, rng.gamma(shape, prm["aov"][plat] / 4.0), 0.0), 2)
    d0 = (P.START_DATE - EPOCH).days

    return pa.table({
        "date": pa.array((d0 + day).astype(np.int32), pa.date32()),
        "campaign": _dict_col(camp, [c[1] for c in CAMPAIGNS]),
        "platform": _dict_col(plat, P.PLATFORMS),
        "impressions": pa.array(imp.astype(np.int32)),
        "clicks": pa.array(clicks.astype(np.int32)),
        "spend": pa.array(spend),
        "conversions": pa.array(conv.astype(np.int32)),
        "revenue": pa.array(revenue),
        "age": pa.array(age.astype(np.int8)),
        "gender": _dict_col(gen, gen_names),
        "city": _dict_col(city, city_names),
        "device": _dict_col(dev, dev_names),
    })


def ads_dir(label: str):
    return P.DATA_DIR / f"ads_{label}"


def ads_files(label: str) -> list:
    return sorted(ads_dir(label).glob("part-*.parquet"))


def _complete(files: list, rows: int) -> bool:
    return bool(files) and sum(pq.ParquetFile(f).metadata.num_rows for f in files) == rows


def ensure_ads(label: str, force: bool = False) -> list:
    """Create the parquet files for one scale if they are missing. Returns the file list."""
    total = P.SCALES[label]
    if not force and _complete(ads_files(label), total):
        return ads_files(label)
    shutil.rmtree(ads_dir(label), ignore_errors=True)
    ads_dir(label).mkdir(parents=True)
    for k, start in enumerate(range(0, total, P.CHUNK_ROWS)):
        n = min(P.CHUNK_ROWS, total - start)
        rng = np.random.default_rng([P.SEED, total, k])
        pq.write_table(ads_chunk(start, n, total, rng), ads_dir(label) / f"part-{k:04d}.parquet",
                       row_group_size=P.ROW_GROUP_ROWS, compression="snappy")
        log.info("ads %-4s chunk %d  (%d rows)", label, k, n)
    return ads_files(label)


# ------------------------------------------------------------------ sunscreen comments (for sentiment scoring at scale)
_OPEN = ["", "", "", "honestly ", "so far ", "update: ", "after two weeks ", "tbh ", "used this all summer and ", "my dermatologist suggested it and "]
_ASPECT = ["this sunscreen", "the spf 50", "the gel formula", "the sun stick", "this one", "the texture", "the smell", "the white cast",
           "the price", "the bottle", "the pump", "it", "the finish under makeup", "the protection", "this lotion", "the spray"]
_PRED = {
    "positive": ["is amazing", "feels so light on my skin", "leaves no white cast at all", "is worth every rupee",
                 "works perfectly under makeup", "absorbs quickly and feels weightless", "is my holy grail",
                 "protected me all day at the beach", "smells lovely and does not sting", "is hydrating and never greasy",
                 "I love it", "is really good for oily skin", "is the best I have tried", "is smooth and invisible"],
    "negative": ["is terrible", "leaves a thick white cast", "broke me out badly", "feels greasy and sticky", "stings my eyes",
                 "is way too expensive", "smells awful", "pilled under my foundation", "gave me a rash", "is a waste of money",
                 "burned me even after reapplying", "made my skin oily and clogged", "I hate it", "is chalky and ashy on dark skin"],
    "neutral": ["is available in two sizes", "comes in a 50 ml tube", "is what I apply every morning", "was delivered yesterday",
                "has spf 50 and pa++", "is sold at the pharmacy near me", "is the one my sister mentioned", "I bought it last week",
                "arrived in a plain box", "can be found online", "is a lotion with a pump", "needs to be reapplied every two hours"],
}
_CLOSE = {"positive": ["", "", " highly recommend", " will buy again", " five stars", " thank you for the review"],
          "negative": ["", "", " never again", " do not buy", " very disappointed", " returning it"],
          "neutral": ["", "", " not sure yet", " will see", " any thoughts", " just sharing"]}
CLASS_P = {"positive": 0.45, "neutral": 0.20, "negative": 0.35}
MIXED_PREDICATE = 0.12          # share of comments whose opinion phrase comes from another class (keeps the task realistic)


def synthetic_comments(n: int, rng: np.random.Generator) -> pd.DataFrame:
    labels = np.array(list(CLASS_P))
    lab = rng.choice(labels, size=n, p=list(CLASS_P.values()))
    pred_src = np.where(rng.random(n) < MIXED_PREDICATE, rng.choice(labels, size=n), lab)
    pred, close = np.empty(n, dtype=object), np.empty(n, dtype=object)
    for c in labels:
        k = pred_src == c
        pred[k] = rng.choice(_PRED[c], size=int(k.sum()))
        k2 = lab == c
        close[k2] = rng.choice(_CLOSE[c], size=int(k2.sum()))
    text = (pd.Series(rng.choice(_OPEN, size=n)) + pd.Series(rng.choice(_ASPECT, size=n)) + " " + pd.Series(pred) + pd.Series(close))
    return pd.DataFrame({"text": text.str.strip(), "sentiment": lab})


def real_comments_available() -> bool:
    return P.LABELED_CSV is not None and P.LABELED_CSV.exists()


def comments_chunk(n: int, rng: np.random.Generator, source: str) -> pd.DataFrame:
    if source == "real":     # bootstrap labelled comments from the sentiment project
        real = pd.read_csv(P.LABELED_CSV, encoding="utf-8-sig", usecols=["text", "sentiment"])
        return real.sample(n=n, replace=True, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    return synthetic_comments(n, rng)


def comments_dir(label: str):
    return P.DATA_DIR / f"comments_{label}"


def comments_files(label: str) -> list:
    return sorted(comments_dir(label).glob("part-*.parquet"))


def ensure_comments(label: str, source: str = "auto", force: bool = False) -> list:
    source = ("real" if real_comments_available() else "synthetic") if source == "auto" else source
    total = P.SCALES[label]
    meta = comments_dir(label) / "_source.txt"
    tag = f"{source}|{P.COMMENT_ROW_GROUP_ROWS}"
    if not force and _complete(comments_files(label), total) and meta.exists() and meta.read_text() == tag:
        return comments_files(label)
    shutil.rmtree(comments_dir(label), ignore_errors=True)
    comments_dir(label).mkdir(parents=True)
    for k, start in enumerate(range(0, total, P.CHUNK_ROWS)):
        n = min(P.CHUNK_ROWS, total - start)
        df = comments_chunk(n, np.random.default_rng([P.SEED, 99, total, k]), source)
        df.insert(0, "comment_id", np.arange(start, start + n, dtype=np.int64))
        df["sentiment"] = df["sentiment"].astype("category")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), comments_dir(label) / f"part-{k:04d}.parquet",
                       row_group_size=P.COMMENT_ROW_GROUP_ROWS, compression="snappy")
        log.info("comments %-4s chunk %d (%d rows, %s)", label, k, n, source)
    meta.write_text(tag)
    return comments_files(label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scales", nargs="+", default=list(P.SCALES), choices=list(P.SCALES))
    ap.add_argument("--with-comments", action="store_true", help="also generate sunscreen comments up to NLP_MAX_SCALE")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for s in args.scales:
        ensure_ads(s, args.force)
        if args.with_comments and P.SCALES[s] <= P.SCALES[P.NLP_MAX_SCALE]:
            ensure_comments(s, force=args.force)


if __name__ == "__main__":
    main()
