"""Compare Pandas vs MapReduce vs Spark at 10K, 100K, 1M and 10M records.

    python -m adbudget.benchmark                       # all scales; sentiment job up to 1M
    python -m adbudget.benchmark --scales 10K 100K     # quick run
    python -m adbudget.benchmark --nlp-max 10M         # also score 10M comments (slow: ~10 minutes per engine)

Every engine's answer is compared with Pandas, so a fast but wrong result is flagged. Results are saved after each
measurement to outputs/benchmark_results.csv (+ benchmark_meta.json with the machine description).
"""
import argparse
import json
import logging
import os
import platform
import statistics
import time

import pandas as pd

from adbudget import engines as E
from adbudget import generate_data as G
from adbudget import nlp_model
from adbudget import settings as P

log = logging.getLogger("benchmark")
RESULTS = P.OUT_DIR / "benchmark_results.csv"
META = P.OUT_DIR / "benchmark_meta.json"


def timed(fn, files, repeats: int, **kw):
    runs, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(files, **kw)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs), out


def run(scales: list[str], nlp_max: str, engines: list[str]) -> pd.DataFrame:
    model_path, model_kind = nlp_model.ensure_model()
    rows = []
    if "Spark" in engines:                                   # warm up the JVM so the first measurement is not unfairly slow
        E.get_spark()
        E.ads_spark(G.ensure_ads("10K"))
        E.nlp_spark(G.ensure_comments("10K"), model_path=model_path)
    for label in scales:
        n = P.SCALES[label]
        repeats = 3 if n <= 100_000 else 1
        for job in ("ads", "nlp"):
            if job == "nlp" and n > P.SCALES[nlp_max]:
                continue
            files = G.ensure_ads(label) if job == "ads" else G.ensure_comments(label)
            reference = None
            for name in engines:
                secs, out = timed(E.ENGINES[job][name], files, repeats, model_path=model_path)
                if name == "Pandas" or reference is None:
                    reference = out if reference is None else reference
                ok = E.same_answer(job, reference, out)
                rows.append({"job": job, "scale": label, "rows": n, "engine": name, "seconds": round(secs, 3),
                             "rows_per_second": round(n / secs), "repeats": repeats, "same_answer_as_pandas": ok})
                log.info("%-4s %-4s %-9s %8.2fs  %s", job, label, name, secs, "ok" if ok else "DIFFERENT ANSWER")
                pd.DataFrame(rows).to_csv(RESULTS, index=False)
    meta = {"cpu_cores": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform(),
            "pandas": pd.__version__, "model_used_for_nlp_job": model_kind, "nlp_max_scale": nlp_max,
            "spark_startup_seconds": round(E.spark_startup_seconds, 2) if E.spark_startup_seconds else None,
            "note": "Seconds are the median of `repeats` runs, measured from reading the files to having the final table. "
                    "Spark start-up is reported separately and a warm-up job was run before timing."}
    try:
        import pyspark
        meta["pyspark"] = pyspark.__version__
    except ImportError:
        pass
    META.write_text(json.dumps(meta, indent=2))
    E.stop_spark()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scales", nargs="+", default=list(P.SCALES), choices=list(P.SCALES))
    ap.add_argument("--nlp-max", default=P.NLP_MAX_SCALE, choices=list(P.SCALES))
    ap.add_argument("--engines", nargs="+", default=["Pandas", "MapReduce", "Spark"], choices=["Pandas", "MapReduce", "Spark"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    run(args.scales, args.nlp_max, args.engines)


if __name__ == "__main__":
    main()
