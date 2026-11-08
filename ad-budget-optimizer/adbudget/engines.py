"""Three ways to process the same data: Pandas, MapReduce and Spark.

Two jobs are run on every engine, so the comparison is fair and the answers can be checked against each other:

  Job A  "ads":  sum impressions, clicks, spend, conversions, revenue per (platform, campaign)  -> every KPI follows from these
  Job B  "nlp":  classify every sunscreen comment with a TF-IDF + Logistic Regression model, count labels

* Pandas     one machine, one process: load into memory, vectorised operations.
* MapReduce  the classic model, written out in plain Python: map (one task per parquet row-group, one record at a time,
             with a per-mapper combiner) -> shuffle (partition by key hash) -> reduce (merge partial sums).
             Runs on all CPU cores with multiprocessing. It shows the idea; it is not Hadoop.
* Spark      PySpark in local mode (all cores): the same job as DataFrame operations (job B uses mapInPandas + a model
             broadcast through a file path), planned and executed by the Spark engine.
"""
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ADS_COLS = ["platform", "campaign", "impressions", "clicks", "spend", "conversions", "revenue"]
ADS_SUMS = ["rows", "impressions", "clicks", "spend", "conversions", "revenue"]
WORKERS = os.cpu_count() or 2
NO_PRED = "no_prediction"      # comment with no word the model knows (same rule as predict.py)


# ================================================================== shared helpers
def row_groups(files) -> list[tuple[str, int]]:
    """One task per parquet row-group: the unit of parallel work for MapReduce (Spark splits the same way)."""
    return [(str(f), i) for f in files for i in range(pq.ParquetFile(f).num_row_groups)]


def _tidy_ads(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["platform"], df["campaign"] = df["platform"].astype(str), df["campaign"].astype(str)
    return df[["platform", "campaign"] + ADS_SUMS].sort_values(["platform", "campaign"]).reset_index(drop=True)


_MODELS: dict = {}


def load_model(path: str):
    if path not in _MODELS:
        _MODELS[path] = joblib.load(path)
    return _MODELS[path]


def score_texts(texts: list[str], model) -> tuple[np.ndarray, np.ndarray]:
    """Classify a batch: clean -> TF-IDF -> probabilities; no known word = no prediction."""
    from adbudget.text_clean import clean_text
    cleaned = [clean_text(t) for t in texts]
    x = model.named_steps["tfidf"].transform(cleaned)
    proba = model.named_steps["clf"].predict_proba(x)
    labels = model.classes_[proba.argmax(axis=1)].astype(object)
    labels[x.getnnz(axis=1) == 0] = NO_PRED
    return labels, proba.max(axis=1)


def _count_labels(labels: np.ndarray, conf: np.ndarray, acc: dict) -> None:
    for lab in np.unique(labels):
        m = labels == lab
        cur = acc.setdefault(str(lab), [0, 0.0])
        cur[0] += int(m.sum())
        cur[1] += float(conf[m].sum())


def _nlp_frame(acc: dict) -> pd.DataFrame:
    df = pd.DataFrame([(k, v[0], v[1] / v[0]) for k, v in acc.items()], columns=["label", "count", "mean_confidence"])
    return df.sort_values("label").reset_index(drop=True)


NLP_BATCH = 50_000


# ================================================================== PANDAS
def ads_pandas(files, **_) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f, columns=ADS_COLS) for f in files], ignore_index=True)
    g = df.groupby(["platform", "campaign"], observed=True).agg(
        rows=("impressions", "size"), impressions=("impressions", "sum"), clicks=("clicks", "sum"),
        spend=("spend", "sum"), conversions=("conversions", "sum"), revenue=("revenue", "sum")).reset_index()
    return _tidy_ads(g)


def nlp_pandas(files, model_path: str, **_) -> pd.DataFrame:
    model, acc = load_model(model_path), {}
    for f in files:
        texts = pd.read_parquet(f, columns=["text"])["text"].tolist()
        for i in range(0, len(texts), NLP_BATCH):
            _count_labels(*score_texts(texts[i:i + NLP_BATCH], model), acc)
    return _nlp_frame(acc)


# ================================================================== MAPREDUCE
def _ads_map(task):
    """MAP: read one row-group, handle records one by one, combine locally: key (platform, campaign) -> sums."""
    path, rg = task
    t = pq.ParquetFile(path).read_row_group(rg, columns=ADS_COLS)
    acc: dict = {}
    for p, c, imp, clk, spend, conv, rev in zip(*[t[col].to_pylist() for col in ADS_COLS]):
        v = acc.get((p, c))
        if v is None:
            acc[(p, c)] = [1, imp, clk, spend, conv, rev]
        else:
            v[0] += 1
            v[1] += imp
            v[2] += clk
            v[3] += spend
            v[4] += conv
            v[5] += rev
    return acc


def _nlp_map(task):
    path, rg, model_path = task
    model = load_model(model_path)
    texts = pq.ParquetFile(path).read_row_group(rg, columns=["text"])["text"].to_pylist()
    acc: dict = {}
    for i in range(0, len(texts), NLP_BATCH):
        _count_labels(*score_texts(texts[i:i + NLP_BATCH], model), acc)
    return acc


def run_mapreduce(tasks, map_fn, workers: int = WORKERS, n_reducers: int = 4) -> dict:
    """MAP in parallel -> SHUFFLE (partition by hash of key) -> REDUCE (add up partial sums for each key)."""
    buckets: list[list] = [[] for _ in range(n_reducers)]
    if workers > 1 and len(tasks) > 1:
        # 'spawn' (fresh interpreters) instead of 'fork': forking after Spark has started would copy its open JVM connection
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            mapped = list(pool.imap_unordered(map_fn, tasks))
    else:
        mapped = [map_fn(t) for t in tasks]
    for out in mapped:                                         # shuffle
        for key, val in out.items():
            buckets[hash(key) % n_reducers].append((key, val))
    result: dict = {}
    for bucket in buckets:                                     # reduce
        for key, val in bucket:
            cur = result.get(key)
            if cur is None:
                result[key] = list(val)
            else:
                for j, x in enumerate(val):
                    cur[j] += x
    return result


def ads_mapreduce(files, workers: int = WORKERS, **_) -> pd.DataFrame:
    res = run_mapreduce(row_groups(files), _ads_map, workers)
    df = pd.DataFrame([(p, c, *v) for (p, c), v in res.items()], columns=["platform", "campaign"] + ADS_SUMS)
    return _tidy_ads(df)


def nlp_mapreduce(files, model_path: str, workers: int = WORKERS, **_) -> pd.DataFrame:
    res = run_mapreduce([(f, rg, model_path) for f, rg in row_groups(files)], _nlp_map, workers)
    return _nlp_frame(res)


# ================================================================== SPARK
_SPARK = None
spark_startup_seconds = None


def get_spark():
    """Create (once) a local Spark session. Its start-up time is recorded separately from the job timings."""
    global _SPARK, spark_startup_seconds
    if _SPARK is None:
        t0 = time.perf_counter()
        warnings.filterwarnings("ignore", category=FutureWarning, module="pyspark")
        os.environ["PYSPARK_PYTHON"] = sys.executable                         # workers use this interpreter
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])   # workers can import our code
        from pyspark.sql import SparkSession
        _SPARK = (SparkSession.builder.master(f"local[{WORKERS}]").appName("adbudget")
                  .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", str(WORKERS * 4))
                  .config("spark.driver.memory", "3g").config("spark.sql.files.maxPartitionBytes", str(2 * 1024 * 1024))
                  .config("spark.sql.session.timeZone", "UTC").config("spark.sql.execution.arrow.pyspark.enabled", "true")
                  .getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
        spark_startup_seconds = time.perf_counter() - t0
    return _SPARK


def stop_spark() -> None:
    global _SPARK
    if _SPARK is not None:
        _SPARK.stop()
        _SPARK = None


def ads_spark(files, **_) -> pd.DataFrame:
    from pyspark.sql import functions as F
    spark = get_spark()
    df = spark.read.parquet(*[str(f) for f in files]).select(*ADS_COLS)
    out = df.groupBy("platform", "campaign").agg(
        F.count(F.lit(1)).alias("rows"), F.sum("impressions").alias("impressions"), F.sum("clicks").alias("clicks"),
        F.sum("spend").alias("spend"), F.sum("conversions").alias("conversions"), F.sum("revenue").alias("revenue"))
    return _tidy_ads(out.toPandas())


def _spark_score_batches(model_path: str):
    def run(batches):
        model = load_model(model_path)
        for pdf in batches:
            acc: dict = {}
            texts = pdf["text"].tolist()
            for i in range(0, len(texts), NLP_BATCH):
                _count_labels(*score_texts(texts[i:i + NLP_BATCH], model), acc)
            yield pd.DataFrame([(k, v[0], v[1]) for k, v in acc.items()], columns=["label", "n", "conf_sum"])
    return run


def nlp_spark(files, model_path: str, **_) -> pd.DataFrame:
    from pyspark.sql import functions as F
    spark = get_spark()
    df = spark.read.parquet(*[str(f) for f in files]).select("text")
    part = df.mapInPandas(_spark_score_batches(model_path), schema="label string, n long, conf_sum double")
    out = part.groupBy("label").agg(F.sum("n").alias("count"), (F.sum("conf_sum") / F.sum("n")).alias("mean_confidence"))
    return out.toPandas().sort_values("label").reset_index(drop=True)


ENGINES = {"ads": {"Pandas": ads_pandas, "MapReduce": ads_mapreduce, "Spark": ads_spark},
           "nlp": {"Pandas": nlp_pandas, "MapReduce": nlp_mapreduce, "Spark": nlp_spark}}


# ================================================================== answer check
def same_answer(job: str, a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True when two engines produced the same numbers (integers exactly, money / probabilities to 1e-9 relative)."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for col in a.columns:
        if a[col].dtype == object or str(a[col].dtype) in ("str", "string"):
            if not (a[col].astype(str).values == b[col].astype(str).values).all():
                return False
        elif np.issubdtype(a[col].dtype, np.integer):
            if not (a[col].values == b[col].values.astype(a[col].dtype)).all():
                return False
        elif not np.allclose(a[col].values, b[col].values.astype(float), rtol=1e-9, atol=1e-9):
            return False
    return True
