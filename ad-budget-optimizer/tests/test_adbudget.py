"""Tests.   Run from the project folder:  python -m pytest tests -q"""
import shutil
from datetime import date

import numpy as np
import pandas as pd
import pytest

from adbudget import budget_recommender as BR
from adbudget import engines as E
from adbudget import generate_data as G
from adbudget import kpis
from adbudget import settings as P


def toy_curves(a=(1000.0, 800.0, 400.0), b=(0.7, 0.8, 0.9)):
    return {n: {"alpha": float(np.log(x)), "b": bb, "b_used": bb, "c_sin": 0.0, "c_cos": 0.0, "sigma2": 0.0,
                "hist_min_weekly_spend": 1.0, "hist_max_weekly_spend": 1e9, "revenue_per_conversion": 1000.0}
            for n, x, bb in zip(["A", "B", "C"], a, b)}


# ---------------------------------------------------------------- KPIs
def test_kpis_from_sums():
    df = kpis.add_kpis(pd.DataFrame({"impressions": [1000], "clicks": [20], "spend": [400.0], "conversions": [4], "revenue": [1600.0]}))
    r = df.iloc[0]
    assert r.ctr == pytest.approx(0.02) and r.cpc == pytest.approx(20) and r.cpa == pytest.approx(100)
    assert r.conversion_rate == pytest.approx(0.2) and r.roas == pytest.approx(4) and r.profit == pytest.approx(1200)


def test_kpis_zero_division_is_nan():
    df = kpis.add_kpis(pd.DataFrame({"impressions": [0], "clicks": [0], "spend": [0.0], "conversions": [0], "revenue": [0.0]}))
    assert df[["ctr", "cpc", "cpa", "conversion_rate", "roas"]].isna().all(axis=None)


def test_verdict_bands():
    assert [kpis.verdict(x) for x in (3.5, 2.5, 1.5, 0.5)] == ["Scale up", "Maintain", "Optimise", "Cut / rework"]


# ---------------------------------------------------------------- budget engine
def test_allocation_spends_whole_budget_within_limits():
    rec = BR.allocate(toy_curves(), budget=1_000_000, weeks=4, plan_start=date(2026, 6, 1), min_share=0.1, max_share=0.7, history_cap=None)
    assert rec["spend"].sum() == pytest.approx(1_000_000)
    assert rec["share_pct"].between(10 - 0.1, 70 + 0.1).all()


def test_better_platform_gets_more_money():
    rec = BR.allocate(toy_curves(a=(2000.0, 800.0, 400.0), b=(0.8, 0.8, 0.8)), 1_000_000, 4, date(2026, 6, 1), 0.0, 1.0, None).set_index("platform")
    assert rec.loc["A", "spend"] > rec.loc["B", "spend"] > rec.loc["C", "spend"]


def test_equal_marginal_roas_when_no_limit_binds():
    # big budget so that rounding to whole thousands of rupees is negligible
    rec = BR.allocate(toy_curves(), 500_000_000, 4, date(2026, 6, 1), 0.0, 1.0, None)
    assert rec["marginal_roas"].max() / rec["marginal_roas"].min() < 1.01


def test_recommended_beats_equal_split():
    c = toy_curves()
    rec = BR.allocate(c, 1_000_000, 4, date(2026, 6, 1), 0.05, 0.9, None)
    best = BR.expected_revenue(c, dict(zip(rec["platform"], rec["spend"])), 4, date(2026, 6, 1))
    equal = BR.expected_revenue(c, {p: 1_000_000 / 3 for p in c}, 4, date(2026, 6, 1))
    assert best > equal


def test_impossible_limits_raise():
    with pytest.raises(ValueError):
        BR.allocate(toy_curves(), 1_000_000, 4, date(2026, 6, 1), min_share=0.4, max_share=0.5)


def test_history_cap_limits_a_platform():
    c = toy_curves()
    c["A"]["hist_max_weekly_spend"] = 100_000
    rec = BR.allocate(c, 1_000_000, 4, date(2026, 6, 1), 0.05, 0.9, history_cap=1.25).set_index("platform")
    assert rec.loc["A", "weekly_spend"] <= 125_000 + 1


# ---------------------------------------------------------------- data generator
def test_generator_schema_and_determinism():
    a = G.ads_chunk(0, 2000, 10_000, np.random.default_rng(1)).to_pandas()
    b = G.ads_chunk(0, 2000, 10_000, np.random.default_rng(1)).to_pandas()
    assert list(a.columns) == ["date", "campaign", "platform", "impressions", "clicks", "spend", "conversions", "revenue",
                               "age", "gender", "city", "device"]
    pd.testing.assert_frame_equal(a, b)
    assert (a["clicks"] <= a["impressions"]).all() and (a["conversions"] <= a["clicks"]).all()
    assert (a["spend"] >= 0).all() and ((a["conversions"] == 0) <= (a["revenue"] == 0)).all()


def test_synthetic_comments_have_three_classes():
    df = G.synthetic_comments(3000, np.random.default_rng(1))
    assert set(df["sentiment"]) == {"positive", "neutral", "negative"} and df["text"].str.len().min() > 5


# ---------------------------------------------------------------- engines give the same answer
def test_pandas_and_mapreduce_agree():
    files = G.ensure_ads("10K")
    assert E.same_answer("ads", E.ads_pandas(files), E.ads_mapreduce(files, workers=2))


@pytest.mark.skipif(shutil.which("java") is None, reason="Java is needed for Spark")
def test_spark_agrees_with_pandas():
    pytest.importorskip("pyspark")
    files = G.ensure_ads("10K")
    try:
        assert E.same_answer("ads", E.ads_pandas(files), E.ads_spark(files))
    finally:
        E.stop_spark()


def test_batch_scoring_labels_and_no_prediction():
    import joblib

    from adbudget import nlp_model
    path, _ = nlp_model.ensure_model()
    labels, conf = E.score_texts(["Love it, no white cast at all", "terrible greasy and sticky", "zzzz qqqq"], joblib.load(path))
    assert list(labels) == ["positive", "negative", E.NO_PRED]
    assert ((conf > 0) & (conf <= 1)).all()


def test_clean_text():
    from adbudget.text_clean import clean_text
    assert clean_text("Soooooo good!! http://x.com @bob #ad 3:45 <b>wow</b>") == "Sooo good!! wow"
