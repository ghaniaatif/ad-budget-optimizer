# Digital Advertising Budget Optimizer (Big Data Analytics, Project 2)

A company advertises a sunscreen on **Facebook, Instagram, Google, TikTok and YouTube**. Management asks:
**"Where should we spend the next Rs. 1 million?"**

This project simulates the advertising history (10 thousand up to **10 million** records), computes the campaign KPIs, compares
**Pandas vs MapReduce vs Spark** at every size, and recommends how to split the budget. A Streamlit dashboard and a Word report present the results.

| Requirement | Where it is |
|---|---|
| Data: date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device (10K, 100K, 1M, 10M) | `adbudget/generate_data.py` |
| CTR, CPC, CPA, conversion rate, ROAS, revenue, campaign performance | `adbudget/kpis.py`, `adbudget/analysis.py` |
| Pandas vs MapReduce vs Spark | `adbudget/engines.py`, `adbudget/benchmark.py` |
| Marketing Budget Recommendation Engine | `adbudget/budget_recommender.py` |
| Dashboard | `app.py` |
| Report | `outputs/Project2_Report.docx` (rebuilt by `adbudget/report.py`) |

This is the follow-on to the *Sunscreen Product Sentiment Analysis* project (Phase 1), submitted as a separate repository: the advertised product is the
same sunscreen, and the Phase 1 sentiment model can be plugged in as the workload for Job B (see below). It runs fully on its own.

All advertising data is **simulated**. The simulated business is the same at every size (about Rs 12 million over 52 weeks, 15 campaigns); a bigger
file is a finer-grained log of it. Diminishing returns and a summer peak in sunscreen sales are built in, so the engine's answer can be checked
against the known truth.

## Run it

Spark needs **Java 17 or newer** (check with `java -version`). Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m adbudget.run_all --quick     # about 1 minute: 10K and 100K records
python -m adbudget.run_all             # all sizes up to 10 million (about 5 minutes on 2 cores)
streamlit run app.py                   # dashboard
python -m pytest tests -q              # 15 tests
```

The `outputs/` folder already contains results from a full run, so `streamlit run app.py` works straight after `pip install`.

Individual steps:

```bash
python -m adbudget.generate_data --scales 10K 1M          # make data (data/, git-ignored, recreated in seconds)
python -m adbudget.benchmark --scales 10K 100K            # Pandas vs MapReduce vs Spark
python -m adbudget.benchmark --nlp-max 10M                # also classify 10M comments (slow, about 15 minutes)
python -m adbudget.analysis --scale 10M                   # KPI tables
python -m adbudget.budget_recommender --budget 1000000 --weeks 4
python -m adbudget.report                                 # rebuild the Word report
```

## What the benchmark compares

Two jobs run on every engine, and each answer is checked against Pandas:

* **Job A (advertising):** sum impressions, clicks, spend, conversions and revenue per platform and campaign. All KPIs follow from these sums.
* **Job B (sentiment):** clean text, TF-IDF, Logistic Regression: classify each sunscreen comment and count the labels. By default a demo model trained
  on synthetic comments is used; set `SENTIMENT_MODEL` (and optionally `SENTIMENT_LABELED_CSV`) to use your own model and comments.

Pandas = in-memory, vectorised. MapReduce = the classic map / shuffle / reduce model in plain Python on all cores (an illustration, not Hadoop).
Spark = PySpark in local mode. Results from the run in `outputs/` (2 CPU cores; your numbers will differ):

| Records | Job A Pandas | Job A MapReduce | Job A Spark | Job B Pandas | Job B MapReduce | Job B Spark |
|---|---|---|---|---|---|---|
| 10K | 0.02 s | 0.03 s | 0.45 s | 0.47 s | 0.55 s | 0.80 s |
| 100K | 0.05 s | 0.34 s | 0.36 s | 4.6 s | 3.4 s | 5.3 s |
| 1M | 0.16 s | 3.3 s | 0.66 s | 46 s | 24 s | 27 s |
| 10M | 1.5 s | 21 s | 3.8 s | not run by default | | |

On one machine Pandas is hard to beat for simple column sums while the data fits in memory; Spark pays a fixed start-up and planning cost and is meant
for data or compute that outgrows one machine; Python MapReduce shows the idea but is slow per record; for the CPU-heavy sentiment job the engines that
use every core are about twice as fast as single-process Pandas. All engines share the same few cores here, so this does **not** show cluster scaling.

## The recommendation engine

1. Learn a **response curve** per platform from weekly campaign history: weekly revenue = A x spend^b with a yearly seasonal term. b < 1 means
   diminishing returns.
2. Choose the split that maximises predicted revenue (equal marginal ROAS across platforms), with a minimum share (5%) and maximum share (50%) per
   platform, and a safeguard: no platform above 125% of its highest weekly spend in history (the curve is not trusted beyond what it has seen).
3. Check it: held-out weeks, a bootstrap of the history, and agreement across dataset sizes.

Recommendation for Rs 1,000,000 over 4 weeks: **Google 46%, Instagram 29%, Facebook 11%, YouTube 9%, TikTok 5%**, predicted about 9% more revenue than an
equal split and 8% more than repeating last year's split (model-based estimates inside a simulation). The dashboard lets you change the budget, number
of weeks, start date, limits and the history safeguard and see the split update.

## Honest limits

* The advertising data is simulated. Real data has attribution delays, overlapping audiences and creative fatigue that this model ignores.
* The fitted b values are within roughly 0.05 to 0.1 of the true ones and slightly too high (the engine is a little optimistic about extra spend).
  Held-out revenue is under-predicted by about 13%, partly because one year of data gives the seasonal term little to learn from. The bootstrap
  shows how far each share could move; Facebook and YouTube are the least certain.
* Platforms planned above their historical weekly spend (Google here) are extrapolations: increase in steps and re-measure.
* With 10K records the advice is noisy (a few conversions per campaign-week); the shares settle from 100K upward.
* One machine, two cores: MapReduce is an in-process illustration and Spark runs locally.

## Folder map

```
app.py                      Streamlit dashboard
adbudget/                   settings, data generator, KPIs, engines, benchmark, budget engine, report
tests/                      pytest tests
outputs/                    results (CSV / JSON), charts and Project2_Report.docx
data/, models/              generated on demand (git-ignored)
```
