"""Settings: the simulated advertising business, data scales, and budget-engine limits.

Everything here is simulated, so every number can be traced back to these parameters.
"""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"                       # big parquet files (git-ignored)
OUT_DIR = ROOT / "outputs"                     # small result files used by the dashboard + report
MODEL_PATH = ROOT / "models" / "demo_sentiment_model.joblib"
# Optional: a sentiment model (joblib scikit-learn Pipeline with steps "tfidf" and "clf") and labelled comments
# (CSV with columns text, sentiment) from the Sunscreen Sentiment Analysis project. Used instead of the demo model / synthetic text.
EXTERNAL_MODEL_PATH = Path(os.environ["SENTIMENT_MODEL"]) if os.environ.get("SENTIMENT_MODEL") else None
LABELED_CSV = Path(os.environ["SENTIMENT_LABELED_CSV"]) if os.environ.get("SENTIMENT_LABELED_CSV") else None
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
for _d in (DATA_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 2026
SCALES = {"10K": 10_000, "100K": 100_000, "1M": 1_000_000, "10M": 10_000_000}
ROW_GROUP_ROWS = 250_000       # parquet row-group size = unit of parallel work for MapReduce / Spark
COMMENT_ROW_GROUP_ROWS = 25_000   # text is CPU-heavy per row, so use finer splits to let engines share the work
CHUNK_ROWS = 1_000_000         # rows generated in memory at once
NLP_MAX_SCALE = "1M"           # sentiment-scoring benchmark stops here by default (text cleaning is slow); --nlp-max 10M lifts it

# ---------------------------------------------------------------- the simulated business
START_DATE = date(2025, 11, 3)                 # a Monday; data covers 52 whole weeks
N_WEEKS = 52
THEMES = ["SPF50 Summer Launch", "Oily Skin Gel", "No White Cast Stick"]    # 3 campaigns per platform
PLATFORMS = ["Facebook", "Instagram", "Google", "TikTok", "YouTube"]
ANNUAL_SPEND = 12_000_000                      # Rs spent over the whole year, at every data scale
BUDGET_NOISE = 0.5                             # week-to-week variation of each campaign's budget (log-normal sigma)

# roas   = revenue / spend when the campaign spends its usual weekly budget
# b      = true "diminishing returns" exponent: revenue grows like spend**b (b < 1 = each extra rupee earns less)
# share  = share of the historical budget this platform received
PLATFORM_PARAMS = {
    "Facebook":  dict(cpm=420, ctr=0.011, aov=1800, roas=2.0, b=0.80, share=0.28),
    "Instagram": dict(cpm=520, ctr=0.009, aov=1900, roas=2.5, b=0.75, share=0.22),
    "Google":    dict(cpm=900, ctr=0.030, aov=1800, roas=3.6, b=0.70, share=0.20),
    "TikTok":    dict(cpm=300, ctr=0.008, aov=1500, roas=1.3, b=0.90, share=0.15),
    "YouTube":   dict(cpm=450, ctr=0.006, aov=1700, roas=1.8, b=0.85, share=0.15),
}
SEASON_AMPLITUDE = 0.35        # sunscreen converts better in summer (peak mid-June)
CITIES = {"Karachi": 0.27, "Lahore": 0.24, "Islamabad": 0.11, "Rawalpindi": 0.08, "Faisalabad": 0.08,
          "Multan": 0.07, "Peshawar": 0.08, "Quetta": 0.07}
DEVICES = {"mobile": (0.60, 1.05, 0.95), "desktop": (0.30, 0.90, 1.15), "tablet": (0.10, 0.95, 0.90)}   # share, ctr x, cvr x
GENDERS = {"female": (0.55, 1.00, 1.08), "male": (0.43, 1.00, 0.92), "other": (0.02, 1.00, 1.00)}
AGE_GROUPS = {"18-24": (18, 24, 0.25, 1.20, 0.85), "25-34": (25, 34, 0.35, 1.05, 1.10),
              "35-44": (35, 44, 0.22, 0.95, 1.10), "45-64": (45, 64, 0.18, 0.75, 0.95)}   # lo, hi, share, ctr x, cvr x

# ---------------------------------------------------------------- budget recommendation engine
BUDGET = 1_000_000             # Rs to allocate ("Where should we spend the next Rs. 1 million?")
HORIZON_WEEKS = 4              # the budget is spent evenly over this many weeks
MIN_SHARE, MAX_SHARE = 0.05, 0.50
HISTORY_CAP = 1.25             # by default no platform is planned above 125% of the highest weekly spend seen in history
HOLDOUT_WEEKS = 10             # last weeks kept aside to test the response curves

# ROAS bands used to label campaigns
ROAS_BANDS = [(3.0, "Scale up"), (2.0, "Maintain"), (1.0, "Optimise"), (0.0, "Cut / rework")]
