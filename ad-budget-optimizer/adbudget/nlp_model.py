"""The sentiment model used by the scale test (job B).

If you point SENTIMENT_MODEL at a model from the Sunscreen Sentiment Analysis project, that model is used.
If not, a small demo model with the SAME design (TF-IDF words + word pairs -> Logistic Regression) is trained on
synthetic sunscreen comments, so this project runs on its own.
"""
import json
import logging

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from adbudget import settings as P
from adbudget.generate_data import synthetic_comments

log = logging.getLogger("nlp_model")


def ensure_model(n_train: int = 40_000) -> tuple[str, str]:
    """Return (path to a trained model, "external" or "demo")."""
    if P.EXTERNAL_MODEL_PATH is not None and P.EXTERNAL_MODEL_PATH.exists():
        return str(P.EXTERNAL_MODEL_PATH), "external"
    if not P.MODEL_PATH.exists():
        df = synthetic_comments(n_train, np.random.default_rng(P.SEED))
        x_tr, x_te, y_tr, y_te = train_test_split(df["text"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=42)
        model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                          ("clf", LogisticRegression(max_iter=2000, C=3, class_weight="balanced"))]).fit(x_tr, y_tr)
        pred = model.predict(x_te)
        P.OUT_DIR.mkdir(parents=True, exist_ok=True)
        (P.OUT_DIR / "demo_model_metrics.json").write_text(json.dumps({
            "trained_on": "synthetic sunscreen comments", "n_train": len(x_tr), "n_test": len(x_te),
            "accuracy": round(float(accuracy_score(y_te, pred)), 3),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 3)}, indent=2))
        joblib.dump(model, P.MODEL_PATH)
        log.info("Trained demo sentiment model (no external model given).")
    return str(P.MODEL_PATH), "demo"
