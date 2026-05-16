"""
feature_extractor.py — TF-IDF vectorisation + VADER sentiment scores.
Falls back to rule-based scoring if vader_lexicon is unavailable.
"""
import os, re, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import MAX_FEATURES, NGRAM_RANGE, VECTORIZER_PATH

# ── VADER (with graceful fallback) ───────────────────────────────────────────
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _VADER_OK = True
except Exception:
    _VADER_OK = False

_POS_WORDS = {"love","great","excellent","fantastic","amazing","wonderful","best",
              "outstanding","superb","brilliant","happy","perfect","recommend","good",
              "fast","quality","value","impressed","delighted","pleased"}
_NEG_WORDS = {"terrible","awful","horrible","bad","poor","disappointing","hate","broken",
              "useless","waste","worst","damaged","scam","disgusting","cheap","returned",
              "avoid","incomplete","defective","refused"}


def _simple_scores(text: str) -> dict:
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg or 1
    compound = (pos - neg) / total
    compound = max(-1.0, min(1.0, compound))
    return {"neg": neg/total, "neu": 1 - abs(compound), "pos": pos/total, "compound": compound}


# ─── TF-IDF ──────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, sublinear_tf=True)


def fit_transform_tfidf(train_texts: pd.Series):
    vec = build_tfidf_vectorizer()
    X   = vec.fit_transform(train_texts)
    joblib.dump(vec, VECTORIZER_PATH)
    print(f"[feature_extractor] Vectoriser saved → {VECTORIZER_PATH}")
    return X, vec


def transform_tfidf(texts: pd.Series, vec: TfidfVectorizer) -> np.ndarray:
    return vec.transform(texts)


def load_vectorizer() -> TfidfVectorizer:
    return joblib.load(VECTORIZER_PATH)


# ─── VADER ───────────────────────────────────────────────────────────────────

def get_vader_scores(texts: pd.Series) -> pd.DataFrame:
    if _VADER_OK:
        scores = texts.apply(lambda t: _VADER.polarity_scores(str(t)))
    else:
        scores = texts.apply(lambda t: _simple_scores(str(t)))
    return pd.DataFrame(list(scores))


def vader_label(compound: float) -> str:
    if compound >= 0.05:  return "Positive"
    if compound <= -0.05: return "Negative"
    return "Neutral"
