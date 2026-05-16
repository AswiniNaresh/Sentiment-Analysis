"""
preprocessor.py — Text cleaning and normalisation using NLTK.
Falls back to simple regex-based processing if NLTK corpora are unavailable.
"""
import re
import string
import nltk
import pandas as pd

from src.config import (
    NLTK_RESOURCES,
    STOP_WORDS_LANG,
    MIN_TOKEN_LENGTH,
    PROCESSED_DATA_PATH,
)


def download_nltk_resources() -> None:
    """Download all required NLTK corpora / models (silent if already present)."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            pass   # offline environment — handled gracefully below


download_nltk_resources()

# ── Stopwords ─────────────────────────────────────────────────────────────────
try:
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words(STOP_WORDS_LANG))
except Exception:
    _STOP_WORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
        "it","its","itself","they","them","their","theirs","themselves","what","which",
        "who","whom","this","that","these","those","am","is","are","was","were","be",
        "been","being","have","has","had","having","do","does","did","doing","would",
        "could","should","may","might","will","shall","can","to","of","in","for","on",
        "with","at","by","from","up","about","into","through","during","before","after",
        "above","below","under","over","between","out","off","the","a","an","and","but",
        "or","nor","not","so","yet","both","either","neither","while","if","unless",
        "until","when","where","how","all","each","few","more","most","other","some",
        "such","no","only","same","than","too","very","just","because","although","though",
    }

# ── Lemmatizer ────────────────────────────────────────────────────────────────
try:
    from nltk.stem import WordNetLemmatizer
    _LEMMATIZER = WordNetLemmatizer()
    _LEMMATIZER.lemmatize("test")   # warm-up: triggers corpus load
    _USE_LEMMATIZER = True
except Exception:
    _USE_LEMMATIZER = False

# ── Tokenizer ─────────────────────────────────────────────────────────────────
try:
    from nltk.tokenize import word_tokenize
    word_tokenize("test sentence")
    _word_tokenize = word_tokenize
except Exception:
    _word_tokenize = lambda text: re.findall(r"[a-z]+", text)

_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE   = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")


# ─── Single-text pipeline ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list:
    return _word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= MIN_TOKEN_LENGTH]


def lemmatize(tokens: list) -> list:
    if _USE_LEMMATIZER:
        return [_LEMMATIZER.lemmatize(t) for t in tokens]
    # Minimal suffix-stripping fallback
    def _strip(w):
        for sfx in ("ing", "tion", "ness", "ment", "ful", "less", "er", "ed"):
            if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                return w[: -len(sfx)]
        return w
    return [_strip(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for one document."""
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    tokens  = remove_stopwords(tokens)
    tokens  = lemmatize(tokens)
    return " ".join(tokens)


# ─── DataFrame-level helper ───────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    print("[preprocessor] Preprocessing text …")
    df = df.copy()
    df["processed_text"] = df[text_col].astype(str).apply(preprocess_text)
    df.dropna(subset=["processed_text"], inplace=True)
    df = df[df["processed_text"].str.strip() != ""]
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[preprocessor] Processed data saved → {PROCESSED_DATA_PATH}  ({len(df)} rows)")
    return df
