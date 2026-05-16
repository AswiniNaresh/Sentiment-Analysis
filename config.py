"""
Configuration settings for the Sentiment Analysis project.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
SRC_DIR     = os.path.join(BASE_DIR, "src")

RAW_DATA_PATH      = os.path.join(DATA_DIR, "raw_reviews.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_reviews.csv")
MODEL_PATH         = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH    = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ─── NLTK Resources ───────────────────────────────────────────────────────────
NLTK_RESOURCES = [
    "punkt",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "omw-1.4",
]

# ─── Text Preprocessing ───────────────────────────────────────────────────────
STOP_WORDS_LANG    = "english"
MIN_TOKEN_LENGTH   = 2
MAX_FEATURES       = 10_000   # TF-IDF vocabulary size
NGRAM_RANGE        = (1, 2)   # unigrams + bigrams

# ─── Model ────────────────────────────────────────────────────────────────────
TEST_SIZE          = 0.20
RANDOM_STATE       = 42
CV_FOLDS           = 5

# ─── Labels ───────────────────────────────────────────────────────────────────
SENTIMENT_LABELS   = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS       = {"Negative": "#ef4444", "Neutral": "#f59e0b", "Positive": "#22c55e"}
