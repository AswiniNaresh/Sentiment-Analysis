"""
data_loader.py — Load raw data or generate a sample dataset for demo.
"""
import os
import random
import pandas as pd
from src.config import RAW_DATA_PATH, DATA_DIR

# ─── Sample reviews used when no external CSV is provided ────────────────────
_POSITIVE_REVIEWS = [
    "Absolutely fantastic product! Exceeded all my expectations.",
    "Best purchase I've made this year. Highly recommend to everyone.",
    "The quality is outstanding and delivery was super fast.",
    "Works perfectly. Very happy with this purchase.",
    "Brilliant product, great value for money. Five stars!",
    "Amazing customer service and the item is even better in person.",
    "I love this! Will definitely buy again from this seller.",
    "Top-notch quality. My whole family is impressed.",
    "Superb experience from start to finish. Thank you!",
    "This product changed my life. Worth every penny.",
]

_NEUTRAL_REVIEWS = [
    "It's okay. Does what it says on the box.",
    "Average product. Nothing special but no complaints either.",
    "Received on time. Product is as described.",
    "It works fine for my needs, nothing extraordinary.",
    "Decent quality for the price. Could be better.",
    "Not bad, not great. Somewhere in the middle.",
    "The product is functional. Packaging was a bit simple.",
    "It does the job. I have no strong feelings either way.",
    "Reasonable quality. Matches the product description.",
    "Standard item. Arrived promptly and in good condition.",
]

_NEGATIVE_REVIEWS = [
    "Terrible product. Broke after just two days of use.",
    "Very disappointed. This is nothing like the photos.",
    "Poor quality and awful customer support. Avoid!",
    "Complete waste of money. Do not buy this.",
    "Stopped working after a week. Absolutely useless.",
    "The worst product I have ever purchased. Returning immediately.",
    "Cheap materials and shoddy craftsmanship. Very unhappy.",
    "Arrived damaged and the seller refused to refund me.",
    "Does not work as advertised. Total scam.",
    "Would give zero stars if I could. Disgusting quality.",
]


def generate_sample_dataset(n: int = 300, save: bool = True) -> pd.DataFrame:
    """
    Generate a balanced synthetic review dataset (Positive / Neutral / Negative).

    Args:
        n:    Total number of rows (split evenly across three classes).
        save: If True, persist to RAW_DATA_PATH.

    Returns:
        pd.DataFrame with columns: review_text, sentiment_label, sentiment_score
    """
    random.seed(42)
    per_class = n // 3

    rows = []
    for review in random.choices(_POSITIVE_REVIEWS, k=per_class):
        rows.append({"review_text": review, "sentiment_label": 2, "sentiment_score": 1.0})
    for review in random.choices(_NEUTRAL_REVIEWS, k=per_class):
        rows.append({"review_text": review, "sentiment_label": 1, "sentiment_score": 0.0})
    for review in random.choices(_NEGATIVE_REVIEWS, k=per_class):
        rows.append({"review_text": review, "sentiment_label": 0, "sentiment_score": -1.0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"[data_loader] Sample dataset saved → {RAW_DATA_PATH}  ({len(df)} rows)")

    return df


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load a CSV dataset.  Falls back to the synthetic dataset when the file
    does not exist.

    Expected columns: review_text, sentiment_label
    """
    if not os.path.exists(path):
        print(f"[data_loader] '{path}' not found — generating synthetic dataset.")
        return generate_sample_dataset()

    df = pd.read_csv(path)
    required = {"review_text", "sentiment_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[data_loader] Missing columns: {missing}")

    print(f"[data_loader] Loaded {len(df)} rows from '{path}'")
    return df
