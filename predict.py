"""
predict.py — Run sentiment prediction on one or more texts.

Usage:
    python predict.py "I absolutely love this product!"
    python predict.py "It's okay." "Terrible quality, very disappointed."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_PATH, SENTIMENT_LABELS, VECTORIZER_PATH
from src.feature_extractor import get_vader_scores, load_vectorizer, vader_label
from src.model import load_model, predict, predict_proba
from src.preprocessor import preprocess_text

import pandas as pd


def analyse(texts: list[str]) -> pd.DataFrame:
    """
    Analyse a list of raw text strings.

    Returns a DataFrame with:
        - text           : original input
        - ml_label       : ML model prediction
        - ml_confidence  : model confidence (%)
        - vader_label    : rule-based VADER prediction
        - vader_compound : VADER compound score
    """
    # Check required artefacts exist
    for path in (MODEL_PATH, VECTORIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run  python train.py  first."
            )

    model = load_model()
    vec   = load_vectorizer()

    processed = [preprocess_text(t) for t in texts]
    X = vec.transform(processed)

    preds  = predict(model, X)
    probas = predict_proba(model, X)

    vader_df = get_vader_scores(pd.Series(texts))

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text":          text,
            "ml_label":      SENTIMENT_LABELS[preds[i]],
            "ml_confidence": f"{probas[i].max() * 100:.1f}%",
            "vader_label":   vader_label(vader_df["compound"].iloc[i]),
            "vader_compound": round(vader_df["compound"].iloc[i], 4),
        })

    return pd.DataFrame(results)


def main() -> None:
    texts = sys.argv[1:] if len(sys.argv) > 1 else [
        "I absolutely love this product! It's amazing.",
        "The item is okay, nothing special.",
        "Terrible quality. Completely broken on arrival.",
    ]

    print("\n" + "=" * 70)
    print("  Sentiment Analysis — Prediction")
    print("=" * 70)

    df = analyse(texts)

    for _, row in df.iterrows():
        print(f"\n  Text      : {row['text']}")
        print(f"  ML Label  : {row['ml_label']}  ({row['ml_confidence']} confidence)")
        print(f"  VADER     : {row['vader_label']}  (compound = {row['vader_compound']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
