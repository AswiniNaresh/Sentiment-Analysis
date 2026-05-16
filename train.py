"""
train.py — End-to-end training pipeline.

Usage:
    python train.py
"""
import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_DIR, OUTPUT_DIR
from src.data_loader import load_data
from src.feature_extractor import fit_transform_tfidf, get_vader_scores
from src.model import get_confusion_matrix, save_model, train
from src.preprocessor import preprocess_dataframe
from src.visualizer import (
    plot_confusion_matrix,
    plot_cv_scores,
    plot_label_distribution,
    plot_vader_distribution,
    plot_wordcloud,
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run() -> None:
    print("=" * 60)
    print("  Sentiment Analysis — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Preprocess
    df = preprocess_dataframe(df)

    # 3. Visualise label distribution
    plot_label_distribution(df["sentiment_label"])
    print("[train] Label distribution plot saved.")

    # 4. VADER scores + distribution plot
    vader_df = get_vader_scores(df["review_text"])
    df["vader_compound"] = vader_df["compound"]
    plot_vader_distribution(df["vader_compound"])
    print("[train] VADER distribution plot saved.")

    # 5. Word cloud on processed text
    plot_wordcloud(df["processed_text"].tolist())
    print("[train] Word cloud saved.")

    # 6. TF-IDF feature extraction
    X, vectorizer = fit_transform_tfidf(df["processed_text"])
    y = df["sentiment_label"].values

    # 7. Train + evaluate
    model, X_test, y_test, y_pred, cv_scores = train(X, y)

    # 8. Confusion matrix + CV plot
    cm = get_confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)
    plot_cv_scores(cv_scores)
    print("[train] Evaluation plots saved.")

    # 9. Persist model
    save_model(model)

    print("\n" + "=" * 60)
    print("  Training complete!  Artefacts saved to:")
    print(f"    Models  → {MODEL_DIR}")
    print(f"    Outputs → {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
