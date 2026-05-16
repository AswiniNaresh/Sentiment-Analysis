"""
model.py — Train, evaluate, save, and load the sentiment classifier.
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import CV_FOLDS, MODEL_PATH, RANDOM_STATE, SENTIMENT_LABELS, TEST_SIZE


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model() -> LogisticRegression:
    """Return a configured (unfitted) Logistic Regression classifier."""
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


# ─── Train / evaluate ────────────────────────────────────────────────────────

def train(X, y) -> tuple:
    """
    Split data, run cross-validation, fit on the full training split, and
    evaluate on the held-out test split.

    Returns:
        model, X_test, y_test, y_pred, cv_scores
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()

    # Cross-validation on training data
    print(f"[model] Running {CV_FOLDS}-fold cross-validation …")
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
    print(f"[model] CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[model] Test Accuracy: {acc:.4f}")
    print("\n[model] Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=list(SENTIMENT_LABELS.values())
        )
    )

    return model, X_test, y_test, y_pred, cv_scores


def get_confusion_matrix(y_test, y_pred) -> np.ndarray:
    return confusion_matrix(y_test, y_pred)


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_model(model: LogisticRegression, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)
    print(f"[model] Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> LogisticRegression:
    return joblib.load(path)


# ─── Inference ───────────────────────────────────────────────────────────────

def predict(model: LogisticRegression, X) -> np.ndarray:
    return model.predict(X)


def predict_proba(model: LogisticRegression, X) -> np.ndarray:
    return model.predict_proba(X)
