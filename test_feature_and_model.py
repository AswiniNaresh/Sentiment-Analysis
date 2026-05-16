"""
tests/test_feature_extractor.py — Unit tests for feature extraction helpers.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.feature_extractor import (
    build_tfidf_vectorizer,
    get_vader_scores,
    vader_label,
)


class TestTfidfVectorizer:
    def test_fit_transform_shape(self):
        vec    = build_tfidf_vectorizer()
        texts  = pd.Series(["great product", "terrible experience", "okay item"])
        X      = vec.fit_transform(texts)
        assert X.shape[0] == 3

    def test_transform_consistent(self):
        vec   = build_tfidf_vectorizer()
        train = pd.Series(["good product", "bad product"])
        vec.fit_transform(train)
        test  = pd.Series(["good experience"])
        X     = vec.transform(test)
        assert X.shape[0] == 1


class TestVader:
    _texts = pd.Series([
        "I absolutely love this! It is wonderful.",
        "This is terrible, I hate it completely.",
        "The item arrived on time.",
    ])

    def test_returns_dataframe(self):
        result = get_vader_scores(self._texts)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        result = get_vader_scores(self._texts)
        for col in ("neg", "neu", "pos", "compound"):
            assert col in result.columns

    def test_positive_text_high_compound(self):
        scores = get_vader_scores(pd.Series(["I absolutely love this amazing product!"]))
        assert scores["compound"].iloc[0] > 0

    def test_negative_text_low_compound(self):
        scores = get_vader_scores(pd.Series(["This is terrible and I hate everything about it!"]))
        assert scores["compound"].iloc[0] < 0

    def test_vader_label_positive(self):
        assert vader_label(0.6)  == "Positive"

    def test_vader_label_negative(self):
        assert vader_label(-0.6) == "Negative"

    def test_vader_label_neutral(self):
        assert vader_label(0.0)  == "Neutral"


# ─── tests/test_model.py ──────────────────────────────────────────────────────

"""
tests/test_model.py — Unit tests for the model training / inference helpers.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.model import build_model, get_confusion_matrix, predict, predict_proba


class TestBuildModel:
    def test_returns_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        model = build_model()
        assert isinstance(model, LogisticRegression)


class TestModelInference:
    @pytest.fixture(autouse=True)
    def _fit(self):
        X, y = make_classification(
            n_samples=150, n_features=20, n_classes=3,
            n_informative=10, random_state=42
        )
        self.model = build_model()
        self.model.fit(X, y)
        self.X_test = X[:20]
        self.y_test = y[:20]

    def test_predict_shape(self):
        preds = predict(self.model, self.X_test)
        assert preds.shape == (20,)

    def test_predict_proba_shape(self):
        probas = predict_proba(self.model, self.X_test)
        assert probas.shape == (20, 3)

    def test_predict_proba_sums_to_one(self):
        probas = predict_proba(self.model, self.X_test)
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(20), atol=1e-6)

    def test_confusion_matrix_shape(self):
        preds = predict(self.model, self.X_test)
        cm = get_confusion_matrix(self.y_test, preds)
        assert cm.shape == (3, 3)
