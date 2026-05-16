"""
tests/test_preprocessor.py — Unit tests for text preprocessing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.preprocessor import clean_text, lemmatize, preprocess_text, remove_stopwords, tokenize


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO World") == "hello world"

    def test_removes_url(self):
        assert "http" not in clean_text("Visit https://example.com today")

    def test_removes_html(self):
        assert "<b>" not in clean_text("<b>Bold</b> text")

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! It's great.")
        for ch in ",.!'":
            assert ch not in result


class TestTokenize:
    def test_splits_on_whitespace(self):
        tokens = tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_returns_list(self):
        assert isinstance(tokenize("test"), list)


class TestRemoveStopwords:
    def test_removes_the(self):
        assert "the" not in remove_stopwords(["the", "quick", "brown"])

    def test_removes_short_tokens(self):
        assert "a" not in remove_stopwords(["a", "is", "hello"])

    def test_keeps_content_words(self):
        result = remove_stopwords(["excellent", "product", "very"])
        assert "excellent" in result


class TestLemmatize:
    def test_lemmatises_running(self):
        result = lemmatize(["running"])
        assert isinstance(result[0], str) and len(result[0]) >= 3

    def test_lemmatises_better(self):
        result = lemmatize(["better"])
        assert isinstance(result[0], str)

    def test_returns_list(self):
        assert isinstance(lemmatize(["dogs", "cats"]), list)


class TestPreprocessText:
    def test_returns_string(self):
        assert isinstance(preprocess_text("This is a great product!"), str)

    def test_no_punctuation(self):
        result = preprocess_text("Wow! Amazing!!!")
        for ch in "!?,.'":
            assert ch not in result

    def test_empty_string(self):
        # Should not raise; may return empty string
        result = preprocess_text("")
        assert isinstance(result, str)

    def test_real_review(self):
        text  = "The product is absolutely fantastic! Highly recommend."
        result = preprocess_text(text)
        assert "fantastic" in result or "absolut" in result   # stemmed/lemmatised
