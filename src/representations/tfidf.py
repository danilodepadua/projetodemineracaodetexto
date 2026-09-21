from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from ._common import LEXICAL_CONFIG, Representation, fit_and_transform


def build_tfidf(
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
) -> Representation:
    """Build TF-IDF matrices with train-only vocabulary and IDF fitting."""
    vectorizer = TfidfVectorizer(**LEXICAL_CONFIG)
    return fit_and_transform(
        vectorizer,
        train_texts,
        valid_texts,
        test_texts,
        name="tfidf",
        family="frequency",
    )
