from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import CountVectorizer

from ._common import LEXICAL_CONFIG, Representation, fit_and_transform


def build_bow(
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
) -> Representation:
    """Build raw document-term count matrices."""
    vectorizer = CountVectorizer(**LEXICAL_CONFIG)
    return fit_and_transform(
        vectorizer,
        train_texts,
        valid_texts,
        test_texts,
        name="bow",
        family="frequency",
    )
