from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from ._common import LEXICAL_CONFIG, Representation, fit_and_transform


def build_tf(
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
) -> Representation:
    """Build L2-normalized term-frequency matrices without IDF weighting."""
    vectorizer = TfidfVectorizer(use_idf=False, norm="l2", **LEXICAL_CONFIG)
    return fit_and_transform(
        vectorizer,
        train_texts,
        valid_texts,
        test_texts,
        name="tf",
        family="frequency",
    )
