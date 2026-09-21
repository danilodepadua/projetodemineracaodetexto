from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

LEXICAL_CONFIG: dict[str, Any] = {
    "lowercase": False,
    "strip_accents": None,
    "stop_words": None,
    "ngram_range": (1, 2),
    "min_df": 2,
}

Vectorizer = CountVectorizer | TfidfVectorizer


@dataclass(frozen=True)
class Representation:
    """A fitted vectorizer and its train, validation, and test matrices."""

    vectorizer: Vectorizer
    train: csr_matrix
    valid: csr_matrix
    test: csr_matrix


def fit_and_transform(
    vectorizer: Vectorizer,
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
) -> Representation:
    """Fit on train texts and transform validation and test texts."""
    train = csr_matrix(vectorizer.fit_transform(train_texts))
    valid = csr_matrix(vectorizer.transform(valid_texts))
    test = csr_matrix(vectorizer.transform(test_texts))
    return Representation(vectorizer=vectorizer, train=train, valid=valid, test=test)
