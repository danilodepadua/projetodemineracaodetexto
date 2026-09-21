from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass(frozen=True)
class Representation:
    """A fitted vectorizer and its train, validation, and test matrices."""

    vectorizer: CountVectorizer | TfidfVectorizer
    train: csr_matrix
    valid: csr_matrix
    test: csr_matrix


def build_representation(
    name: str,
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
) -> Representation:
    """Fit a named representation on train texts and transform other splits."""
    if name == "tf":
        vectorizer: CountVectorizer | TfidfVectorizer = CountVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
        )
    elif name == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
        )
    else:
        raise ValueError(f"Unsupported representation: {name}")

    train = csr_matrix(vectorizer.fit_transform(train_texts))
    valid = csr_matrix(vectorizer.transform(valid_texts))
    test = csr_matrix(vectorizer.transform(test_texts))
    return Representation(vectorizer=vectorizer, train=train, valid=valid, test=test)
