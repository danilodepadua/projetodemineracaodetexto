from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
Matrix = csr_matrix | np.ndarray


@dataclass(frozen=True)
class Representation:
    """A fitted representation and its train, validation, and test matrices."""

    vectorizer: Any
    train: Any
    valid: Any
    test: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_and_transform(
    vectorizer: Vectorizer,
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
    name: str | None = None,
    family: str | None = None,
) -> Representation:
    """Fit on train texts and transform validation and test texts."""
    train: Any = csr_matrix(vectorizer.fit_transform(train_texts))
    valid: Any = csr_matrix(vectorizer.transform(valid_texts))
    test: Any = csr_matrix(vectorizer.transform(test_texts))
    metadata: dict[str, Any] = {
        "name": name,
        "family": family,
        "storage": "sparse",
        "format": "npz",
        "dtype": str(train.dtype),
        "matrix_shapes": {
            "train": list(train.shape),
            "valid": list(valid.shape),
            "test": list(test.shape),
        },
        "feature_dimension": int(train.shape[1]),
    }
    return Representation(
        vectorizer=vectorizer,
        train=train,
        valid=valid,
        test=test,
        metadata=metadata,
    )
