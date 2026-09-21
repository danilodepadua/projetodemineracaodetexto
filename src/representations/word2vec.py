from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from gensim.models import Word2Vec

from ._common import Representation

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class Word2VecConfig:
    """Explicit, reproducible configuration for a Word2Vec representation."""

    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    architecture: str = "cbow"
    epochs: int = 20
    workers: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        architecture = self.architecture.lower()
        if architecture not in {"cbow", "skipgram"}:
            raise ValueError("architecture must be 'cbow' or 'skipgram'")
        if self.vector_size < 1 or self.window < 1 or self.min_count < 1:
            raise ValueError("vector_size, window, and min_count must be positive")
        if self.epochs < 1 or self.workers < 1:
            raise ValueError("epochs and workers must be positive")
        object.__setattr__(self, "architecture", architecture)

    @property
    def sg(self) -> int:
        """Return Gensim's architecture flag: 0 for CBOW, 1 for Skip-Gram."""
        return int(self.architecture == "skipgram")

    def parameters(self) -> dict[str, Any]:
        """Return serializable training parameters for manifests and tests."""
        return {
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "sg": self.sg,
            "epochs": self.epochs,
            "workers": self.workers,
            "seed": self.seed,
        }


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text without lowercasing, accent stripping, or stopwords."""
    return _TOKEN_PATTERN.findall(text)


def _tokenize_documents(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize(text) for text in texts]


def fit_word2vec(train_texts: Iterable[str], config: Word2VecConfig) -> Word2Vec:
    """Fit a Word2Vec model on train text only."""
    train_documents = _tokenize_documents(train_texts)
    if not any(train_documents):
        raise ValueError("Word2Vec requires at least one token in train text")

    return Word2Vec(
        sentences=train_documents,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        sg=config.sg,
        epochs=config.epochs,
        workers=config.workers,
        seed=config.seed,
    )


def document_vectors(
    model: Word2Vec, texts: Iterable[str], vector_size: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Mean-pool known model vectors for text documents."""
    return _document_vectors(model, _tokenize_documents(texts), vector_size)


def _document_vectors(
    model: Word2Vec,
    documents: Sequence[Sequence[str]],
    vector_size: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    vectors = np.zeros((len(documents), vector_size), dtype=np.float32)
    known_tokens = 0
    total_tokens = 0
    zero_vector_documents = 0
    vocabulary = model.wv.key_to_index

    for row, tokens in enumerate(documents):
        total_tokens += len(tokens)
        known = [model.wv[token] for token in tokens if token in vocabulary]
        known_tokens += len(known)
        if not known:
            zero_vector_documents += 1
            continue
        vectors[row] = np.mean(np.asarray(known), axis=0, dtype=np.float32)

    coverage = known_tokens / total_tokens if total_tokens else 0.0
    return vectors, {
        "token_count": total_tokens,
        "known_token_count": known_tokens,
        "token_coverage": coverage,
        "zero_vector_documents": zero_vector_documents,
    }


def build_word2vec(
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
    config: Word2VecConfig | None = None,
    architecture: str = "cbow",
) -> Representation:
    """Fit Word2Vec on train text and mean-pool all three document splits."""
    config = config or Word2VecConfig(architecture=architecture)
    model = fit_word2vec(train_texts, config)
    train, train_stats = document_vectors(model, train_texts, config.vector_size)
    valid, valid_stats = document_vectors(model, valid_texts, config.vector_size)
    test, test_stats = document_vectors(model, test_texts, config.vector_size)
    metadata: dict[str, Any] = {
        "representation": "word2vec",
        "architecture": config.architecture,
        **config.parameters(),
        "vocabulary_size": len(model.wv),
        "aggregation": "mean",
        "coverage": {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
        },
        "shapes": {
            "train": list(train.shape),
            "valid": list(valid.shape),
            "test": list(test.shape),
        },
    }
    return Representation(
        vectorizer=model,
        train=train,
        valid=valid,
        test=test,
        metadata=metadata,
    )
