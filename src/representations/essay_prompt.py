from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.cleaning import clean_text
from ._common import LEXICAL_CONFIG, Representation
from .word2vec import (
    Word2VecConfig,
    document_vectors,
    fit_word2vec,
    tokenize,
)

ESSAY_PROMPT_FEATURE_NAMES = (
    "tfidf_cosine_similarity",
    "word2vec_cbow_cosine_similarity",
    "word2vec_skipgram_cosine_similarity",
    "shared_token_count",
    "essay_prompt_token_jaccard",
    "prompt_token_coverage",
)

FEATURE_DEFINITIONS: dict[str, str] = {
    "tfidf_cosine_similarity": (
        "Cosine similarity between train-fitted TF-IDF essay_clean and "
        "cleaned prompt vectors."
    ),
    "word2vec_cbow_cosine_similarity": (
        "Cosine similarity between mean-pooled CBOW vectors trained on "
        "train essay tokens."
    ),
    "word2vec_skipgram_cosine_similarity": (
        "Cosine similarity between mean-pooled Skip-Gram vectors trained on "
        "train essay tokens."
    ),
    "shared_token_count": (
        "Number of unique Unicode tokens shared by essay_clean and cleaned prompt."
    ),
    "essay_prompt_token_jaccard": (
        "Size of the unique-token intersection divided by the unique-token union."
    ),
    "prompt_token_coverage": (
        "Size of the unique-token intersection divided by the unique prompt-token set."
    ),
}

_REQUIRED_COLUMNS = {"essay_clean", "prompt"}


def _validate_dataset(dataset: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(dataset.columns)
    if missing:
        raise ValueError(f"Essay-prompt features require columns: {sorted(missing)}")


def _prompt_texts(dataset: pd.DataFrame) -> list[str]:
    """Return cleaned prompt views without modifying the source DataFrame."""
    return [clean_text(str(prompt)) for prompt in dataset["prompt"]]


def _essay_texts(dataset: pd.DataFrame) -> list[str]:
    return [str(text) for text in dataset["essay_clean"]]


def _safe_row_cosine(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute row-wise cosine similarity, returning zero for undefined rows."""
    left_norms = np.linalg.norm(left, axis=1)
    right_norms = np.linalg.norm(right, axis=1)
    valid = (left_norms > 0) & (right_norms > 0)
    scores = np.zeros(left.shape[0], dtype=np.float64)
    if valid.any():
        scores[valid] = np.einsum("ij,ij->i", left[valid], right[valid]) / (
            left_norms[valid] * right_norms[valid]
        )
    return scores, int((~valid).sum())


def _sparse_row_cosine(left: Any, right: Any) -> tuple[np.ndarray, int]:
    """Compute sparse row-wise cosine similarity without NaN or Inf."""
    products = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norms = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norms = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    valid = (left_norms > 0) & (right_norms > 0)
    scores = np.zeros(left.shape[0], dtype=np.float64)
    if valid.any():
        scores[valid] = products[valid] / (left_norms[valid] * right_norms[valid])
    return scores, int((~valid).sum())


def _overlap_features(essays: list[str], prompts: list[str]) -> np.ndarray:
    values: list[list[float]] = []
    for essay, prompt in zip(essays, prompts, strict=True):
        essay_tokens = set(tokenize(essay))
        prompt_tokens = set(tokenize(prompt))
        shared = essay_tokens & prompt_tokens
        union = essay_tokens | prompt_tokens
        values.append(
            [
                float(len(shared)),
                len(shared) / len(union) if union else 0.0,
                len(shared) / len(prompt_tokens) if prompt_tokens else 0.0,
            ]
        )
    return np.asarray(values, dtype=np.float64)


def build_essay_prompt(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    word2vec_config: Word2VecConfig | None = None,
) -> Representation:
    """Build dense, target-free features describing each essay-prompt pair."""
    for dataset in (train_data, valid_data, test_data):
        _validate_dataset(dataset)

    datasets = (train_data, valid_data, test_data)
    essay_texts = [_essay_texts(dataset) for dataset in datasets]
    prompt_texts = [_prompt_texts(dataset) for dataset in datasets]

    tfidf = TfidfVectorizer(**LEXICAL_CONFIG)
    tfidf.fit(essay_texts[0])
    tfidf_scores: list[np.ndarray] = []
    tfidf_zero_counts: dict[str, int] = {}
    for split, essays, prompts in zip(
        ("train", "valid", "test"), essay_texts, prompt_texts, strict=True
    ):
        essay_vectors = tfidf.transform(essays)
        prompt_vectors = tfidf.transform(prompts)
        scores, zero_count = _sparse_row_cosine(essay_vectors, prompt_vectors)
        tfidf_scores.append(scores)
        tfidf_zero_counts[split] = zero_count

    base_config = word2vec_config or Word2VecConfig()
    semantic_scores: dict[str, list[np.ndarray]] = {
        "cbow": [],
        "skipgram": [],
    }
    semantic_zero_counts: dict[str, dict[str, int]] = {
        "cbow": {},
        "skipgram": {},
    }
    models: dict[str, Any] = {}
    for architecture in ("cbow", "skipgram"):
        config = replace(base_config, architecture=architecture)
        model = fit_word2vec(essay_texts[0], config=config)
        models[architecture] = model
        for split, essays, prompts in zip(
            ("train", "valid", "test"), essay_texts, prompt_texts, strict=True
        ):
            essay_vectors, _ = document_vectors(model, essays, config.vector_size)
            prompt_vectors, _ = document_vectors(model, prompts, config.vector_size)
            scores, zero_count = _safe_row_cosine(essay_vectors, prompt_vectors)
            semantic_scores[architecture].append(scores)
            semantic_zero_counts[architecture][split] = zero_count

    matrices: list[np.ndarray] = []
    for index, split in enumerate(("train", "valid", "test")):
        overlap = _overlap_features(essay_texts[index], prompt_texts[index])
        matrix = np.column_stack(
            [
                tfidf_scores[index],
                semantic_scores["cbow"][index],
                semantic_scores["skipgram"][index],
                overlap,
            ]
        )
        if not np.isfinite(matrix).all():
            raise AssertionError(f"Non-finite essay-prompt features in {split}")
        matrices.append(matrix)

    shapes = {
        split: list(matrix.shape)
        for split, matrix in zip(("train", "valid", "test"), matrices, strict=True)
    }
    metadata: dict[str, Any] = {
        "representation": "essay_prompt",
        "feature_count": len(ESSAY_PROMPT_FEATURE_NAMES),
        "feature_names": list(ESSAY_PROMPT_FEATURE_NAMES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "tokenization": (
            r"Existing Unicode \b\w+\b tokenization; case, accents, and "
            "stopwords preserved."
        ),
        "prompt_cleaning": (
            "clean_text(prompt) view; source prompt column is unchanged."
        ),
        "tfidf": {
            "fit_scope": "train essay_clean only",
            "configuration": tfidf.get_params(),
            "vocabulary_size": len(tfidf.vocabulary_),
            "zero_norm_comparisons": tfidf_zero_counts,
        },
        "word2vec": {
            "fit_scope": "train essay_clean tokens only",
            "aggregation": "mean over known tokens",
            "architectures": {
                architecture: {
                    **replace(base_config, architecture=architecture).parameters(),
                    "vocabulary_size": len(models[architecture].wv),
                    "zero_vector_comparisons": semantic_zero_counts[architecture],
                }
                for architecture in ("cbow", "skipgram")
            },
        },
        "zero_vector_comparisons": {
            "tfidf": tfidf_zero_counts,
            "word2vec_cbow": semantic_zero_counts["cbow"],
            "word2vec_skipgram": semantic_zero_counts["skipgram"],
        },
        "matrix_shapes": shapes,
        "shapes": shapes,
    }
    return Representation(
        vectorizer={
            "tfidf": tfidf,
            "word2vec_cbow": models["cbow"],
            "word2vec_skipgram": models["skipgram"],
        },
        train=matrices[0],
        valid=matrices[1],
        test=matrices[2],
        metadata=metadata,
    )
