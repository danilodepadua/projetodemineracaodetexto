from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ._common import Representation

_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_TERMINAL_PATTERN = re.compile(r"[.!?]+")

STRUCTURAL_FEATURE_NAMES = (
    "character_count",
    "word_count",
    "sentence_count",
    "paragraph_count",
    "unique_word_count",
    "lexical_diversity",
    "average_word_length",
    "average_sentence_length_words",
    "average_paragraph_length_words",
    "punctuation_count",
    "comma_count",
    "sentence_terminal_count",
    "uppercase_ratio",
    "digit_count",
    "paragraph_marker_count",
    "title_marker_count",
    "erasure_marker_count",
    "symbol_marker_count",
    "unknown_marker_count",
    "out_of_line_marker_count",
    "undocumented_marker_count",
)

_MARKER_COLUMNS = STRUCTURAL_FEATURE_NAMES[14:]
_REQUIRED_COLUMNS = {"essay_clean", "character_count", "word_count", *_MARKER_COLUMNS}

FEATURE_DEFINITIONS: dict[str, str] = {
    "character_count": "Number of characters in canonical essay_clean.",
    "word_count": "Number of Unicode word tokens in canonical essay_clean.",
    "sentence_count": (
        "Terminal-punctuation runs, with one for non-empty text without one."
    ),
    "paragraph_count": "Non-empty lines in canonical essay_clean.",
    "unique_word_count": "Number of distinct Unicode word tokens, preserving case.",
    "lexical_diversity": "unique_word_count divided by word_count, or zero when empty.",
    "average_word_length": "Mean Unicode word-token length, or zero when empty.",
    "average_sentence_length_words": (
        "word_count divided by sentence_count, or zero when empty."
    ),
    "average_paragraph_length_words": (
        "word_count divided by paragraph_count, or zero when empty."
    ),
    "punctuation_count": "Count of non-alphanumeric, non-whitespace characters.",
    "comma_count": "Number of comma characters.",
    "sentence_terminal_count": "Number of runs of '.', '!', or '?' characters.",
    "uppercase_ratio": (
        "Uppercase alphabetic characters divided by alphabetic characters."
    ),
    "digit_count": "Number of Unicode digit characters.",
    "paragraph_marker_count": (
        "Canonical paragraph-marker count from cleaning metadata."
    ),
    "title_marker_count": "Canonical title-marker count from cleaning metadata.",
    "erasure_marker_count": "Canonical erasure-marker count from cleaning metadata.",
    "symbol_marker_count": "Canonical symbol-marker count from cleaning metadata.",
    "unknown_marker_count": "Canonical unknown-marker count from cleaning metadata.",
    "out_of_line_marker_count": (
        "Canonical out-of-line-marker count from cleaning metadata."
    ),
    "undocumented_marker_count": (
        "Canonical undocumented-marker count from cleaning metadata."
    ),
}


def _paragraphs(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _features(row: Any) -> list[float]:
    text = str(row["essay_clean"])
    tokens = _WORD_PATTERN.findall(text)
    words = len(tokens)
    unique_words = len(set(tokens))
    paragraphs = _paragraphs(text)
    sentence_terminal_count = len(_SENTENCE_TERMINAL_PATTERN.findall(text))
    sentence_count = 0 if not text.strip() else max(1, sentence_terminal_count)
    alphabetic_count = sum(character.isalpha() for character in text)
    uppercase_count = sum(character.isupper() for character in text)

    values: dict[str, float] = {
        "character_count": float(row["character_count"]),
        "word_count": float(row["word_count"]),
        "sentence_count": float(sentence_count),
        "paragraph_count": float(len(paragraphs)),
        "unique_word_count": float(unique_words),
        "lexical_diversity": unique_words / words if words else 0.0,
        "average_word_length": (sum(map(len, tokens)) / words if words else 0.0),
        "average_sentence_length_words": (
            words / sentence_count if sentence_count else 0.0
        ),
        "average_paragraph_length_words": (
            words / len(paragraphs) if paragraphs else 0.0
        ),
        "punctuation_count": float(
            sum(
                not character.isalnum() and not character.isspace()
                for character in text
            )
        ),
        "comma_count": float(text.count(",")),
        "sentence_terminal_count": float(sentence_terminal_count),
        "uppercase_ratio": (
            uppercase_count / alphabetic_count if alphabetic_count else 0.0
        ),
        "digit_count": float(sum(character.isdigit() for character in text)),
    }
    values.update({column: float(row[column]) for column in _MARKER_COLUMNS})
    return [values[name] for name in STRUCTURAL_FEATURE_NAMES]


def _validate_dataset(dataset: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(dataset.columns)
    if missing:
        raise ValueError(
            f"Structural features require cleaned columns: {sorted(missing)}"
        )


def build_structural(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Representation:
    """Build raw dense document features from cleaned essay datasets."""
    datasets: Iterable[pd.DataFrame] = (train_data, valid_data, test_data)
    for dataset in datasets:
        _validate_dataset(dataset)

    matrices = [
        np.asarray([_features(row) for _, row in dataset.iterrows()], dtype=np.float64)
        for dataset in (train_data, valid_data, test_data)
    ]
    shapes = {
        split: list(matrix.shape)
        for split, matrix in zip(("train", "valid", "test"), matrices, strict=True)
    }
    metadata: dict[str, Any] = {
        "representation": "structural",
        "feature_count": len(STRUCTURAL_FEATURE_NAMES),
        "feature_names": list(STRUCTURAL_FEATURE_NAMES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "matrix_shapes": shapes,
        "shapes": shapes,
        "tokenization": (
            r"Unicode \b\w+\b tokens; case, accents, and stopwords preserved."
        ),
        "sentence_strategy": (
            "Count terminal-punctuation runs with a non-empty fallback of one."
        ),
        "paragraph_strategy": "Count non-empty lines in essay_clean.",
        "scaling": "none",
    }
    return Representation(
        vectorizer=None,
        train=matrices[0],
        valid=matrices[1],
        test=matrices[2],
        metadata=metadata,
    )
