from __future__ import annotations

from collections import Counter
import re
from typing import Pattern
import unicodedata

import pandas as pd

MARKERS: dict[str, list[str]] = {
    "paragraph": ["[P]", "[ P]", "[P}", "[p]", "{p}"],
    "title": ["[T]", "[t]", "{t}"],
    "erasure": ["[R]", "[X]", "[X~]", r"[X\~]", "[r]", "[x]", "{x}"],
    "symbol": ["[S]", "[s]"],
    "unknown": ["[?]", "{?}", "[?}", "{?]"],
    "out_of_line": ["[LC]", "[LT]", "[lt]"],
}

TOKEN_GROUP = {
    token: group
    for group, tokens in MARKERS.items()
    for token in tokens
}

PATTERNS: dict[str, Pattern[str]] = {
    group: re.compile("|".join(re.escape(token)
                      for token in sorted(tokens, key=len, reverse=True)))
    for group, tokens in MARKERS.items()
}

CANDIDATES: Pattern[str] = re.compile(
    r"[\[{][^\[\]{}\n]{0,30}[\]}]|<[^<>\n]{1,30}>")
WORD_PATTERN: Pattern[str] = re.compile(r"\b\w+\b")


def clean_text(text: str) -> str:
    """Apply the conservative notebook cleaning rules to one essay."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    for group, pattern in PATTERNS.items():
        replacement = "\n" if group == "paragraph" else " "
        normalized = pattern.sub(replacement, normalized)

    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    return re.sub(r" *\n *", "\n", normalized).strip()


def _as_text(value: object) -> str:
    """Convert a pandas cell to text before passing it to regex functions."""
    return str(value)


def _markers(value: object) -> list[str]:
    return CANDIDATES.findall(_as_text(value))


def _marker_count(pattern: Pattern[str], value: object) -> int:
    return len(pattern.findall(_as_text(value)))


def _word_count(value: object) -> int:
    return len(WORD_PATTERN.findall(_as_text(value)))


def marker_inventory(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count documented and undocumented markers before cleaning."""
    rows: list[dict[str, str | int]] = []

    for dataset_name, frame in datasets.items():
        counts = Counter(
            marker
            for essay in frame["essay"]
            for marker in _markers(essay)
        )
        for token, occurrences in sorted(counts.items()):
            rows.append(
                {
                    "dataset": dataset_name,
                    "token": token,
                    "occurrences": occurrences,
                    "group": TOKEN_GROUP.get(token, "undocumented"),
                }
            )

    return pd.DataFrame(rows, columns=["dataset", "token", "occurrences", "group"])


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with cleaned essay text and notebook-derived metadata."""
    result = frame.copy(deep=True)
    essays = frame["essay"]

    for group, pattern in PATTERNS.items():
        result[f"{group}_marker_count"] = essays.map(
            lambda essay: _marker_count(pattern, essay)
        )

    result["undocumented_marker_count"] = essays.map(
        lambda essay: sum(
            marker not in TOKEN_GROUP for marker in _markers(essay))
    )
    result["essay_clean"] = essays.map(
        lambda essay: clean_text(_as_text(essay)))
    result["clean_text_empty"] = result["essay_clean"].eq("")
    result["character_count"] = result["essay_clean"].str.len()
    result["word_count"] = result["essay_clean"].map(_word_count)

    cleaned_again = result["essay_clean"].map(
        lambda essay: clean_text(_as_text(essay)))
    if not result["essay_clean"].equals(cleaned_again):
        raise AssertionError("Cleaning is not idempotent")

    return result
