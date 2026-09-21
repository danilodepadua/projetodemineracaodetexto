from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    family: str
    storage: str
    matrix_format: str


REPRESENTATION_CATALOG: dict[str, RepresentationSpec] = {
    "bow": RepresentationSpec("bow", "frequency", "sparse", "npz"),
    "tf": RepresentationSpec("tf", "frequency", "sparse", "npz"),
    "tfidf": RepresentationSpec("tfidf", "frequency", "sparse", "npz"),
    "word2vec_cbow": RepresentationSpec(
        "word2vec_cbow", "static_semantic", "dense", "npy"
    ),
    "word2vec_skipgram": RepresentationSpec(
        "word2vec_skipgram", "static_semantic", "dense", "npy"
    ),
    "structural": RepresentationSpec("structural", "structural", "dense", "npy"),
    "essay_prompt": RepresentationSpec("essay_prompt", "relationship", "dense", "npy"),
    "bert": RepresentationSpec("bert", "contextual", "dense", "npy"),
}

REPRESENTATION_NAMES = tuple(REPRESENTATION_CATALOG)


def canonicalize_representation(name: str, architecture: str = "cbow") -> str:
    """Return the canonical machine-facing representation name."""
    if name == "word2vec":
        name = f"word2vec_{architecture.lower()}"
    if name not in REPRESENTATION_CATALOG:
        available = ", ".join(REPRESENTATION_NAMES)
        raise ValueError(f"Unsupported representation: {name}. Available: {available}")
    return name


__all__ = [
    "REPRESENTATION_CATALOG",
    "REPRESENTATION_NAMES",
    "RepresentationSpec",
    "canonicalize_representation",
]
