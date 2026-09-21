from __future__ import annotations

from collections.abc import Iterable

from ._common import Representation
from .bow import build_bow
from .tf import build_tf
from .tfidf import build_tfidf
from .word2vec import Word2VecConfig, build_word2vec

_BUILDER_BY_NAME = {
    "bow": build_bow,
    "tf": build_tf,
    "tfidf": build_tfidf,
}


def build_representation(
    name: str,
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
    word2vec_architecture: str = "cbow",
    word2vec_config: Word2VecConfig | None = None,
) -> Representation:
    """Fit a named representation on train texts and transform other splits."""
    if name == "word2vec":
        return build_word2vec(
            train_texts,
            valid_texts,
            test_texts,
            config=word2vec_config,
            architecture=word2vec_architecture,
        )
    try:
        builder = _BUILDER_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"Unsupported representation: {name}") from error
    return builder(train_texts, valid_texts, test_texts)
