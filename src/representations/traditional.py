from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from ._common import Representation
from .bow import build_bow
from .structural import build_structural
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
    train_texts: Iterable[str] | pd.DataFrame,
    valid_texts: Iterable[str] | pd.DataFrame,
    test_texts: Iterable[str] | pd.DataFrame,
    word2vec_architecture: str = "cbow",
    word2vec_config: Word2VecConfig | None = None,
) -> Representation:
    """Build a named representation for text iterables or cleaned datasets."""
    if name == "structural":
        datasets = (train_texts, valid_texts, test_texts)
        if not all(isinstance(dataset, pd.DataFrame) for dataset in datasets):
            raise TypeError("structural representation requires cleaned DataFrames")
        return build_structural(
            cast(pd.DataFrame, train_texts),
            cast(pd.DataFrame, valid_texts),
            cast(pd.DataFrame, test_texts),
        )
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
