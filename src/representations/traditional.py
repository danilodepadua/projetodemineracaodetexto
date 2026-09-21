from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from ._common import Representation
from .bert import BertConfig, build_bert
from .bow import build_bow
from .catalog import canonicalize_representation
from .essay_prompt import build_essay_prompt
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
    bert_config: BertConfig | None = None,
) -> Representation:
    """Build a canonical representation for text or cleaned datasets."""
    canonical_name = canonicalize_representation(name, word2vec_architecture)
    if canonical_name == "bert":
        if any(
            isinstance(dataset, pd.DataFrame)
            for dataset in (train_texts, valid_texts, test_texts)
        ):
            raise TypeError("bert representation requires cleaned essay text iterables")
        return build_bert(train_texts, valid_texts, test_texts, config=bert_config)
    if canonical_name in {"structural", "essay_prompt"}:
        datasets = (train_texts, valid_texts, test_texts)
        if not all(isinstance(dataset, pd.DataFrame) for dataset in datasets):
            raise TypeError(
                f"{canonical_name} representation requires cleaned DataFrames"
            )
        typed_datasets = tuple(cast(pd.DataFrame, dataset) for dataset in datasets)
        if canonical_name == "structural":
            return build_structural(*typed_datasets)
        return build_essay_prompt(
            *typed_datasets,
            word2vec_config=word2vec_config,
        )
    if canonical_name in {"word2vec_cbow", "word2vec_skipgram"}:
        architecture = canonical_name.removeprefix("word2vec_")
        return build_word2vec(
            train_texts,
            valid_texts,
            test_texts,
            config=word2vec_config,
            architecture=architecture,
        )
    builder = _BUILDER_BY_NAME[canonical_name]
    return builder(train_texts, valid_texts, test_texts)
