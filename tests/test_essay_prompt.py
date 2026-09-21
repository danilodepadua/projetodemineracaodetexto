import numpy as np
import pandas as pd
from src.data.cleaning import clean_dataset
from src.representations import build_representation
from src.representations.essay_prompt import ESSAY_PROMPT_FEATURE_NAMES
from src.representations.word2vec import Word2VecConfig


def _clean(
    essays: list[str], prompts: list[str], **columns: list[object]
) -> pd.DataFrame:
    return clean_dataset(pd.DataFrame({"essay": essays, "prompt": prompts, **columns}))


def _config() -> Word2VecConfig:
    return Word2VecConfig(
        vector_size=6,
        window=2,
        min_count=1,
        epochs=4,
        workers=1,
        seed=7,
    )


def _result(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    return build_representation(
        "essay_prompt",
        train,
        valid,
        test,
        word2vec_config=_config(),
    )


def test_lexical_overlap_features():
    train = _clean(
        ["Alpha beta common", "Alpha gamma common"],
        ["Alpha", "gamma"],
    )
    valid = _clean(
        ["alpha alpha beta", "", "Olá mãe"],
        ["alpha alpha", "", "mãe café"],
    )
    result = _result(train, valid, _clean(["Alpha"], ["future-only"]))
    values = dict(zip(ESSAY_PROMPT_FEATURE_NAMES, result.valid[0], strict=True))
    assert values["shared_token_count"] == 1
    assert values["essay_prompt_token_jaccard"] == 0.5
    assert values["prompt_token_coverage"] == 1.0

    empty = dict(zip(ESSAY_PROMPT_FEATURE_NAMES, result.valid[1], strict=True))
    assert empty["shared_token_count"] == 0
    assert empty["essay_prompt_token_jaccard"] == 0.0
    assert empty["prompt_token_coverage"] == 0.0

    accented = dict(zip(ESSAY_PROMPT_FEATURE_NAMES, result.valid[2], strict=True))
    assert accented["shared_token_count"] == 1
    assert accented["essay_prompt_token_jaccard"] == 1 / 3
    assert accented["prompt_token_coverage"] == 0.5


def test_tfidf_is_train_only_and_finite():
    train = _clean(
        ["Alpha beta common", "Alpha gamma common"],
        ["Alpha", "gamma"],
    )
    valid = _clean(["validationonly"], ["validationonly"])
    test = _clean(["testonly"], ["testonly"])
    result = _result(train, valid, test)
    vectorizer = result.vectorizer["tfidf"]

    assert "validationonly" not in vectorizer.vocabulary_
    assert "testonly" not in vectorizer.vocabulary_
    assert result.valid.shape == (1, len(ESSAY_PROMPT_FEATURE_NAMES))
    assert result.test.shape == (1, len(ESSAY_PROMPT_FEATURE_NAMES))
    assert result.valid[0, 0] == 0.0
    assert result.test[0, 0] == 0.0
    assert np.isfinite(result.train).all()
    assert np.isfinite(result.valid).all()
    assert np.isfinite(result.test).all()


def test_word2vec_architectures_oov_and_zero_vector_fallback():
    train = _clean(
        ["Alpha beta common", "Alpha gamma common"],
        ["Alpha", "gamma"],
    )
    valid = _clean(["future-only", ""], ["future-only", "future-only"])
    test = _clean(["Alpha"], ["Alpha"])
    result = _result(train, valid, test)

    for architecture in ("cbow", "skipgram"):
        model = result.vectorizer[f"word2vec_{architecture}"]
        assert "future-only" not in model.wv.key_to_index
        column = ESSAY_PROMPT_FEATURE_NAMES.index(
            f"word2vec_{architecture}_cosine_similarity"
        )
        assert result.valid[:, column].tolist() == [0.0, 0.0]
        assert np.isfinite(result.train[:, column]).all()
        assert np.isfinite(result.valid[:, column]).all()
        assert np.isfinite(result.test[:, column]).all()


def test_contract_determinism_and_target_independence():
    train = _clean(
        ["Alpha beta common", "Alpha gamma common"],
        ["Alpha", "gamma"],
        thematic_coherence=[1, 5],
    )
    valid = _clean(["Alpha validation"], ["Alpha"])
    test = _clean(["gamma test"], ["gamma"])
    first = _result(train, valid, test)
    second = _result(train, valid, test)

    assert first.metadata["feature_names"] == list(ESSAY_PROMPT_FEATURE_NAMES)
    assert first.train.shape == (2, 6)
    assert first.valid.shape == (1, 6)
    assert first.test.shape == (1, 6)
    np.testing.assert_allclose(first.train, second.train)
    np.testing.assert_allclose(first.valid, second.valid)
    np.testing.assert_allclose(first.test, second.test)

    changed_labels = train.copy()
    changed_labels["thematic_coherence"] = [5, 1]
    changed = _result(changed_labels, valid, test)
    np.testing.assert_allclose(first.train, changed.train)
