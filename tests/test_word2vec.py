import numpy as np
from gensim.models import Word2Vec
from src.data.cleaning import clean_text
from src.representations import build_representation
from src.representations.word2vec import (
    Word2VecConfig,
    _document_vectors,
    tokenize,
)

TRAIN = [
    "Alpha beta beta",
    "Alpha gamma beta",
    "Gamma beta Alpha",
]
VALID = ["Alpha validação exclusiva", "beta gamma"]
TEST = ["teste exclusivo", "Alpha beta"]


def _config(architecture: str = "cbow") -> Word2VecConfig:
    return Word2VecConfig(
        vector_size=8,
        window=2,
        min_count=1,
        architecture=architecture,
        epochs=8,
        workers=1,
        seed=7,
    )


def test_tokenize_is_deterministic_and_compatible_with_cleaned_text():
    cleaned = clean_text("[P] Olá, mãe! [X] café e-mail")

    assert tokenize(cleaned) == ["Olá", "mãe", "café", "e", "mail"]
    assert tokenize("") == []
    assert tokenize(" \t\n ") == []
    assert tokenize(cleaned) == tokenize(cleaned)


def test_word2vec_uses_train_only_vocabulary_and_fixed_dense_shapes():
    result = build_representation(
        "word2vec", TRAIN, VALID, TEST, word2vec_config=_config()
    )

    assert set(result.vectorizer.wv.key_to_index) == {"Alpha", "beta", "Gamma", "gamma"}
    assert "validação" not in result.vectorizer.wv.key_to_index
    assert "exclusiva" not in result.vectorizer.wv.key_to_index
    assert "exclusivo" not in result.vectorizer.wv.key_to_index
    assert result.train.shape == (len(TRAIN), 8)
    assert result.valid.shape == (len(VALID), 8)
    assert result.test.shape == (len(TEST), 8)
    assert np.isfinite(result.train).all()
    assert np.isfinite(result.valid).all()
    assert np.isfinite(result.test).all()


def test_mean_pooling_ignores_oov_and_returns_zero_for_all_oov():
    model = Word2Vec(
        sentences=[["known", "other"]],
        vector_size=2,
        min_count=1,
        workers=1,
        seed=7,
        epochs=1,
    )
    model.wv["known"] = np.array([1.0, 3.0], dtype=np.float32)
    model.wv["other"] = np.array([3.0, 5.0], dtype=np.float32)

    vectors, stats = _document_vectors(
        model,
        [["known", "other", "missing"], ["missing"], []],
        vector_size=2,
    )

    np.testing.assert_allclose(vectors[0], [2.0, 4.0])
    np.testing.assert_array_equal(vectors[1:], np.zeros((2, 2), dtype=np.float32))
    assert stats == {
        "token_count": 4,
        "known_token_count": 2,
        "token_coverage": 0.5,
        "zero_vector_documents": 2,
    }
    assert np.isfinite(vectors).all()


def test_cbow_and_skipgram_map_to_gensim_architectures():
    for architecture, sg in (("cbow", 0), ("skipgram", 1)):
        result = build_representation(
            "word2vec", TRAIN, VALID, TEST, word2vec_config=_config(architecture)
        )
        assert result.vectorizer.sg == sg
        assert result.metadata["architecture"] == architecture
        assert result.metadata["sg"] == sg
        assert result.train.shape[1] == 8


def test_identical_deterministic_runs_match():
    first = build_representation(
        "word2vec", TRAIN, VALID, TEST, word2vec_config=_config("skipgram")
    )
    second = build_representation(
        "word2vec", TRAIN, VALID, TEST, word2vec_config=_config("skipgram")
    )

    np.testing.assert_allclose(first.train, second.train, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.valid, second.valid, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.test, second.test, rtol=1e-6, atol=1e-6)


def test_model_and_vectors_serialize_without_changes(tmp_path):
    result = build_representation(
        "word2vec", TRAIN, VALID, TEST, word2vec_config=_config()
    )
    model_path = tmp_path / "model.model"
    result.vectorizer.save(str(model_path))
    reloaded_model = Word2Vec.load(str(model_path))
    assert set(reloaded_model.wv.key_to_index) == set(result.vectorizer.wv.key_to_index)

    for split, matrix in (
        ("train", result.train),
        ("valid", result.valid),
        ("test", result.test),
    ):
        path = tmp_path / f"X_{split}.npy"
        np.save(path, matrix)
        np.testing.assert_array_equal(np.load(path), matrix)
