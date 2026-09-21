import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from src.representations import build_representation

TRAIN = [
    "Alpha Alpha beta",
    "Alpha beta gamma",
    "Alpha beta gamma",
]
VALID = ["Alpha validationonly", "beta gamma"]
TEST = ["testonly beta", "Alpha beta"]


def test_bow_counts_and_train_vocabulary():
    result = build_representation("bow", TRAIN, VALID, TEST)

    assert sparse.issparse(result.train)
    assert result.vectorizer.vocabulary_ == {
        "Alpha": 0,
        "Alpha beta": 1,
        "beta": 2,
        "beta gamma": 3,
        "gamma": 4,
    }
    np.testing.assert_array_equal(result.train.toarray()[0], [2, 1, 1, 0, 0])
    assert "validationonly" not in result.vectorizer.vocabulary_
    assert "testonly" not in result.vectorizer.vocabulary_
    assert result.valid.shape == (2, 5)
    assert result.test.shape == (2, 5)


def test_tf_is_normalized_without_idf():
    bow = build_representation("bow", TRAIN, VALID, TEST)
    tf = build_representation("tf", TRAIN, VALID, TEST)

    assert isinstance(tf.vectorizer, TfidfVectorizer)
    assert tf.vectorizer.use_idf is False
    assert not hasattr(tf.vectorizer, "idf_")
    expected = sparse.csr_matrix(normalize(bow.train, norm="l2"))
    np.testing.assert_allclose(tf.train.toarray(), expected.toarray())


def test_representation_contract_and_serialization(tmp_path):
    for name in ("bow", "tf", "tfidf"):
        result = build_representation(name, TRAIN, VALID, TEST)
        repeated = build_representation(name, TRAIN, VALID, TEST)
        path = tmp_path / f"{name}.joblib"
        joblib.dump(result.vectorizer, path)
        reloaded = joblib.load(path)

        assert result.train.shape[0] == len(TRAIN)
        assert result.valid.shape[0] == len(VALID)
        assert result.test.shape[0] == len(TEST)
        assert result.train.shape[1] == result.valid.shape[1] == result.test.shape[1]
        assert sparse.issparse(result.train)
        assert sparse.issparse(result.valid)
        assert sparse.issparse(result.test)
        assert result.vectorizer.get_params() == repeated.vectorizer.get_params()
        np.testing.assert_allclose(
            reloaded.transform(VALID).toarray(), result.valid.toarray()
        )

    tfidf = build_representation("tfidf", TRAIN, VALID, TEST)
    assert isinstance(tfidf.vectorizer, TfidfVectorizer)
    assert tfidf.vectorizer.idf_ is not None
    assert "validationonly" not in tfidf.vectorizer.vocabulary_
    assert "testonly" not in tfidf.vectorizer.vocabulary_
