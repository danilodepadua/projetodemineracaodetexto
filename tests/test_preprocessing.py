import joblib
from src.data.cleaning import clean_dataset, clean_text
from src.representations import build_representation


def test_cleaning_preserves_accents_and_punctuation():
    assert clean_text("[T] Olá, mãe! [P] [X] texto <i>") == "Olá, mãe!\ntexto <i>"


def test_metadata_and_empty_text():
    frame = clean_dataset(__import__("pandas").DataFrame({"essay": ["[P] oi", "[X]"]}))
    assert frame.paragraph_marker_count.tolist() == [1, 0]
    assert frame.clean_text_empty.tolist() == [False, True]


def test_train_only_fit_and_reload(tmp_path):
    result = build_representation(
        "tfidf", ["Alpha alpha", "Alpha beta"], ["validationonly"], ["testonly"]
    )
    assert result.vectorizer.vocabulary_ is not None
    assert "validationonly" not in result.vectorizer.vocabulary_

    path = tmp_path / "vectorizer.joblib"
    joblib.dump(result.vectorizer, path)
    reloaded = joblib.load(path).transform(["Alpha"])
    assert reloaded.shape is not None
    assert result.train.shape is not None
    assert reloaded.shape[1] == result.train.shape[1]
