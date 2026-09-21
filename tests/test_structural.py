import numpy as np
import pandas as pd
import pytest
from src.data.cleaning import clean_dataset
from src.representations import build_representation
from src.representations.structural import STRUCTURAL_FEATURE_NAMES


def _clean(texts: list[str], **columns: list[object]) -> pd.DataFrame:
    frame = pd.DataFrame({"essay": texts, **columns})
    return clean_dataset(frame)


def _feature_values(text: str) -> dict[str, float]:
    dataset = _clean([text])
    result = build_representation("structural", dataset, dataset, dataset)
    return dict(zip(STRUCTURAL_FEATURE_NAMES, result.train[0], strict=True))


def test_structural_feature_values():
    text = "Olá, mãe! 2 cães.\nOutra frase?"
    values = _feature_values(text)

    assert values["character_count"] == len(text)
    assert values["word_count"] == 6
    assert values["sentence_count"] == 3
    assert values["paragraph_count"] == 2
    assert values["unique_word_count"] == 6
    assert values["lexical_diversity"] == pytest.approx(1.0)
    assert values["average_word_length"] == pytest.approx(21 / 6)
    assert values["average_sentence_length_words"] == pytest.approx(2.0)
    assert values["average_paragraph_length_words"] == pytest.approx(3.0)
    assert values["punctuation_count"] == 4
    assert values["comma_count"] == 1
    assert values["sentence_terminal_count"] == 3
    assert values["uppercase_ratio"] == pytest.approx(2 / 20)
    assert values["digit_count"] == 1


def test_structural_consumes_cleaning_metadata():
    dataset = _clean(["[T] Olá [P] mãe [X]!"])
    values = _feature_values("[T] Olá [P] mãe [X]!")

    assert dataset.loc[0, "essay_clean"] == "Olá\nmãe !"
    assert values["paragraph_marker_count"] == 1
    assert values["title_marker_count"] == 1
    assert values["erasure_marker_count"] == 1
    assert values["symbol_marker_count"] == 0
    assert values["unknown_marker_count"] == 0
    assert values["out_of_line_marker_count"] == 0
    assert values["undocumented_marker_count"] == 0


def test_structural_edge_cases_are_finite():
    dataset = _clean(["", "   ", "um", "Olá, mãe!", "A1, B2?!"])
    result = build_representation("structural", dataset, dataset, dataset)

    assert result.train.shape == (5, len(STRUCTURAL_FEATURE_NAMES))
    assert np.isfinite(result.train).all()
    assert np.all(result.train[0] == 0)
    assert result.train[2, STRUCTURAL_FEATURE_NAMES.index("paragraph_count")] == 1
    assert result.train[4, STRUCTURAL_FEATURE_NAMES.index("digit_count")] == 2


def test_structural_contract_order_and_no_label_leakage():
    train = _clean(["Alpha beta", "Alpha"], formal_register=[1, 5])
    valid = _clean(["validation unique"])
    test = _clean(["test unique"])
    result = build_representation("structural", train, valid, test)
    repeated = build_representation("structural", train, valid, test)

    assert not hasattr(result.vectorizer, "fit")
    assert result.train.shape[0] == len(train)
    assert result.valid.shape[0] == len(valid)
    assert result.test.shape[0] == len(test)
    assert result.train.shape[1] == result.valid.shape[1] == result.test.shape[1]
    assert result.metadata["feature_names"] == list(STRUCTURAL_FEATURE_NAMES)
    np.testing.assert_array_equal(result.train, repeated.train)
    np.testing.assert_array_equal(result.valid, repeated.valid)
    np.testing.assert_array_equal(result.test, repeated.test)

    changed_labels = train.copy()
    changed_labels["formal_register"] = [5, 1]
    changed = build_representation("structural", changed_labels, valid, test)
    np.testing.assert_array_equal(result.train, changed.train)
