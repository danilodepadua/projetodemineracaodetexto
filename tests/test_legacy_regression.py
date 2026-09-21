from pathlib import Path

import joblib
import pandas as pd
import pytest
from scipy import sparse
from src.data.cleaning import clean_dataset
from src.data.validation import load_datasets
from src.representations import build_representation

RAW_DATA_DIR = Path("data/raw")
LEGACY_DATA_DIR = Path("data")
REQUIRED_FILES = [
    RAW_DATA_DIR / "train.csv",
    RAW_DATA_DIR / "valid.csv",
    RAW_DATA_DIR / "test.csv",
    LEGACY_DATA_DIR / "train_limpo.csv",
    LEGACY_DATA_DIR / "valid_limpo.csv",
    LEGACY_DATA_DIR / "test_limpo.csv",
    LEGACY_DATA_DIR / "tfidf_vectorizer.joblib",
]

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED_FILES),
    reason="Local reference data and legacy artifacts are required.",
)


def test_cleaned_text_matches_legacy_artifacts():
    datasets = load_datasets(RAW_DATA_DIR)
    marker_columns = {
        "paragraph_marker_count": "n_marcadores_paragrafo",
        "title_marker_count": "n_marcadores_titulo",
        "erasure_marker_count": "n_marcadores_rasura",
        "symbol_marker_count": "n_marcadores_simbolo",
        "unknown_marker_count": "n_marcadores_desconhecido",
        "out_of_line_marker_count": "n_marcadores_fora_da_linha",
        "undocumented_marker_count": "n_marcadores_nao_documentados",
    }

    for name, dataset in datasets.items():
        current = clean_dataset(dataset)
        legacy = pd.read_csv(LEGACY_DATA_DIR / f"{name}_limpo.csv")
        assert current["essay_clean"].tolist() == legacy["essay_clean"].tolist()
        for current_column, legacy_column in marker_columns.items():
            assert current[current_column].tolist() == legacy[legacy_column].tolist()


def test_tfidf_matches_legacy_vectorizer_and_matrices():
    datasets = {
        name: clean_dataset(dataset)
        for name, dataset in load_datasets(RAW_DATA_DIR).items()
    }
    current = build_representation(
        "tfidf",
        datasets["train"]["essay_clean"].tolist(),
        datasets["valid"]["essay_clean"].tolist(),
        datasets["test"]["essay_clean"].tolist(),
    )
    legacy_vectorizer = joblib.load(LEGACY_DATA_DIR / "tfidf_vectorizer.joblib")

    assert current.vectorizer.get_params() == legacy_vectorizer.get_params()
    assert current.vectorizer.vocabulary_ == legacy_vectorizer.vocabulary_

    for split, matrix in {
        "train": current.train,
        "valid": current.valid,
        "test": current.test,
    }.items():
        legacy = sparse.load_npz(LEGACY_DATA_DIR / f"X_{split}_tfidf.npz")
        assert matrix.shape == legacy.shape
        assert matrix.nnz == legacy.nnz
        assert (matrix != legacy).nnz == 0
