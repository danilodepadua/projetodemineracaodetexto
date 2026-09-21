import json

import numpy as np
import pandas as pd
from src.data import validation
from src.preprocessing import run


def _write_inputs(data_dir):
    targets = {target: ["1", "2"] for target in validation.TARGET_COLUMNS}
    pd.DataFrame(
        {
            "id": ["train-1", "train-2"],
            "essay": ["[T] Alpha [P] beta", "Alpha beta"],
            "prompt": ["P", "P"],
            **targets,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": ["valid-1"],
            "essay": ["Alpha gamma"],
            "prompt": ["P"],
            **{target: ["3"] for target in validation.TARGET_COLUMNS},
        }
    ).to_csv(data_dir / "valid.csv", index=False)
    pd.DataFrame({"id": ["test-1"], "essay": ["Alpha delta"], "prompt": ["P"]}).to_csv(
        data_dir / "test.csv", index=False
    )
    pd.DataFrame(
        {"id": ["test-1"], **{target: ["1"] for target in validation.TARGET_COLUMNS}}
    ).to_csv(data_dir / "sample_submission.csv", index=False)


def test_run_writes_clean_data_representations_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_inputs(data_dir)

    run_dir = run(data_dir, tmp_path / "artifacts")
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert (run_dir / "train_clean.csv").is_file()
    for name in ("bow", "tf", "tfidf"):
        assert (run_dir / name / "vectorizer.joblib").is_file()
        assert (run_dir / name / "X_train.npz").is_file()
        assert (run_dir / name / "X_valid.npz").is_file()
        assert (run_dir / name / "X_test.npz").is_file()

        metadata = manifest["representations"][name]
        assert metadata["name"] == name
        assert metadata["vectorizer_type"]
        assert metadata["configuration"]
        assert metadata["vocabulary_size"] == metadata["feature_count"]
        assert metadata["matrix_shapes"]["train"][0] == 2
        assert metadata["artifact_paths"]["vectorizer"]

    assert manifest["representations"]["tfidf"]["shapes"]["train"][0] == 2
    assert manifest["representations"]["tfidf"]["feature_count"] > 0
    assert manifest["input_hashes"]["train.csv"]

    structural = manifest["representations"]["structural"]
    assert structural["name"] == "structural"
    assert structural["feature_count"] == len(structural["feature_names"])
    assert structural["artifact_paths"]["feature_names"]
    for split, rows in {"train": 2, "valid": 1, "test": 1}.items():
        matrix_path = run_dir / "structural" / f"X_{split}.npy"
        matrix = np.load(matrix_path)
        assert matrix.shape == (rows, structural["feature_count"])
        assert np.isfinite(matrix).all()


def test_run_writes_word2vec_artifacts_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_inputs(data_dir)

    run_dir = run(
        data_dir,
        tmp_path / "artifacts",
        representation="word2vec",
        word2vec_architecture="skipgram",
    )
    metadata = json.loads((run_dir / "manifest.json").read_text())["representations"][
        "word2vec"
    ]

    assert (run_dir / "word2vec" / "model.model").is_file()
    for split in ("train", "valid", "test"):
        assert (run_dir / "word2vec" / f"X_{split}.npy").is_file()
        assert (
            metadata["shapes"][split][0] == {"train": 2, "valid": 1, "test": 1}[split]
        )
    assert metadata["architecture"] == "skipgram"
    assert metadata["sg"] == 1
    assert metadata["vector_size"] == 100
    assert metadata["aggregation"] == "mean"
    assert metadata["vocabulary_size"] == 2
    assert metadata["coverage"]["train"]["token_coverage"] == 1.0
    assert metadata["coverage"]["valid"]["zero_vector_documents"] == 0
    assert metadata["artifact_paths"]["model"]


def test_run_writes_explicit_structural_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_inputs(data_dir)

    run_dir = run(data_dir, tmp_path / "artifacts", representation="structural")
    metadata = json.loads((run_dir / "manifest.json").read_text())["representations"][
        "structural"
    ]
    feature_names = json.loads(
        (run_dir / "structural" / "feature_names.json").read_text()
    )

    assert feature_names == metadata["feature_names"]
    assert metadata["feature_definitions"]
    assert metadata["scaling"] == "none"
    assert metadata["matrix_shapes"] == {
        "train": [2, metadata["feature_count"]],
        "valid": [1, metadata["feature_count"]],
        "test": [1, metadata["feature_count"]],
    }


def test_run_writes_essay_prompt_artifacts_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_inputs(data_dir)

    run_dir = run(data_dir, tmp_path / "artifacts", representation="essay_prompt")
    metadata = json.loads((run_dir / "manifest.json").read_text())["representations"][
        "essay_prompt"
    ]
    feature_names = json.loads(
        (run_dir / "essay_prompt" / "feature_names.json").read_text()
    )

    assert metadata["name"] == "essay_prompt"
    assert metadata["feature_count"] == len(feature_names) == 6
    assert metadata["feature_names"] == feature_names
    assert metadata["tfidf"]["fit_scope"] == "train essay_clean only"
    assert metadata["word2vec"]["fit_scope"] == "train essay_clean tokens only"
    assert metadata["artifact_paths"]["tfidf_vectorizer"]
    assert metadata["artifact_paths"]["word2vec_cbow"]
    assert metadata["artifact_paths"]["word2vec_skipgram"]

    for split, rows in {"train": 2, "valid": 1, "test": 1}.items():
        matrix = np.load(run_dir / "essay_prompt" / f"X_{split}.npy")
        assert matrix.shape == (rows, 6)
        assert np.isfinite(matrix).all()
