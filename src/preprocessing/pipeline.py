from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse

from ..data.cleaning import MARKERS, clean_dataset, marker_inventory
from ..data.validation import file_hashes, load_datasets, validate_datasets
from ..representations import build_representation
from ..representations.bert import BertConfig


def _write_clean_datasets(run_dir: Path, datasets: dict[str, pd.DataFrame]) -> None:
    """Write cleaned CSV files and immediately verify their round trip."""
    for name, dataset in datasets.items():
        path = run_dir / f"{name}_clean.csv"
        dataset.to_csv(path, index=False, encoding="utf-8")
        reloaded = pd.read_csv(path, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(reloaded, dataset.astype(str))


def _write_reports(
    reports_dir: Path,
    datasets: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> None:
    """Persist validation and marker audit reports."""
    diagnostics.to_csv(reports_dir / "validation_diagnostics.csv", index=False)
    duplicates.to_csv(reports_dir / "duplicate_essays.csv", index=False)
    marker_inventory(datasets).to_csv(reports_dir / "marker_inventory.csv", index=False)


def _texts(dataset: pd.DataFrame) -> list[str]:
    """Return clean essay text as a concrete list for vectorizers."""
    return [str(text) for text in dataset["essay_clean"]]


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    """Return a concrete shape for sparse and dense matrices."""
    shape = matrix.shape
    if shape is None:
        raise ValueError("Matrix shape is unavailable")
    return shape


def _write_representation(
    run_dir: Path,
    name: str,
    datasets: dict[str, pd.DataFrame],
    word2vec_architecture: str = "cbow",
    bert_model: str = BertConfig().model_name,
    bert_device: str = "auto",
    bert_batch_size: int = 4,
) -> dict[str, Any]:
    """Build, persist, and describe one train-fitted representation."""
    if name in {"structural", "essay_prompt"}:
        inputs = (
            datasets["train"],
            datasets["valid"],
            datasets["test"],
        )
    else:
        inputs = (
            _texts(datasets["train"]),
            _texts(datasets["valid"]),
            _texts(datasets["test"]),
        )
    built = build_representation(
        name,
        *inputs,
        word2vec_architecture=word2vec_architecture,
        bert_config=BertConfig(
            model_name=bert_model,
            device=bert_device,
            batch_size=bert_batch_size,
        ),
    )
    representation_dir = run_dir / name
    representation_dir.mkdir()
    matrices = {"train": built.train, "valid": built.valid, "test": built.test}
    artifact_paths: dict[str, str] = {}

    for split, matrix in matrices.items():
        if sparse.issparse(matrix):
            path = representation_dir / f"X_{split}.npz"
            sparse.save_npz(path, matrix)
            reloaded = sparse.load_npz(path)
        else:
            path = representation_dir / f"X_{split}.npy"
            np.save(path, matrix)
            reloaded = np.load(path)
        if _matrix_shape(reloaded) != _matrix_shape(matrix):
            raise AssertionError(f"Serialized {name} {split} matrix shape changed")
        artifact_paths[f"X_{split}"] = str(Path(name) / path.name)

    shapes = {split: list(_matrix_shape(matrix)) for split, matrix in matrices.items()}
    if name == "bert":
        metadata = dict(built.metadata or {})
        metadata.update(
            {
                "name": name,
                "matrix_shapes": shapes,
                "artifact_paths": artifact_paths,
            }
        )
        return metadata

    if name == "word2vec":
        model_path = representation_dir / "model.model"
        built.vectorizer.save(str(model_path))
        metadata = dict(built.metadata or {})
        metadata.update(
            {
                "name": name,
                "model_type": type(built.vectorizer).__name__,
                "model_artifact": str(Path(name) / model_path.name),
                "matrix_shapes": shapes,
                "artifact_paths": {
                    **artifact_paths,
                    "model": str(Path(name) / model_path.name),
                },
            }
        )
        return metadata

    if name in {"structural", "essay_prompt"}:
        metadata = dict(built.metadata or {})
        feature_names_path = representation_dir / "feature_names.json"
        feature_names_path.write_text(
            json.dumps(metadata["feature_names"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if name == "structural":
            metadata.update(
                {
                    "name": name,
                    "matrix_shapes": shapes,
                    "artifact_paths": {
                        **artifact_paths,
                        "feature_names": str(Path(name) / feature_names_path.name),
                    },
                }
            )
            return metadata

        state_paths = {
            "tfidf_vectorizer": representation_dir / "tfidf_vectorizer.joblib",
            "word2vec_cbow": representation_dir / "word2vec_cbow.model",
            "word2vec_skipgram": representation_dir / "word2vec_skipgram.model",
        }
        joblib.dump(built.vectorizer["tfidf"], state_paths["tfidf_vectorizer"])
        built.vectorizer["word2vec_cbow"].save(str(state_paths["word2vec_cbow"]))
        built.vectorizer["word2vec_skipgram"].save(
            str(state_paths["word2vec_skipgram"])
        )
        state_artifact_paths = {
            key: str(Path(name) / path.name) for key, path in state_paths.items()
        }
        metadata.update(
            {
                "name": name,
                "matrix_shapes": shapes,
                "artifact_paths": {
                    **artifact_paths,
                    "feature_names": str(Path(name) / feature_names_path.name),
                    **state_artifact_paths,
                },
            }
        )
        return metadata

    vectorizer_path = representation_dir / "vectorizer.joblib"
    joblib.dump(built.vectorizer, vectorizer_path)
    parameters = built.vectorizer.get_params()
    return {
        "name": name,
        "vectorizer_type": type(built.vectorizer).__name__,
        "configuration": parameters,
        "parameters": parameters,
        "vocabulary_size": _matrix_shape(built.train)[1],
        "feature_count": _matrix_shape(built.train)[1],
        "shapes": shapes,
        "matrix_shapes": shapes,
        "artifact_paths": {
            **artifact_paths,
            "vectorizer": str(Path(name) / vectorizer_path.name),
        },
    }


def run(
    data_dir: Path,
    output_dir: Path,
    representation: str = "all",
    word2vec_architecture: str = "cbow",
    bert_model: str = BertConfig().model_name,
    bert_device: str = "auto",
    bert_batch_size: int = 4,
) -> Path:
    """Validate, clean, represent, and persist the competition datasets."""
    hashes = file_hashes(data_dir)
    datasets, diagnostics, duplicates = validate_datasets(load_datasets(data_dir))
    cleaned_datasets = {
        name: clean_dataset(dataset) for name, dataset in datasets.items()
    }
    run_dir = output_dir / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True)

    _write_clean_datasets(run_dir, cleaned_datasets)
    _write_reports(reports_dir, datasets, diagnostics, duplicates)

    representation_names = (
        ["bow", "tf", "tfidf", "structural"]
        if representation == "all"
        else [representation]
    )
    representations = {
        name: _write_representation(
            run_dir,
            name,
            cleaned_datasets,
            word2vec_architecture=word2vec_architecture,
            bert_model=bert_model,
            bert_device=bert_device,
            bert_batch_size=bert_batch_size,
        )
        for name in representation_names
    }
    manifest = {
        "input_hashes": hashes,
        "row_counts": {
            name: len(dataset) for name, dataset in cleaned_datasets.items()
        },
        "documented_markers": MARKERS,
        "clean_text_column": "essay_clean",
        "representations": representations,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse preprocessing command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate, clean, and represent competition essays."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the original competition CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preprocessing"),
        help="Directory where preprocessing runs are written.",
    )
    parser.add_argument(
        "--representation",
        choices=[
            "bow",
            "tf",
            "tfidf",
            "word2vec",
            "structural",
            "essay_prompt",
            "bert",
            "all",
        ],
        default="all",
        help=(
            "Representation to build (default: all; excludes Word2Vec and BERT; "
            "BERT inference is explicit)."
        ),
    )
    parser.add_argument(
        "--word2vec-architecture",
        choices=["cbow", "skipgram"],
        default="cbow",
        help="Word2Vec architecture when --representation word2vec is selected.",
    )
    parser.add_argument(
        "--bert-model",
        default=BertConfig().model_name,
        help="Hugging Face model identifier for --representation bert.",
    )
    parser.add_argument(
        "--bert-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="BERT device policy: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=4,
        help="BERT chunk batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        run(
            args.data_dir,
            args.output_dir,
            args.representation,
            args.word2vec_architecture,
            args.bert_model,
            args.bert_device,
            args.bert_batch_size,
        )
    )
