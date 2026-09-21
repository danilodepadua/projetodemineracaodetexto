from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from hashlib import sha256
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
from ..representations.catalog import (
    REPRESENTATION_CATALOG,
    canonicalize_representation,
)


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
    artifact_name: str | None = None,
) -> dict[str, Any]:
    """Build, persist, and describe one canonical representation."""
    if name in {"structural", "essay_prompt"}:
        inputs = (datasets["train"], datasets["valid"], datasets["test"])
    else:
        inputs = tuple(_texts(datasets[split]) for split in ("train", "valid", "test"))
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
    artifact_name = artifact_name or name
    artifact_prefix = Path(artifact_name)
    representation_dir = run_dir / artifact_name
    representation_dir.mkdir()
    matrices = {"train": built.train, "valid": built.valid, "test": built.test}
    artifact_paths: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}

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
        key = f"X_{split}"
        artifact_paths[key] = str(artifact_prefix / path.name)
        artifact_hashes[key] = sha256(path.read_bytes()).hexdigest()

    shapes = {split: list(_matrix_shape(matrix)) for split, matrix in matrices.items()}
    storage = "sparse" if sparse.issparse(built.train) else "dense"
    metadata = dict(built.metadata or {})
    metadata.update(
        {
            "name": name,
            "family": REPRESENTATION_CATALOG[name].family,
            "storage": storage,
            "format": REPRESENTATION_CATALOG[name].matrix_format,
            "dtype": str(built.train.dtype),
            "feature_dimension": int(_matrix_shape(built.train)[1]),
            "shapes": shapes,
            "matrix_shapes": shapes,
            "artifact_paths": artifact_paths,
            "artifact_hashes": artifact_hashes,
        }
    )

    if name == "bert":
        return metadata

    if name.startswith("word2vec_"):
        model_path = representation_dir / "model.model"
        built.vectorizer.save(str(model_path))
        metadata.update(
            {
                "model_type": type(built.vectorizer).__name__,
                "model_artifact": str(artifact_prefix / model_path.name),
                "artifact_paths": {
                    **artifact_paths,
                    "model": str(artifact_prefix / model_path.name),
                },
                "artifact_hashes": {
                    **artifact_hashes,
                    "model": sha256(model_path.read_bytes()).hexdigest(),
                },
            }
        )
        return metadata

    if name in {"structural", "essay_prompt"}:
        feature_names_path = representation_dir / "feature_names.json"
        feature_names_path.write_text(
            json.dumps(metadata["feature_names"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata["artifact_paths"]["feature_names"] = str(
            artifact_prefix / feature_names_path.name
        )
        metadata["artifact_hashes"]["feature_names"] = sha256(
            feature_names_path.read_bytes()
        ).hexdigest()
        if name == "essay_prompt":
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
            for key, path in state_paths.items():
                metadata["artifact_paths"][key] = str(artifact_prefix / path.name)
                metadata["artifact_hashes"][key] = sha256(path.read_bytes()).hexdigest()
        return metadata

    vectorizer_path = representation_dir / "vectorizer.joblib"
    joblib.dump(built.vectorizer, vectorizer_path)
    parameters = built.vectorizer.get_params()
    metadata.update(
        {
            "vectorizer_type": type(built.vectorizer).__name__,
            "configuration": parameters,
            "parameters": parameters,
            "vocabulary_size": int(_matrix_shape(built.train)[1]),
            "feature_count": int(_matrix_shape(built.train)[1]),
            "artifact_paths": {
                **metadata["artifact_paths"],
                "vectorizer": str(artifact_prefix / vectorizer_path.name),
            },
            "artifact_hashes": {
                **metadata["artifact_hashes"],
                "vectorizer": sha256(vectorizer_path.read_bytes()).hexdigest(),
            },
        }
    )
    return metadata


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

    row_ids_dir = run_dir / "row_ids"
    row_ids_dir.mkdir()
    row_alignment: dict[str, dict[str, Any]] = {}
    for split, dataset in cleaned_datasets.items():
        ids = [str(value) for value in dataset["id"]]
        ids_path = row_ids_dir / f"{split}.json"
        ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        row_alignment[split] = {
            "path": str(Path("row_ids") / ids_path.name),
            "count": len(ids),
            "sha256": sha256("\\n".join(ids).encode("utf-8")).hexdigest(),
        }

    representation_names = (
        ["bow", "tf", "tfidf", "structural"]
        if representation == "all"
        else [canonicalize_representation(representation, word2vec_architecture)]
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
            artifact_name=("word2vec" if representation == "word2vec" else None),
        )
        for name in representation_names
    }
    if representation == "word2vec":
        legacy_name = canonicalize_representation(representation, word2vec_architecture)
        representations["word2vec"] = representations.pop(legacy_name)
    manifest = {
        "input_hashes": hashes,
        "row_counts": {
            name: len(dataset) for name, dataset in cleaned_datasets.items()
        },
        "documented_markers": MARKERS,
        "clean_text_column": "essay_clean",
        "row_alignment": row_alignment,
        "representations": representations,
        "representation_catalog": {
            name: {
                "family": spec.family,
                "storage": spec.storage,
                "format": spec.matrix_format,
            }
            for name, spec in REPRESENTATION_CATALOG.items()
        },
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
            "word2vec_cbow",
            "word2vec_skipgram",
            "word2vec",
            "structural",
            "essay_prompt",
            "bert",
            "all",
        ],
        default="all",
        help=(
            "Representation to build (default: all; builds BoW, TF, TF-IDF, and "
            "structural; expensive representations are explicit)."
        ),
    )
    parser.add_argument(
        "--word2vec-architecture",
        choices=["cbow", "skipgram"],
        default="cbow",
        help=(
            "Word2Vec architecture for legacy --representation word2vec; canonical "
            "names are word2vec_cbow and word2vec_skipgram."
        ),
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
