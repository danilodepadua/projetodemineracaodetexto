from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..data.schema import SPLITS, TARGET_COLUMNS
from ..representations.catalog import canonicalize_representation


@dataclass(frozen=True)
class RepresentationDataset:
    """Native train, validation, and test matrices plus manifest metadata."""

    train: sparse.spmatrix | np.ndarray
    valid: sparse.spmatrix | np.ndarray
    test: sparse.spmatrix | np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExperimentData:
    """Stable model-team handoff with native matrices and independent targets."""

    X_train: sparse.spmatrix | np.ndarray
    y_train: dict[str, np.ndarray]
    X_valid: sparse.spmatrix | np.ndarray
    y_valid: dict[str, np.ndarray]
    X_test: sparse.spmatrix | np.ndarray
    metadata: dict[str, Any]


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessing manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid preprocessing manifest: {path}") from error
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("representations"), dict
    ):
        raise ValueError(f"Manifest has no representations mapping: {path}")
    return manifest


def _load_matrix(path: Path, storage: str) -> sparse.spmatrix | np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Representation matrix not found: {path}")
    try:
        matrix = sparse.load_npz(path) if storage == "sparse" else np.load(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load representation matrix: {path}") from error
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError(f"Representation matrix must be 2-D: {path}")
    return matrix


def _id_hash(ids: list[str]) -> str:
    return sha256("\\n".join(ids).encode("utf-8")).hexdigest()


def _validate_alignment(
    run_dir: Path,
    manifest: dict[str, Any],
    split: str,
    dataset: pd.DataFrame,
    rows: int,
) -> None:
    alignment = manifest.get("row_alignment", {}).get(split)
    if not isinstance(alignment, dict):
        raise ValueError(f"Missing row-alignment metadata for split {split}")
    ids_path = run_dir / alignment.get("path", "")
    if not ids_path.is_file():
        raise FileNotFoundError(f"Row-alignment IDs not found: {ids_path}")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    current_ids = [str(value) for value in dataset["id"]]
    if ids != current_ids:
        raise ValueError(f"ID alignment mismatch for split {split}")
    if len(ids) != rows or alignment.get("count") != rows:
        raise ValueError(f"Row alignment count mismatch for split {split}")
    if alignment.get("sha256") != _id_hash(ids):
        raise ValueError(f"Row alignment hash mismatch for split {split}")


def load_representation(run_dir: Path, representation: str) -> RepresentationDataset:
    """Load and validate one canonical representation without densifying it."""
    run_dir = Path(run_dir)
    manifest = _manifest(run_dir)
    name = canonicalize_representation(representation)
    entry = manifest["representations"].get(name)
    if entry is None and name == "word2vec_cbow":
        entry = manifest["representations"].get("word2vec")
    if not isinstance(entry, dict):
        raise ValueError(
            f"Representation {representation!r} is not available in {run_dir}"
        )
    storage = entry.get("storage")
    if storage not in {"sparse", "dense"}:
        raise ValueError(f"Invalid storage metadata for representation {name}")
    expected_format = "npz" if storage == "sparse" else "npy"
    if entry.get("format") != expected_format:
        raise ValueError(f"Invalid matrix format metadata for representation {name}")

    matrices: dict[str, sparse.spmatrix | np.ndarray] = {}
    feature_dimension: int | None = None
    for split in SPLITS:
        dataset_path = run_dir / f"{split}_clean.csv"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Cleaned dataset not found: {dataset_path}")
        dataset = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)
        artifact = entry.get("artifact_paths", {}).get(f"X_{split}")
        if not isinstance(artifact, str):
            raise ValueError(f"Missing X_{split} artifact metadata for {name}")
        path = run_dir / artifact
        matrix = _load_matrix(path, storage)
        matrices[split] = matrix
        if len(dataset) != matrix.shape[0]:
            raise ValueError(
                f"Row mismatch for {name}/{split}: {len(dataset)} != {matrix.shape[0]}"
            )
        if feature_dimension is None:
            feature_dimension = int(matrix.shape[1])
        elif feature_dimension != matrix.shape[1]:
            raise ValueError(f"Feature-dimension mismatch for representation {name}")
        expected_shape = entry.get("matrix_shapes", {}).get(split)
        if expected_shape != list(matrix.shape):
            raise ValueError(f"Metadata shape mismatch for {name}/{split}")
        _validate_alignment(run_dir, manifest, split, dataset, matrix.shape[0])
        expected_hash = entry.get("artifact_hashes", {}).get(f"X_{split}")
        if expected_hash and expected_hash != sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Artifact hash mismatch for {name}/{split}")

    if entry.get("feature_dimension") != feature_dimension:
        raise ValueError(
            f"Feature dimension metadata mismatch for representation {name}"
        )
    expected_dtype = entry.get("dtype")
    train_matrix: Any = matrices["train"]
    if expected_dtype and str(train_matrix.dtype) != expected_dtype:
        raise ValueError(f"Dtype metadata mismatch for representation {name}")
    if storage == "sparse" and not sparse.issparse(matrices["train"]):
        raise ValueError(f"Sparse representation {name} was not loaded as sparse")
    if storage == "dense" and sparse.issparse(matrices["train"]):
        raise ValueError(f"Dense representation {name} was loaded as sparse")
    return RepresentationDataset(
        train=matrices["train"],
        valid=matrices["valid"],
        test=matrices["test"],
        metadata=entry,
    )


def load_experiment_data(
    run_dir: Path,
    representation: str,
    targets: list[str] | None = None,
) -> ExperimentData:
    """Load native matrices and independent train/validation target arrays."""
    requested_targets = targets or TARGET_COLUMNS
    loaded = load_representation(run_dir, representation)
    frames = {
        split: pd.read_csv(
            Path(run_dir) / f"{split}_clean.csv", dtype=str, keep_default_na=False
        )
        for split in SPLITS
    }
    missing = [target for target in requested_targets if target not in frames["train"]]
    if missing or any(target not in frames["valid"] for target in requested_targets):
        raise ValueError(f"Missing target columns in handoff: {missing}")
    return ExperimentData(
        X_train=loaded.train,
        y_train={
            target: np.asarray(pd.to_numeric(frames["train"][target]), dtype=np.int64)
            for target in requested_targets
        },
        X_valid=loaded.valid,
        y_valid={
            target: np.asarray(pd.to_numeric(frames["valid"][target]), dtype=np.int64)
            for target in requested_targets
        },
        X_test=loaded.test,
        metadata=loaded.metadata,
    )


def load_split(
    run_dir: Path, split: str, representation: str, targets: list[str] | None = None
):
    """Compatibility wrapper returning one validated split and its DataFrame."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    loaded = load_representation(run_dir, representation)
    dataset_path = Path(run_dir) / f"{split}_clean.csv"
    dataset = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)
    if targets:
        missing = [target for target in targets if target not in dataset.columns]
        if missing:
            raise ValueError(f"Missing target columns in {dataset_path}: {missing}")
    return dataset, getattr(loaded, split)


def validate_run(run_dir: Path, representation: str) -> None:
    """Ensure the requested representation is declared by the run manifest."""
    manifest = _manifest(Path(run_dir))
    name = canonicalize_representation(representation)
    available = manifest["representations"]
    if name not in available and not (
        name == "word2vec_cbow" and "word2vec" in available
    ):
        raise ValueError(
            f"Representation {representation!r} is not available in {run_dir}"
        )


def resolve_run_dir(
    run_dir: str | Path, runs_dir: Path = Path("artifacts/preprocessing")
) -> Path:
    """Resolve an explicit run directory or the newest timestamped run."""
    if str(run_dir) != "latest":
        return Path(run_dir)
    candidates = [
        path
        for path in runs_dir.glob("run-*")
        if path.is_dir() and (path / "manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No preprocessing runs found in: {runs_dir}")
    return max(candidates, key=lambda path: path.name)
