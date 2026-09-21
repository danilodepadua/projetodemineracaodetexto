import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from src.data.schema import TARGET_COLUMNS
from src.models.data import load_experiment_data, load_representation
from src.representations.catalog import REPRESENTATION_CATALOG, REPRESENTATION_NAMES


@pytest.mark.parametrize("name", REPRESENTATION_NAMES)
def test_catalog_declares_common_storage_contract(name):
    spec = REPRESENTATION_CATALOG[name]
    assert spec.family
    assert spec.storage in {"sparse", "dense"}
    assert spec.matrix_format in {"npz", "npy"}


def _write_run(tmp_path, name="tfidf", storage="sparse"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = {"train": 2, "valid": 1, "test": 1}
    for split, count in rows.items():
        values = {
            "id": [f"{split}-{index}" for index in range(count)],
            "essay": ["text"] * count,
            "prompt": ["prompt"] * count,
        }
        if split != "test":
            values.update({target: ["1"] * count for target in TARGET_COLUMNS})
        pd.DataFrame(values).to_csv(run_dir / f"{split}_clean.csv", index=False)
    ids_dir = run_dir / "row_ids"
    ids_dir.mkdir()
    alignment = {}
    for split, count in rows.items():
        ids = [f"{split}-{index}" for index in range(count)]
        (ids_dir / f"{split}.json").write_text(json.dumps(ids))
        alignment[split] = {
            "path": f"row_ids/{split}.json",
            "count": count,
            "sha256": hashlib.sha256("\\n".join(ids).encode()).hexdigest(),
        }
    representation_dir = run_dir / name
    representation_dir.mkdir()
    matrices = {}
    for split, count in rows.items():
        matrix = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        if storage == "sparse":
            path = representation_dir / f"X_{split}.npz"
            sparse.save_npz(path, sparse.csr_matrix(matrix))
        else:
            path = representation_dir / f"X_{split}.npy"
            np.save(path, matrix)
        matrices[split] = matrix
    spec = REPRESENTATION_CATALOG[name]
    metadata = {
        "name": name,
        "family": spec.family,
        "storage": storage,
        "format": spec.matrix_format,
        "dtype": "float32",
        "feature_dimension": 3,
        "matrix_shapes": {
            split: list(matrix.shape) for split, matrix in matrices.items()
        },
        "artifact_paths": {
            f"X_{split}": f"{name}/X_{split}.{spec.matrix_format}" for split in rows
        },
    }
    manifest = {
        "row_alignment": alignment,
        "representations": {name: metadata},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


@pytest.mark.parametrize("name,storage", [("tfidf", "sparse"), ("structural", "dense")])
def test_loader_preserves_native_matrix_type_and_alignment(tmp_path, name, storage):
    loaded = load_representation(_write_run(tmp_path, name, storage), name)
    if storage == "sparse":
        assert sparse.issparse(loaded.train)
    else:
        assert isinstance(loaded.train, np.ndarray)
    assert loaded.train.shape == (2, 3)
    assert loaded.valid.shape == (1, 3)
    assert loaded.test.shape == (1, 3)


def test_experiment_handoff_exposes_independent_targets(tmp_path):
    data = load_experiment_data(_write_run(tmp_path), "tfidf")
    assert data.X_train.shape[0] == len(data.y_train["cohesion"])
    assert data.X_valid.shape[0] == len(data.y_valid["formal_register"])
    assert data.X_test.shape == (1, 3)
    assert set(data.y_train) == set(TARGET_COLUMNS)


def test_loader_rejects_id_alignment_mismatch(tmp_path):
    run_dir = _write_run(tmp_path)
    dataset_path = run_dir / "valid_clean.csv"
    dataset = pd.read_csv(dataset_path)
    dataset.loc[0, "id"] = "wrong-id"
    dataset.to_csv(dataset_path, index=False)
    with pytest.raises(ValueError, match="ID alignment mismatch"):
        load_representation(run_dir, "tfidf")
