from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import sparse


def load_split(
    run_dir: Path, split: str, representation: str, targets: list[str] | None = None
):
    """Load one cleaned split and its matching sparse representation matrix."""
    dataset_path = run_dir / f"{split}_clean.csv"
    matrix_path = run_dir / representation / f"X_{split}.npz"
    if not dataset_path.is_file() or not matrix_path.is_file():
        raise FileNotFoundError(
            f"Missing {split} artifacts in preprocessing run: {run_dir}"
        )
    dataset = pd.read_csv(dataset_path)
    matrix = sparse.load_npz(matrix_path)
    if len(dataset) != matrix.shape[0]:
        raise ValueError(
            f"Row mismatch for {split}: {len(dataset)} != {matrix.shape[0]}"
        )
    if targets:
        missing = [target for target in targets if target not in dataset.columns]
        if missing:
            raise ValueError(f"Missing target columns in {dataset_path}: {missing}")
    return dataset, matrix


def validate_run(run_dir: Path, representation: str) -> None:
    """Ensure the requested representation is declared by the run manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Preprocessing manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if representation not in manifest["representations"]:
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
