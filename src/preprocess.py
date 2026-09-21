from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import scipy
import sklearn
from scipy import sparse

from .data.cleaning import MARKERS, clean_dataset, marker_inventory
from .data.validation import file_hashes, load_datasets, validate_datasets
from .representations import build_representation


def _write_clean_datasets(run_dir: Path, datasets: dict[str, pd.DataFrame]) -> None:
    """Write cleaned CSV files and immediately verify their round trip."""
    for name, dataset in datasets.items():
        path = run_dir / f"{name}_clean.csv"
        dataset.to_csv(path, index=False, encoding="utf-8")
        reloaded = pd.read_csv(path, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(reloaded, dataset.astype(str))


def _write_reports(reports_dir: Path, datasets: dict[str, pd.DataFrame], diagnostics: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    """Persist validation and marker audit reports."""
    diagnostics.to_csv(reports_dir / "validation_diagnostics.csv", index=False)
    duplicates.to_csv(reports_dir / "duplicate_essays.csv", index=False)
    marker_inventory(datasets).to_csv(reports_dir / "marker_inventory.csv", index=False)


def _texts(dataset: pd.DataFrame) -> list[str]:
    """Return clean essay text as a concrete list for vectorizers."""
    return [str(text) for text in dataset["essay_clean"]]


def _write_representation(run_dir: Path, name: str, datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build, persist, and describe one train-fitted representation."""
    built = build_representation(name, _texts(datasets["train"]), _texts(datasets["valid"]), _texts(datasets["test"]))
    representation_dir = run_dir / name
    representation_dir.mkdir()
    matrices = {"train": built.train, "valid": built.valid, "test": built.test}

    for split, matrix in matrices.items():
        path = representation_dir / f"X_{split}.npz"
        sparse.save_npz(path, matrix)
        if sparse.load_npz(path).shape != matrix.shape:
            raise AssertionError(f"Serialized {name} {split} matrix shape changed")

    joblib.dump(built.vectorizer, representation_dir / "vectorizer.joblib")
    return {"parameters": built.vectorizer.get_params(), "feature_count": built.train.shape[1], "shapes": {split: list(matrix.shape) for split, matrix in matrices.items()}}


def run(data_dir: Path, output_dir: Path, representation: str = "all") -> Path:
    """Validate, clean, represent, and persist the competition datasets."""
    hashes = file_hashes(data_dir)
    datasets, diagnostics, duplicates = validate_datasets(load_datasets(data_dir))
    cleaned_datasets = {name: clean_dataset(dataset) for name, dataset in datasets.items()}
    run_dir = output_dir / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True)

    _write_clean_datasets(run_dir, cleaned_datasets)
    _write_reports(reports_dir, datasets, diagnostics, duplicates)

    representation_names = ["tf", "tfidf"] if representation == "all" else [representation]
    representations = {name: _write_representation(run_dir, name, cleaned_datasets) for name in representation_names}
    manifest = {"input_hashes": hashes, "row_counts": {name: len(dataset) for name, dataset in cleaned_datasets.items()}, "documented_markers": MARKERS, "clean_text_column": "essay_clean", "representations": representations, "python": platform.python_version(), "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse preprocessing command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate, clean, and represent competition essays.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing the original competition CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preprocessing"), help="Directory where preprocessing runs are written.")
    parser.add_argument("--representation", choices=["tf", "tfidf", "all"], default="all", help="Representation to build (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args.data_dir, args.output_dir, args.representation))


if __name__ == "__main__":
    main()
