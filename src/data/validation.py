from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path
import re

import unicodedata

import pandas as pd

TARGET_COLUMNS = [
    "formal_register",
    "thematic_coherence",
    "narrative_rhetorical_structure",
    "cohesion",
]

EXPECTED_ROWS = {"train": 740, "valid": 125, "test": 370}

INPUT_FILENAMES = [
    "train.csv",
    "valid.csv",
    "test.csv",
    "sample_submission.csv",
]


def file_hashes(data_dir: Path) -> dict[str, str]:
    """Return SHA-256 hashes after checking all required input files."""
    missing_files = [
        filename
        for filename in INPUT_FILENAMES
        if not (data_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required input files: {missing_files}"
        )

    return {
        filename: sha256((data_dir / filename).read_bytes()).hexdigest()
        for filename in INPUT_FILENAMES
    }


def load_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load raw train, validation, and test datasets as text columns."""
    file_hashes(data_dir)

    return {
        dataset_name: pd.read_csv(
            data_dir / f"{dataset_name}.csv",
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
        for dataset_name in EXPECTED_ROWS
    }


def _comparison_key(value: object) -> str:
    """Normalize text only for duplicate-essay comparison."""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def _expected_columns(dataset_name: str) -> list[str]:
    columns = ["id", "essay", "prompt"]
    if dataset_name in {"train", "valid"}:
        columns.extend(TARGET_COLUMNS)
    return columns


def _validate_scores(
    dataset_name: str,
    dataset: pd.DataFrame,
    problems: list[str],
) -> None:
    """Validate 1–5 target scores and convert valid columns to integers."""
    if dataset_name not in {"train", "valid"}:
        return

    for target in TARGET_COLUMNS:
        scores: pd.Series = pd.Series(
            pd.to_numeric(dataset[target], errors="coerce")
        )
        invalid_scores = ~scores.isin([1, 2, 3, 4, 5])
        if invalid_scores.any():
            problems.append(
                f"{dataset_name}: {int(invalid_scores.sum())} invalid scores "
                f"in {target}"
            )
            continue
        dataset[target] = scores.astype("int64")


def _dataset_summary(dataset_name: str, dataset: pd.DataFrame) -> dict[str, int | str]:
    """Build the notebook-equivalent structural diagnostic for one split."""
    empty_ids = dataset["id"].astype(str).str.strip().eq("")
    duplicate_ids = dataset["id"].duplicated(keep=False)
    return {
        "dataset": dataset_name,
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "empty_ids": int(empty_ids.sum()),
        "duplicate_id_rows": int(duplicate_ids.sum()),
        "empty_essays": int(dataset["essay"].astype(str).str.strip().eq("").sum()),
        "empty_prompts": int(dataset["prompt"].astype(str).str.strip().eq("").sum()),
        "duplicate_rows": int(dataset.duplicated().sum()),
        "duplicate_essays": int(dataset["essay"].duplicated().sum()),
    }


def _duplicate_essay_report(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report normalized duplicate essays without removing any rows."""
    audit = pd.concat(
        [
            dataset[["id", "essay"]].assign(dataset=dataset_name)
            for dataset_name, dataset in datasets.items()
        ],
        ignore_index=True,
    )
    audit["comparison_key"] = audit["essay"].map(_comparison_key)
    duplicate_rows = audit["comparison_key"].ne(
        "") & audit["comparison_key"].duplicated(keep=False)
    duplicates = audit.loc[duplicate_rows].copy()
    duplicates["group"] = pd.factorize(duplicates["comparison_key"])[0] + 1
    duplicates["across_datasets"] = (
        duplicates.groupby("group")["dataset"].transform("nunique").gt(1)
    )
    return duplicates[["group", "dataset", "id", "across_datasets"]]


def validate_datasets(
    datasets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Validate dataset schemas, integrity, scores, and split boundaries."""
    problems: list[str] = []
    summaries: list[dict[str, int | str]] = []

    for dataset_name, dataset in datasets.items():
        expected_columns = _expected_columns(dataset_name)
        missing_columns = set(expected_columns) - set(dataset.columns)
        extra_columns = set(dataset.columns) - set(expected_columns)
        if missing_columns or extra_columns:
            problems.append(
                f"{dataset_name}: missing columns={sorted(missing_columns)}, "
                f"extra columns={sorted(extra_columns)}"
            )
        if len(dataset) != EXPECTED_ROWS[dataset_name]:
            problems.append(
                f"{dataset_name}: expected {EXPECTED_ROWS[dataset_name]} rows, "
                f"found {len(dataset)}"
            )
        if missing_columns:
            continue

        summary = _dataset_summary(dataset_name, dataset)
        if summary["empty_ids"]:
            problems.append(f"{dataset_name}: contains empty IDs")
        if summary["duplicate_id_rows"]:
            problems.append(f"{dataset_name}: contains duplicate IDs")
        _validate_scores(dataset_name, dataset, problems)
        summaries.append(summary)

    for left_name, right_name in combinations(datasets, 2):
        shared_ids = set(datasets[left_name]["id"]) & set(
            datasets[right_name]["id"])
        if shared_ids:
            problems.append(
                f"Shared IDs between {left_name} and {right_name}: "
                f"{sorted(shared_ids)}"
            )

    duplicate_report = _duplicate_essay_report(datasets)
    if problems:
        raise ValueError("Validation failed:\n- " + "\n- ".join(problems))

    return datasets, pd.DataFrame(summaries), duplicate_report
