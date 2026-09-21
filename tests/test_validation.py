import pandas as pd
import pytest
from src.data import validation


def _datasets() -> dict[str, pd.DataFrame]:
    targets = {target: ["1", "2"] for target in validation.TARGET_COLUMNS}
    return {
        "train": pd.DataFrame(
            {
                "id": ["train-1", "train-2"],
                "essay": ["A", "B"],
                "prompt": ["P", "P"],
                **targets,
            }
        ),
        "valid": pd.DataFrame(
            {
                "id": ["valid-1"],
                "essay": ["C"],
                "prompt": ["P"],
                **{target: ["3"] for target in validation.TARGET_COLUMNS},
            }
        ),
        "test": pd.DataFrame({"id": ["test-1"], "essay": ["D"], "prompt": ["P"]}),
    }


def test_validate_datasets_converts_scores_and_returns_reports(monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    datasets, diagnostics, duplicates = validation.validate_datasets(_datasets())

    assert datasets["train"]["formal_register"].dtype == "int64"
    assert diagnostics["rows"].tolist() == [2, 1, 1]
    assert duplicates.empty


def test_validate_datasets_rejects_shared_ids(monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    datasets = _datasets()
    datasets["valid"].loc[0, "id"] = "train-1"

    with pytest.raises(ValueError, match="Shared IDs"):
        validation.validate_datasets(datasets)


def test_validate_datasets_rejects_invalid_scores(monkeypatch):
    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    datasets = _datasets()
    datasets["train"].loc[0, "cohesion"] = "9"

    with pytest.raises(ValueError, match="invalid scores"):
        validation.validate_datasets(datasets)
