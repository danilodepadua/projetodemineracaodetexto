import json
from pathlib import Path

import numpy as np
import pytest
from src.models.evaluate import _validate_model_bundle, evaluate_models
from src.models.train import train_models

TARGETS = ["formal_register"]
PARAMETERS = {"alpha": 1.0, "fit_intercept": True}


def _training_data():
    return (
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        {"formal_register": np.asarray([1, 2, 3, 4])},
    )


def test_model_artifacts_are_isolated_by_representation(tmp_path: Path):
    X_train, y_train = _training_data()
    first = train_models(
        X_train,
        y_train,
        "ridge",
        tmp_path / "models",
        TARGETS,
        PARAMETERS,
        "tfidf",
        tmp_path / "runs" / "run-tfidf",
    )
    second = train_models(
        X_train,
        y_train,
        "ridge",
        tmp_path / "models",
        TARGETS,
        PARAMETERS,
        "bert",
        tmp_path / "runs" / "run-bert",
    )

    assert first == tmp_path / "models" / "tfidf" / "ridge"
    assert second == tmp_path / "models" / "bert" / "ridge"
    assert (first / "formal_register.joblib").is_file()
    assert (second / "formal_register.joblib").is_file()
    assert (
        json.loads((first / "metadata.json").read_text())["source_run"] == "run-tfidf"
    )
    assert str(tmp_path) not in (first / "metadata.json").read_text()


def test_evaluation_rejects_mismatched_model_representation(tmp_path: Path):
    model_dir = tmp_path / "models" / "tfidf" / "ridge"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "representation": "tfidf",
                "source_run": "run-tfidf",
                "model": "ridge",
                "targets": TARGETS,
            }
        )
    )

    with pytest.raises(ValueError, match="representation"):
        _validate_model_bundle(
            model_dir,
            "ridge",
            "bert",
            Path("run-bert"),
            TARGETS,
        )


def test_matching_model_bundle_evaluates(tmp_path: Path):
    X_train, y_train = _training_data()
    model_dir = train_models(
        X_train,
        y_train,
        "ridge",
        tmp_path / "models",
        TARGETS,
        PARAMETERS,
        "tfidf",
        Path("run-tfidf"),
    )
    _validate_model_bundle(model_dir, "ridge", "tfidf", Path("run-tfidf"), TARGETS)

    metrics = evaluate_models(
        {"formal_register": np.asarray([2, 3])},
        np.asarray([[1.0], [2.0]]),
        model_dir,
        TARGETS,
    )

    assert set(metrics) == {"formal_register", "overall"}
    assert set(metrics["overall"]) == {"rmse", "mae"}
