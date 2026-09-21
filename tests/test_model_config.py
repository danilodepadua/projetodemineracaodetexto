import json
from pathlib import Path

import numpy as np
from src.models.config import get_model_config, load_config
from src.models.data import resolve_representation_run
from src.models.tune import tune_model


def test_model_config_defines_tunable_models_and_parameters():
    config = load_config()

    assert set(config["models"]) == {"ridge", "linear_svr"}
    assert config["representations"] == [
        "bow",
        "tf",
        "tfidf",
        "structural",
        "word2vec_cbow",
        "word2vec_skipgram",
        "essay_prompt",
        "bert",
    ]
    ridge = get_model_config(config, "ridge")
    assert ridge["params"]["alpha"] == 1.0
    assert ridge["tuning"]["alpha"] == [0.01, 0.1, 1.0, 10.0, 100.0]


def test_tune_model_uses_configured_parameter_grid():
    config = load_config()
    ridge = get_model_config(config, "ridge")
    targets = ["formal_register"]
    y_train = {"formal_register": np.array([1, 2, 3, 4])}
    y_valid = {"formal_register": np.array([2, 3])}
    X_train = np.arange(8, dtype=float).reshape(4, 2)
    X_valid = np.arange(4, dtype=float).reshape(2, 2)

    results = tune_model(
        "ridge",
        ridge,
        y_train,
        y_valid,
        X_train,
        X_valid,
        targets,
    )

    expected_combinations = 1
    for values in ridge["tuning"].values():
        expected_combinations *= len(values)
    assert len(results["formal_register"]) == expected_combinations
    assert {"alpha", "rmse", "mae"} <= set(results["formal_register"][0])


def test_resolve_representation_run_scans_latest_runs(tmp_path: Path):
    first = tmp_path / "run-20260101T000000Z"
    second = tmp_path / "run-20260102T000000Z"
    for run_dir, representations in (
        (first, {"tfidf": {}}),
        (second, {"bert": {}}),
    ):
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps({"representations": representations})
        )

    assert resolve_representation_run("latest", "tfidf", tmp_path) == first
    assert resolve_representation_run("latest", "bert", tmp_path) == second
    assert resolve_representation_run("latest", "bow", tmp_path) is None
