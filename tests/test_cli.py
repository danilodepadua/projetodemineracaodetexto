import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.preprocessing",
        "src.models.train",
        "src.models.evaluate",
        "src.models.tune",
        "src.models.submit",
    ],
)
def test_module_entrypoint_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_train_cli_defaults_are_repository_relative(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--run-dir", "latest"])
    from src.models.train import parse_args

    args = parse_args()

    assert str(args.runs_dir) == "artifacts/preprocessing"
    assert str(args.output_dir) == "artifacts/models"
    assert str(args.config) == "config/models.yaml"


def _write_cli_inputs(data_dir):
    from src.data import validation

    targets = {target: ["1", "2"] for target in validation.TARGET_COLUMNS}
    import pandas as pd

    pd.DataFrame(
        {
            "id": ["train-1", "train-2"],
            "essay": ["Alpha beta", "Alpha beta"],
            "prompt": ["P", "P"],
            **targets,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": ["valid-1"],
            "essay": ["Alpha gamma"],
            "prompt": ["P"],
            **{target: ["3"] for target in validation.TARGET_COLUMNS},
        }
    ).to_csv(data_dir / "valid.csv", index=False)
    pd.DataFrame({"id": ["test-1"], "essay": ["Alpha delta"], "prompt": ["P"]}).to_csv(
        data_dir / "test.csv", index=False
    )
    pd.DataFrame(
        {"id": ["test-1"], **{target: ["1"] for target in validation.TARGET_COLUMNS}}
    ).to_csv(data_dir / "sample_submission.csv", index=False)


def test_public_cli_pipeline_connects_with_fixture(tmp_path, monkeypatch):
    from pathlib import Path

    from src.data import validation
    from src.models.evaluate import main as evaluate_main
    from src.models.train import main as train_main
    from src.models.tune import main as tune_main
    from src.preprocessing.pipeline import main as preprocessing_main
    from src.models.submit import main as submit_main

    monkeypatch.setattr(
        validation, "EXPECTED_ROWS", {"train": 2, "valid": 1, "test": 1}
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_cli_inputs(data_dir)
    preprocessing_dir = tmp_path / "preprocessing"
    config = str(Path("config/models.yaml").resolve())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(preprocessing_dir),
            "--representation",
            "tfidf",
        ],
    )
    preprocessing_main()
    run_dir = next(preprocessing_dir.glob("run-*"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune",
            "--run-dir",
            str(run_dir),
            "--runs-dir",
            str(preprocessing_dir),
            "--representation",
            "tfidf",
            "--config",
            config,
            "--output",
            str(tmp_path / "tuning.json"),
        ],
    )
    tune_main()

    models_dir = tmp_path / "models"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--run-dir",
            str(run_dir),
            "--runs-dir",
            str(preprocessing_dir),
            "--representation",
            "tfidf",
            "--model",
            "ridge",
            "--config",
            config,
            "--output-dir",
            str(models_dir),
        ],
    )
    train_main()

    evaluation_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--run-dir",
            str(run_dir),
            "--runs-dir",
            str(preprocessing_dir),
            "--representation",
            "tfidf",
            "--model",
            "ridge",
            "--config",
            config,
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(evaluation_dir),
        ],
    )
    evaluate_main()

    submission_file = tmp_path / "submission.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit",
            "--tuning-file",
            str(tmp_path / "tuning.json"),
            "--runs-dir",
            str(preprocessing_dir),
            "--config",
            config,
            "--sample-submission",
            str(data_dir / "sample_submission.csv"),
            "--output",
            str(submission_file),
        ],
    )
    submit_main()

    assert (tmp_path / "tuning.json").is_file()
    assert (models_dir / "tfidf" / "ridge" / "metadata.json").is_file()
    assert (evaluation_dir / "tfidf" / "ridge.json").is_file()
    assert submission_file.is_file()
