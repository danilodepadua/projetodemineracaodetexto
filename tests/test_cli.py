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
