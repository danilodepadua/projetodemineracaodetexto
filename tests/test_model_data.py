import json

import pytest
from src.models.data import (
    resolve_representation_run_or_raise,
    resolve_run_dir,
)


def test_resolve_run_dir_returns_latest_timestamped_run(tmp_path):
    older = tmp_path / "run-20260101T000000Z"
    latest = tmp_path / "run-20260102T000000Z"
    for run_dir in (older, latest):
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"representations": {}}))

    assert resolve_run_dir("latest", tmp_path) == latest


def test_resolve_run_dir_rejects_missing_latest_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="No preprocessing runs"):
        resolve_run_dir("latest", tmp_path)


def test_resolve_run_dir_preserves_explicit_path(tmp_path):
    explicit = tmp_path / "custom-run"
    assert resolve_run_dir(explicit) == explicit


def test_resolve_representation_run_or_raise_reports_missing_latest(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run preprocessing"):
        resolve_representation_run_or_raise("latest", "bert", tmp_path)


def test_resolve_representation_run_or_raise_rejects_explicit_mismatch(tmp_path):
    run_dir = tmp_path / "run-20260101T000000Z"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"representations": {"tfidf": {}}})
    )

    with pytest.raises(ValueError, match="not available"):
        resolve_representation_run_or_raise(run_dir, "bert", tmp_path)
