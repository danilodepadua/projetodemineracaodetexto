from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..representations.catalog import canonicalize_representation
from .config import DEFAULT_CONFIG_PATH, get_model_config, load_config
from .data import load_experiment_data, resolve_representation_run_or_raise


def _validate_model_bundle(
    model_dir: Path,
    model_name: str,
    representation: str,
    source_run: Path,
    targets: list[str],
) -> None:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Retrain the model with "
            "the current representation-aware training command."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid model metadata: {metadata_path}") from error

    expected = {
        "model": model_name,
        "representation": representation,
        "source_run": source_run.name,
        "targets": targets,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Model artifact mismatch for {key}: expected {value!r}, "
                f"found {metadata.get(key)!r} in {metadata_path}"
            )

    missing = [
        target for target in targets if not (model_dir / f"{target}.joblib").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts missing for {representation}/{model_name}: {missing}"
        )


def evaluate_models(
    y_valid: Mapping[str, np.ndarray],
    X_valid,
    model_dir: Path,
    targets: list[str],
) -> dict[str, dict[str, float]]:
    metrics = {}
    for target in targets:
        model = joblib.load(model_dir / f"{target}.joblib")
        predictions = np.clip(model.predict(X_valid), 1.0, 5.0)
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(y_valid[target], predictions))),
            "mae": float(mean_absolute_error(y_valid[target], predictions)),
        }
    metrics["overall"] = {
        key: float(np.mean([metrics[target][key] for target in targets]))
        for key in ("rmse", "mae")
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models from a preprocessing run."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Preprocessing run directory or latest."
    )
    parser.add_argument("--representation", default="tfidf")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/preprocessing"),
        help="Directory containing timestamped preprocessing runs.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    get_model_config(config, args.model)
    targets = config["targets"]
    representation = canonicalize_representation(args.representation)
    run_dir = resolve_representation_run_or_raise(
        args.run_dir, representation, args.runs_dir
    )
    data = load_experiment_data(run_dir, representation, targets)
    model_dir = args.models_dir / representation / args.model
    _validate_model_bundle(model_dir, args.model, representation, run_dir, targets)
    metrics = evaluate_models(data.y_valid, data.X_valid, model_dir, targets)
    output_dir = args.output_dir / representation
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.model}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
