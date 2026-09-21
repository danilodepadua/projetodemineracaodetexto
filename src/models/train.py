from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np

from ..representations.catalog import canonicalize_representation
from .config import DEFAULT_CONFIG_PATH, create_model, get_model_config, load_config
from .data import load_experiment_data, resolve_representation_run_or_raise


def train_models(
    X_train,
    y_train: Mapping[str, np.ndarray],
    model_name: str,
    output_dir: Path,
    targets: list[str],
    model_params: dict,
    representation: str,
    source_run: Path,
) -> Path:
    """Train and persist one representation-specific model bundle."""
    model_dir = output_dir / representation / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        model = create_model(model_name, model_params)
        model.fit(X_train, y_train[target])
        joblib.dump(model, model_dir / f"{target}.joblib")

    metadata = {
        "representation": representation,
        "source_run": source_run.name,
        "model": model_name,
        "parameters": model_params,
        "targets": targets,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models from a preprocessing run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Preprocessing run directory or latest.",
    )
    parser.add_argument(
        "--representation",
        default="tfidf",
        help="Representation directory in the preprocessing run.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/preprocessing"),
        help="Directory containing timestamped preprocessing runs.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = get_model_config(config, args.model)
    targets = config["targets"]
    representation = canonicalize_representation(args.representation)
    run_dir = resolve_representation_run_or_raise(
        args.run_dir, representation, args.runs_dir
    )
    data = load_experiment_data(run_dir, representation, targets)
    train_models(
        data.X_train,
        data.y_train,
        args.model,
        args.output_dir,
        targets,
        model_config["params"],
        representation,
        run_dir,
    )


if __name__ == "__main__":
    main()
