from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np

from .config import DEFAULT_CONFIG_PATH, create_model, get_model_config, load_config
from .data import load_experiment_data, resolve_run_dir


def train_models(
    X_train,
    y_train: Mapping[str, np.ndarray],
    model_name: str,
    output_dir: Path,
    targets: list[str],
    model_params: dict,
) -> None:
    """Train and persist one model for each competition target."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        model = create_model(model_name, model_params)
        model.fit(X_train, y_train[target])
        joblib.dump(model, model_dir / f"{target}.joblib")


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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = get_model_config(config, args.model)
    targets = config["targets"]
    run_dir = resolve_run_dir(args.run_dir)
    data = load_experiment_data(run_dir, args.representation, targets)
    train_models(
        data.X_train,
        data.y_train,
        args.model,
        args.output_dir,
        targets,
        model_config["params"],
    )


if __name__ == "__main__":
    main()
