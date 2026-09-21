from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DEFAULT_CONFIG_PATH, get_model_config, load_config
from .data import load_experiment_data, resolve_run_dir


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
    run_dir = resolve_run_dir(args.run_dir)
    data = load_experiment_data(run_dir, args.representation, targets)
    metrics = evaluate_models(
        data.y_valid,
        data.X_valid,
        args.models_dir / args.model,
        targets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
