from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DEFAULT_CONFIG_PATH, get_model_config, load_config
from .data import load_split, resolve_run_dir, validate_run


def evaluate_models(
    valid_df, X_valid, model_dir: Path, targets: list[str]
) -> dict[str, dict[str, float]]:
    metrics = {}
    for target in targets:
        model = joblib.load(model_dir / f"{target}.joblib")
        predictions = np.clip(model.predict(X_valid), 1.0, 5.0)
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(valid_df[target], predictions))),
            "mae": float(mean_absolute_error(valid_df[target], predictions)),
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
    validate_run(run_dir, args.representation)
    valid_df, X_valid = load_split(run_dir, "valid", args.representation, targets)
    metrics = evaluate_models(valid_df, X_valid, args.models_dir / args.model, targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
