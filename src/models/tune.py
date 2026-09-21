from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DEFAULT_CONFIG_PATH, create_model, get_model_config, load_config
from .data import load_split, resolve_run_dir, validate_run


def evaluate(y_true, y_pred) -> dict[str, float]:
    predictions = np.clip(y_pred, 1.0, 5.0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


def tune_model(name, config, train_df, valid_df, X_train, X_valid, targets):
    results = {target: [] for target in targets}
    tuning = config["tuning"]
    for target in targets:
        for values in product(*tuning.values()):
            params = dict(config["params"])
            params.update(dict(zip(tuning, values, strict=True)))
            model = create_model(name, params)
            model.fit(X_train, train_df[target].to_numpy())
            results[target].append(
                {
                    **dict(zip(tuning, values, strict=True)),
                    **evaluate(valid_df[target].to_numpy(), model.predict(X_valid)),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune models from a preprocessing run."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Preprocessing run directory or latest."
    )
    parser.add_argument("--representation", default="tfidf")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/tuning.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    targets = config["targets"]
    run_dir = resolve_run_dir(args.run_dir)
    validate_run(run_dir, args.representation)
    train_df, X_train = load_split(run_dir, "train", args.representation, targets)
    valid_df, X_valid = load_split(run_dir, "valid", args.representation, targets)
    results = {
        name: tune_model(
            name,
            get_model_config(config, name),
            train_df,
            valid_df,
            X_train,
            X_valid,
            targets,
        )
        for name in ("ridge", "linear_svr")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
