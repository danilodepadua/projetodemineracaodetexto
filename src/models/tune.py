from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..representations.catalog import canonicalize_representation
from .config import DEFAULT_CONFIG_PATH, create_model, get_model_config, load_config
from .data import load_experiment_data, resolve_representation_run


def evaluate(y_true, y_pred) -> dict[str, float]:
    predictions = np.clip(y_pred, 1.0, 5.0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


def tune_model(name, config, y_train, y_valid, X_train, X_valid, targets):
    results = {target: [] for target in targets}
    tuning = config["tuning"]
    for target in targets:
        for values in product(*tuning.values()):
            params = dict(config["params"])
            params.update(dict(zip(tuning, values, strict=True)))
            model = create_model(name, params)
            model.fit(X_train, y_train[target])
            results[target].append(
                {
                    **dict(zip(tuning, values, strict=True)),
                    **evaluate(y_valid[target], model.predict(X_valid)),
                }
            )
    return results


def tune_representations(
    config: dict[str, Any],
    run_dir: str | Path,
    runs_dir: Path = Path("artifacts/preprocessing"),
    representations: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Tune configured models over every available configured representation."""
    targets = config["targets"]
    names = representations or config["representations"]
    results: dict[str, dict[str, Any]] = {}

    for requested_name in names:
        name = canonicalize_representation(requested_name)
        selected_run = resolve_representation_run(run_dir, name, runs_dir)
        if selected_run is None:
            results[name] = {
                "status": "skipped",
                "reason": (
                    "No preprocessing artifact run contains this representation"
                ),
                "models": {},
            }
            continue

        try:
            data = load_experiment_data(selected_run, name, targets)
        except (FileNotFoundError, ValueError, OSError) as error:
            results[name] = {
                "status": "skipped",
                "reason": str(error),
                "run_dir": str(selected_run),
                "models": {},
            }
            continue

        results[name] = {
            "status": "ran",
            "run_dir": str(selected_run),
            "models": {
                model_name: tune_model(
                    model_name,
                    get_model_config(config, model_name),
                    data.y_train,
                    data.y_valid,
                    data.X_train,
                    data.X_valid,
                    targets,
                )
                for model_name in config["models"]
            },
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune configured models over available preprocessing artifacts."
    )
    parser.add_argument(
        "--run-dir",
        default="latest",
        help="Preprocessing run directory or latest available run per representation.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/preprocessing"),
        help="Directory containing timestamped preprocessing runs.",
    )
    parser.add_argument(
        "--representation",
        help="Tune one canonical representation instead of all configured candidates.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/tuning.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    results = tune_representations(
        config,
        args.run_dir,
        args.runs_dir,
        [args.representation] if args.representation else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for representation, result in results.items():
        run_dir = result.get("run_dir", "-")
        print(f"{representation}: {result['status']} ({run_dir})")
        if result["status"] == "skipped":
            print(f"  reason: {result['reason']}")


if __name__ == "__main__":
    main()
