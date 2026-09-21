from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..settings import SETTINGS
from .config import (
    DEFAULT_CONFIG_PATH,
    create_model,
    get_model_config,
    load_config,
)


def load_data(data_dir: Path, targets: list[str]):
    data_paths = {
        "train labels": data_dir / SETTINGS.train_labels_filename,
        "validation labels": data_dir / SETTINGS.valid_labels_filename,
        "training TF-IDF": data_dir / SETTINGS.train_tfidf_filename,
        "validation TF-IDF": data_dir / SETTINGS.valid_tfidf_filename,
    }

    for file_description, path in data_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required {file_description} file not found: {path}"
            )

    train_df = pd.read_csv(data_paths["train labels"])

    valid_df = pd.read_csv(data_paths["validation labels"])

    X_train = sparse.load_npz(data_paths["training TF-IDF"])

    X_valid = sparse.load_npz(data_paths["validation TF-IDF"])

    if len(train_df) != X_train.shape[0]:
        raise ValueError(
            "Row mismatch between "
            f"{SETTINGS.train_labels_filename} and "
            f"{SETTINGS.train_tfidf_filename}: "
            f"{len(train_df)} != {X_train.shape[0]}"
        )

    if len(valid_df) != X_valid.shape[0]:
        raise ValueError(
            "Row mismatch between "
            f"{SETTINGS.valid_labels_filename} and "
            f"{SETTINGS.valid_tfidf_filename}: "
            f"{len(valid_df)} != {X_valid.shape[0]}"
        )

    if X_train.shape[1] != X_valid.shape[1]:
        raise ValueError(
            "Feature mismatch between training and validation TF-IDF matrices: "
            f"{X_train.shape[1]} != {X_valid.shape[1]}"
        )

    for dataframe_name, dataframe in (
        (SETTINGS.train_labels_filename, train_df),
        (SETTINGS.valid_labels_filename, valid_df),
    ):
        missing_targets = [
            target for target in targets if target not in dataframe.columns
        ]

        if missing_targets:
            raise ValueError(
                f"Missing target columns in {dataframe_name}: {missing_targets}"
            )

    return train_df, valid_df, X_train, X_valid


def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 1.0, 5.0)

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
    }


def parameter_combinations(model_config: dict):
    tuning = model_config["tuning"]
    parameter_names = list(tuning)
    parameter_values = [tuning[name] for name in parameter_names]

    for values in product(*parameter_values):
        yield dict(zip(parameter_names, values, strict=True))


def tune_model(
    model_name: str,
    model_config: dict,
    train_df,
    valid_df,
    X_train,
    X_valid,
    targets: list[str],
):
    results = {}

    for target in targets:
        results[target] = []

        y_train = train_df[target].to_numpy()
        y_valid = valid_df[target].to_numpy()

        for tuned_params in parameter_combinations(model_config):
            params = dict(model_config["params"])
            params.update(tuned_params)
            model = create_model(model_name, params)

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(X_valid)

            metrics = evaluate(
                y_valid,
                predictions,
            )

            results[target].append(
                {
                    **tuned_params,
                    **metrics,
                }
            )

    return results


def find_best(results):
    best = {}

    for target, experiments in results.items():
        best[target] = min(
            experiments,
            key=lambda result: result["rmse"],
        )

    return best


def print_best(model_name: str, results: dict):
    print(f"\nBest {model_name} parameters:")

    for target, result in results.items():
        parameters = {
            key: value for key, value in result.items() if key not in {"rmse", "mae"}
        }
        parameter_text = " ".join(
            f"{name}={value}" for name, value in parameters.items()
        )

        print(
            f"{target:<35} "
            f"{parameter_text:<20} "
            f"RMSE={result['rmse']:.4f} "
            f"MAE={result['mae']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML model configuration (default: MODEL_CONFIG_PATH from .env).",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SETTINGS.data_dir,
        help=(
            "Directory containing training and validation data "
            "(default: DATA_DIR from .env)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=SETTINGS.evaluation_dir / SETTINGS.tuning_filename,
        help="JSON output path (default: EVALUATION_DIR/TUNING_FILENAME from .env).",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    targets = config["targets"]

    (
        train_df,
        valid_df,
        X_train,
        X_valid,
    ) = load_data(args.data_dir, targets)

    print("Tuning Ridge...")
    ridge_results = tune_model(
        "ridge",
        get_model_config(config, "ridge"),
        train_df,
        valid_df,
        X_train,
        X_valid,
        targets,
    )

    print("Tuning LinearSVR...")
    svr_results = tune_model(
        "linear_svr",
        get_model_config(config, "linear_svr"),
        train_df,
        valid_df,
        X_train,
        X_valid,
        targets,
    )

    results = {
        "ridge": ridge_results,
        "linear_svr": svr_results,
        "best": {
            "ridge": find_best(ridge_results),
            "linear_svr": find_best(svr_results),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print_best("Ridge", results["best"]["ridge"])
    print_best("LinearSVR", results["best"]["linear_svr"])


if __name__ == "__main__":
    main()
