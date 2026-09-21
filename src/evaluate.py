from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __package__:
    from .model_config import DEFAULT_CONFIG_PATH, get_model_config, load_config
    from .settings import SETTINGS
else:
    from model_config import DEFAULT_CONFIG_PATH, get_model_config, load_config
    from settings import SETTINGS


def load_validation_data(data_dir: Path, targets: list[str]):
    """Load validation labels and the precomputed TF-IDF matrix."""
    valid_path = data_dir / SETTINGS.valid_labels_filename
    tfidf_path = data_dir / SETTINGS.valid_tfidf_filename

    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found: {valid_path}")

    if not tfidf_path.exists():
        raise FileNotFoundError(f"Validation TF-IDF matrix not found: {tfidf_path}")

    valid_df = pd.read_csv(valid_path)
    X_valid = sparse.load_npz(tfidf_path)

    if len(valid_df) != X_valid.shape[0]:
        raise ValueError(
            "Row mismatch between "
            f"{SETTINGS.valid_labels_filename} and "
            f"{SETTINGS.valid_tfidf_filename}: "
            f"{len(valid_df)} != {X_valid.shape[0]}"
        )

    missing_targets = [target for target in targets if target not in valid_df.columns]

    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    return valid_df, X_valid


def load_models(model_dir: Path, targets: list[str]):
    """Load one trained model for each target."""
    models = {}

    for target in targets:
        model_filename = SETTINGS.model_filename_template.format(
            model=model_dir.name,
            target=target,
        )
        model_path = model_dir / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        models[target] = joblib.load(model_path)

    return models


def evaluate_models(
    valid_df: pd.DataFrame,
    X_valid,
    models: dict,
    targets: list[str],
):
    """Evaluate all target models on the validation set."""
    metrics = {}
    predictions = {}

    for target in targets:
        y_true = valid_df[target].to_numpy()

        y_pred_raw = models[target].predict(X_valid)

        # the labels are restricted to the 1-5 range.
        range = [1.0, 5.0]

        y_pred = np.clip(y_pred_raw, range[0], range[1])

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        metrics[target] = {
            "rmse": float(rmse),
            "mae": float(mae),
        }

        predictions[target] = y_pred

    metrics["overall"] = {
        "rmse": float(np.mean([metrics[target]["rmse"] for target in targets])),
        "mae": float(np.mean([metrics[target]["mae"] for target in targets])),
    }

    return metrics, predictions


def save_metrics(
    metrics: dict,
    output_dir: Path,
    model_name: str,
):
    """Persist evaluation metrics as JSON."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = SETTINGS.metrics_filename_template.format(
        model=model_name,
    )
    output_path = output_dir / output_filename

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def print_metrics(
    model_name: str,
    metrics: dict,
    targets: list[str],
):
    print(f"\nModel: {model_name}")
    print("-" * 60)

    for target in targets:
        result = metrics[target]

        print(f"{target:<35} RMSE={result['rmse']:.4f} MAE={result['mae']:.4f}")

    print("-" * 60)

    overall = metrics["overall"]

    print(f"{'Overall':<35} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained text-mining models.")

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
        help="Directory containing validation data (default: DATA_DIR from .env).",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model family to evaluate (must be defined in the YAML configuration).",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=SETTINGS.models_dir,
        help="Directory containing trained models (default: MODELS_DIR from .env).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SETTINGS.evaluation_dir,
        help="Directory for evaluation results (default: EVALUATION_DIR from .env).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    get_model_config(config, args.model)
    targets = config["targets"]

    valid_df, X_valid = load_validation_data(
        args.data_dir,
        targets,
    )

    model_dir = args.models_dir / args.model

    models = load_models(model_dir, targets)

    print(
        f"Loaded {X_valid.shape[0]} validation samples "
        f"with {X_valid.shape[1]} TF-IDF features."
    )

    metrics, _ = evaluate_models(
        valid_df=valid_df,
        X_valid=X_valid,
        models=models,
        targets=targets,
    )

    print_metrics(
        model_name=args.model,
        metrics=metrics,
        targets=targets,
    )

    output_path = save_metrics(
        metrics=metrics,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
