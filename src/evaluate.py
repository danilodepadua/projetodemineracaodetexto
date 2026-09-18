from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 4 specific target columns in the dataset that we want to predict
TARGETS = [
    "formal_register",
    "thematic_coherence",
    "narrative_rhetorical_structure",
    "cohesion",
]


def load_validation_data(data_dir: Path):
    """Load validation labels and the precomputed TF-IDF matrix."""
    valid_path = data_dir / "valid_limpo.csv"
    tfidf_path = data_dir / "X_valid_tfidf.npz"

    if not valid_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {valid_path}"
        )

    if not tfidf_path.exists():
        raise FileNotFoundError(
            f"Validation TF-IDF matrix not found: {tfidf_path}"
        )

    valid_df = pd.read_csv(valid_path)
    X_valid = sparse.load_npz(tfidf_path)

    if len(valid_df) != X_valid.shape[0]:
        raise ValueError(
            "Row mismatch between valid_limpo.csv and "
            f"X_valid_tfidf.npz: {len(valid_df)} != {X_valid.shape[0]}"
        )

    missing_targets = [
        target
        for target in TARGETS
        if target not in valid_df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    return valid_df, X_valid


def load_models(model_dir: Path):
    """Load one trained model for each target."""
    models = {}

    for target in TARGETS:
        model_path = model_dir / f"{target}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[target] = joblib.load(model_path)

    return models


def evaluate_models(
    valid_df: pd.DataFrame,
    X_valid,
    models: dict,
):
    """Evaluate all target models on the validation set."""
    metrics = {}
    predictions = {}

    for target in TARGETS:
        y_true = valid_df[target].to_numpy()

        y_pred_raw = models[target].predict(X_valid)

        # the labels are restricted to the 1-5 range.
        range = [1.0, 5.0]

        y_pred = np.clip(y_pred_raw, range[0], range[1])

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

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
        "rmse": float(
            np.mean([
                metrics[target]["rmse"]
                for target in TARGETS
            ])
        ),
        "mae": float(
            np.mean([
                metrics[target]["mae"]
                for target in TARGETS
            ])
        ),
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

    output_path = output_dir / f"{model_name}.json"

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
):
    print(f"\nModel: {model_name}")
    print("-" * 60)

    for target in TARGETS:
        result = metrics[target]

        print(
            f"{target:<35} "
            f"RMSE={result['rmse']:.4f} "
            f"MAE={result['mae']:.4f}"
        )

    print("-" * 60)

    overall = metrics["overall"]

    print(
        f"{'Overall':<35} "
        f"RMSE={overall['rmse']:.4f} "
        f"MAE={overall['mae']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained text-mining models."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the cleaned validation data (default: data).",
    )

    parser.add_argument(
        "--model",
        choices=["ridge", "linear_svr"],
        required=True,
        help="Model family to evaluate.",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory containing trained models.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Directory where evaluation results will be stored.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    valid_df, X_valid = load_validation_data(
        args.data_dir
    )

    model_dir = args.models_dir / args.model

    models = load_models(model_dir)

    print(
        f"Loaded {X_valid.shape[0]} validation samples "
        f"with {X_valid.shape[1]} TF-IDF features."
    )

    metrics, _ = evaluate_models(
        valid_df=valid_df,
        X_valid=X_valid,
        models=models,
    )

    print_metrics(
        model_name=args.model,
        metrics=metrics,
    )

    output_path = save_metrics(
        metrics=metrics,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
