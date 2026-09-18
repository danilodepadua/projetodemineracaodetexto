from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

from model_config import DEFAULT_CONFIG_PATH, create_model, get_model_config, load_config
from settings import SETTINGS


def load_training_data(data_dir: Path, targets: list[str]):
    """Load the cleaned training labels and precomputed TF-IDF matrix."""
    train_path = data_dir / SETTINGS.train_labels_filename
    tfidf_path = data_dir / SETTINGS.train_tfidf_filename

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if not tfidf_path.exists():
        raise FileNotFoundError(f"TF-IDF matrix not found: {tfidf_path}")

    train_df = pd.read_csv(train_path)
    X_train = sparse.load_npz(tfidf_path)

    if len(train_df) != X_train.shape[0]:
        raise ValueError(
            "Row mismatch between "
            f"{SETTINGS.train_labels_filename} and "
            f"{SETTINGS.train_tfidf_filename}: "
            f"{len(train_df)} != {X_train.shape[0]}"
        )

    missing_targets = [
        target for target in targets
        if target not in train_df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    return train_df, X_train


def train_models(
    train_df: pd.DataFrame,
    X_train,
    model_name: str,
    output_dir: Path,
    targets: list[str],
    model_params: dict,
):
    """Train and persist one model for each competition target."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"Training {model_name} for {target}...")

        y_train = train_df[target].to_numpy()

        model = create_model(model_name, model_params)
        model.fit(X_train, y_train)

        output_filename = SETTINGS.model_filename_template.format(
            model=model_name,
            target=target,
        )
        output_path = model_dir / output_filename
        joblib.dump(model, output_path)

        print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train text-mining regression models."
    )

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
        help="Directory containing data (default: DATA_DIR from .env).",
    )

    parser.add_argument(
        "--model",
        default="ridge",
        help="Model to train (must be defined in the YAML configuration).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SETTINGS.models_dir,
        help="Directory where trained models are stored (default: MODELS_DIR from .env).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = get_model_config(config, args.model)
    targets = config["targets"]

    print(f"Loading training data from: {args.data_dir}")

    train_df, X_train = load_training_data(args.data_dir, targets)

    print(
        f"Loaded {X_train.shape[0]} samples "
        f"with {X_train.shape[1]} TF-IDF features."
    )

    train_models(
        train_df=train_df,
        X_train=X_train,
        model_name=args.model,
        output_dir=args.output_dir,
        targets=targets,
        model_params=model_config["params"],
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
