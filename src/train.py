from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR


TARGETS = [
    "formal_register",
    "thematic_coherence",
    "narrative_rhetorical_structure",
    "cohesion",
]


def load_training_data(data_dir: Path):
    """Load the cleaned training labels and precomputed TF-IDF matrix."""
    train_path = data_dir / "train_limpo.csv"
    tfidf_path = data_dir / "X_train_tfidf.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if not tfidf_path.exists():
        raise FileNotFoundError(f"TF-IDF matrix not found: {tfidf_path}")

    train_df = pd.read_csv(train_path)
    X_train = sparse.load_npz(tfidf_path)

    if len(train_df) != X_train.shape[0]:
        raise ValueError(
            "Row mismatch between train_limpo.csv and X_train_tfidf.npz: "
            f"{len(train_df)} != {X_train.shape[0]}"
        )

    missing_targets = [
        target for target in TARGETS
        if target not in train_df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    return train_df, X_train


def create_model(model_name: str):
    """Create an unfitted regression model."""
    if model_name == "ridge":
        return Ridge(alpha=1.0)  # TODO: tune hyperparameters later

    if model_name == "linear_svr":
        return LinearSVR(
            C=1.0,
            epsilon=0.0,
            random_state=42,
            max_iter=10_000,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def train_models(
    train_df: pd.DataFrame,
    X_train,
    model_name: str,
    output_dir: Path,
):
    """Train and persist one model for each competition target."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        print(f"Training {model_name} for {target}...")

        y_train = train_df[target].to_numpy()

        model = create_model(model_name)
        model.fit(X_train, y_train)

        output_path = model_dir / f"{target}.joblib"
        joblib.dump(model, output_path)

        print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train text-mining regression models."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the cleaned dataset and TF-IDF matrices.",
    )

    parser.add_argument(
        "--model",
        choices=["ridge", "linear_svr"],
        default="ridge",
        help="Model to train.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory where trained models will be stored.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading training data from: {args.data_dir}")

    train_df, X_train = load_training_data(args.data_dir)

    print(
        f"Loaded {X_train.shape[0]} samples "
        f"with {X_train.shape[1]} TF-IDF features."
    )

    train_models(
        train_df=train_df,
        X_train=X_train,
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
