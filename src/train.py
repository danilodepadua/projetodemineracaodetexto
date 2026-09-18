from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, cast

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


def create_model(
        model_name: str,
        *,
        alpha: float = 1.0,
        epsilon: float = 0.0,
        tol: float = 1e-4,
        c: float = 1.0,
        loss: Literal[
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ] = "epsilon_insensitive",
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        dual: bool | Literal["auto"] = "auto",
        verbose: int = 0,
        random_state: int | None = 42,
        max_iter: int = 10_000,
):
    """Create an unfitted regression model."""
    if model_name == "ridge":
        return Ridge(alpha=alpha)

    if model_name == "linear_svr":
        return LinearSVR(
            C=c,
            epsilon=epsilon,
            tol=tol,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=cast(Any, dual),
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def train_models(
    train_df: pd.DataFrame,
    X_train,
    model_name: str,
    output_dir: Path,
    model_params: dict | None = None,
):
    """Train and persist one model for each competition target."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        print(f"Training {model_name} for {target}...")

        y_train = train_df[target].to_numpy()

        model = create_model(model_name, **(model_params or {}))
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
        default=Path("data"),
        help="Directory containing the cleaned dataset and TF-IDF matrices (default: data).",
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

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="LinearSVR epsilon in the epsilon-insensitive loss function.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="LinearSVR stopping tolerance.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="LinearSVR regularization parameter.",
    )
    parser.add_argument(
        "--loss",
        choices=["epsilon_insensitive", "squared_epsilon_insensitive"],
        default="epsilon_insensitive",
        help="LinearSVR loss function.",
    )
    parser.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LinearSVR should fit an intercept.",
    )
    parser.add_argument(
        "--intercept-scaling",
        type=float,
        default=1.0,
        help="LinearSVR intercept scaling factor.",
    )
    parser.add_argument(
        "--dual",
        choices=["auto", "true", "false"],
        default="auto",
        help="LinearSVR dual optimization mode.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="LinearSVR verbosity level.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="LinearSVR random seed.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10_000,
        help="LinearSVR maximum number of iterations.",
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
        model_params={
            "alpha": args.alpha,
            "epsilon": args.epsilon,
            "tol": args.tol,
            "c": args.c,
            "loss": args.loss,
            "fit_intercept": args.fit_intercept,
            "intercept_scaling": args.intercept_scaling,
            "dual": {
                "true": True,
                "false": False,
            }.get(args.dual, args.dual),
            "verbose": args.verbose,
            "random_state": args.random_state,
            "max_iter": args.max_iter,
        },
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
