from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import LinearSVR


TARGETS = [
    "formal_register",
    "thematic_coherence",
    "narrative_rhetorical_structure",
    "cohesion",
]


RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

SVR_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def load_data(data_dir: Path):
    train_df = pd.read_csv(
        data_dir / "train_limpo.csv"
    )

    valid_df = pd.read_csv(
        data_dir / "valid_limpo.csv"
    )

    X_train = sparse.load_npz(
        data_dir / "X_train_tfidf.npz"
    )

    X_valid = sparse.load_npz(
        data_dir / "X_valid_tfidf.npz"
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


def tune_ridge(
    train_df,
    valid_df,
    X_train,
    X_valid,
):
    results = {}

    for target in TARGETS:
        results[target] = []

        y_train = train_df[target].to_numpy()
        y_valid = valid_df[target].to_numpy()

        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha)

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(X_valid)

            metrics = evaluate(
                y_valid,
                predictions,
            )

            results[target].append({
                "alpha": alpha,
                **metrics,
            })

    return results


def tune_linear_svr(
    train_df,
    valid_df,
    X_train,
    X_valid,
):
    results = {}

    for target in TARGETS:
        results[target] = []

        y_train = train_df[target].to_numpy()
        y_valid = valid_df[target].to_numpy()

        for c in SVR_C_VALUES:
            model = LinearSVR(
                C=c,
                epsilon=0.0,
                random_state=42,
                max_iter=10_000,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(X_valid)

            metrics = evaluate(
                y_valid,
                predictions,
            )

            results[target].append({
                "C": c,
                **metrics,
            })

    return results


def find_best(results):
    best = {}

    for target, experiments in results.items():
        best[target] = min(
            experiments,
            key=lambda result: result["rmse"],
        )

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/tuning.json"
        ),
    )

    args = parser.parse_args()

    (
        train_df,
        valid_df,
        X_train,
        X_valid,
    ) = load_data(args.data_dir)

    print("Tuning Ridge...")
    ridge_results = tune_ridge(
        train_df,
        valid_df,
        X_train,
        X_valid,
    )

    print("Tuning LinearSVR...")
    svr_results = tune_linear_svr(
        train_df,
        valid_df,
        X_train,
        X_valid,
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

    print("\nBest Ridge parameters:")
    for target, result in results["best"]["ridge"].items():
        print(
            f"{target:<35} "
            f"alpha={result['alpha']:<8} "
            f"RMSE={result['rmse']:.4f} "
            f"MAE={result['mae']:.4f}"
        )

    print("\nBest LinearSVR parameters:")
    for target, result in results["best"]["linear_svr"].items():
        print(
            f"{target:<35} "
            f"C={result['C']:<8} "
            f"RMSE={result['rmse']:.4f} "
            f"MAE={result['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
