from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, create_model, load_config
from .data import load_experiment_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a submission file from the best models in tuning-all.json."
    )
    parser.add_argument(
        "--tuning-file",
        type=Path,
        default=Path("artifacts/evaluation/tuning-all.json"),
        help="Path to the JSON file containing tuning results.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/preprocessing"),
        help="Directory containing timestamped preprocessing runs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to models.yaml configuration.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/raw/sample_submission.csv"),
        help="Path to the sample submission CSV file (used to locate the directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path to save the generated submission CSV.",
    )
    args = parser.parse_args()

    if not args.tuning_file.is_file():
        raise FileNotFoundError(f"Tuning file not found: {args.tuning_file}")

    with args.tuning_file.open("r", encoding="utf-8") as f:
        tuning_results = json.load(f)

    config = load_config(args.config)
    targets = config["targets"]

    best_configs: dict[str, dict[str, Any]] = {target: {} for target in targets}
    best_metrics = {target: float("inf") for target in targets}

    for rep_name, rep_data in tuning_results.items():
        if rep_data.get("status") != "ran":
            continue
            
        run_dir = rep_data["run_dir"]
        for model_name, model_results in rep_data.get("models", {}).items():
            for target in targets:
                target_results = model_results.get(target, [])
                for params_res in target_results:
                    rmse = params_res.get("rmse", float("inf"))
                    
                    if rmse < best_metrics[target]:
                        best_metrics[target] = rmse
                        best_params = {
                            k: v for k, v in params_res.items() 
                            if k not in ("rmse", "mae")
                        }
                        best_configs[target] = {
                            "representation": rep_name,
                            "run_dir": run_dir,
                            "model_name": model_name,
                            "params": best_params,
                        }

    test_file_path = args.sample_submission.with_name("test.csv")
    if not test_file_path.is_file():
         raise FileNotFoundError(f"Could not find test.csv at {test_file_path}")
         
    submission = pd.DataFrame({"id": pd.read_csv(test_file_path)["id"]})

    for target in targets:
        best = best_configs[target]
        if not best:
            raise ValueError(f"No valid tuning results found for target '{target}'.")
            
        print(f"Target '{target}': using {best['model_name']} with {best['representation']} "
              f"from {best['run_dir']} (Best RMSE: {best_metrics[target]:.4f})")
        
        selected_run = args.runs_dir / best["run_dir"]
        data = load_experiment_data(selected_run, best["representation"], targets)
        
        model = create_model(best["model_name"], best["params"])
        model.fit(data.X_train, data.y_train[target])
        
        if not hasattr(data, "X_test"):
            raise AttributeError("The loaded experiment data does not contain 'X_test'.")
            
        preds = model.predict(data.X_test)
        preds = np.clip(preds, 1.0, 5.0)
        
        preds = np.round(preds).astype(int)
        
        submission[target] = preds

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission successfully saved to: {args.output}")


if __name__ == "__main__":
    main()