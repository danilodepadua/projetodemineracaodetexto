# Tune model hyperparameters

This guide explains how to compare the fixed Ridge and LinearSVR hyperparameter grids used by `src/tune.py`. The script trains temporary models on the training data, evaluates them on the validation data, and writes the full comparison without replacing the saved production model artifacts.

## Prerequisites

Install the dependencies and prepare the environment as described in [README.md](../README.md). From the repository root, the `data/` directory must contain:

- `train_limpo.csv` and `X_train_tfidf.npz`.
- `valid_limpo.csv` and `X_valid_tfidf.npz`.

The training and validation CSV files must contain the target columns `formal_register`, `thematic_coherence`, `narrative_rhetorical_structure`, and `cohesion`. Their row counts must match their corresponding TF-IDF matrices.

## Run tuning

Run the default search from the repository root:

```bash
.venv/bin/python src/tune.py
```

The script reads from `data` when `--data-dir` is omitted. To use another data directory, pass it explicitly:

```bash
.venv/bin/python src/tune.py --data-dir /path/to/data
```

The tested values are `alpha` = `0.01`, `0.1`, `1.0`, `10.0`, and `100.0` for Ridge, and `C` = `0.01`, `0.1`, `1.0`, `10.0`, and `100.0` for LinearSVR. The remaining LinearSVR settings use the fixed values in [tune.py](../src/tune.py).

## Results

Predictions are clipped to the 1–5 label range before calculating RMSE and MAE. The script prints the best result for each target, choosing the experiment with the lowest RMSE, and writes the complete results to `artifacts/evaluation/tuning.json` by default. Use `--output` to choose another JSON path:

```bash
.venv/bin/python src/tune.py --output /tmp/tuning.json
```

The JSON contains every experiment under `ridge` and `linear_svr`, plus the selected result for each target under `best`. Tuning does not save trained models; pass selected values to [train.py](../src/train.py) to create model artifacts.

## Validation and source of truth

Missing or incompatible input files fail during data loading or model fitting. The tuning implementation, parameter grids, metrics, and output structure are defined in [src/tune.py](../src/tune.py). The package dependencies are defined in [requirements.txt](../requirements.txt).
