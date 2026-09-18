# Evaluate trained models

This guide explains how to evaluate a trained text-mining regression model against the validation dataset. Evaluation uses the cleaned validation labels, the precomputed validation TF-IDF matrix, and one saved model per target. Run the commands from the repository root with the project virtual environment.

## Prerequisites

Install the dependencies and prepare the environment as described in [README.md](../README.md). The `data/` directory must contain:

- `valid_limpo.csv` with the four target columns.
- `X_valid_tfidf.npz` with the validation TF-IDF features.

The selected model must already be trained. By default, the evaluator expects these files under `artifacts/models/<model>/`:

- `formal_register.joblib`
- `thematic_coherence.joblib`
- `narrative_rhetorical_structure.joblib`
- `cohesion.joblib`

Train the models with [train.py](../src/train.py) or follow [trainning.md](trainning.md) before evaluating them.

## Run evaluation

Evaluate the Ridge models against `data/` with:

```bash
.venv/bin/python src/evaluate.py --model ridge
```

The supported model names are `ridge` and `linear_svr`. Use the same name used during training. 

For non-default locations, pass the parent directory containing the model-specific folder and the directory for the metrics file:

```bash
.venv/bin/python src/evaluate.py \
  --data-dir /path/to/data \
  --model ridge \
  --models-dir artifacts/models \
  --output-dir artifacts/evaluation
```

## Results and validation

The script clips predictions to the valid label range of 1–5, prints RMSE and MAE for each target, and prints the average RMSE and MAE across all four targets. It writes the same values as JSON to `artifacts/evaluation/<model>.json`.

Evaluation stops when a validation input, target column, or model file is missing, or when the CSV row count does not match the TF-IDF matrix row count. The trained models and validation matrix must use compatible TF-IDF features; incompatible features cause prediction to fail.

## Source of truth

- [src/evaluate.py](../src/evaluate.py) defines the inputs, model paths, prediction clipping, metrics, and JSON output.
- [requirements.txt](../requirements.txt) defines the Python dependencies.
- [trainning.md](trainning.md) documents the broader training workflow.
