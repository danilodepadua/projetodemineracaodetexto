# Evaluate trained models

This guide explains how to evaluate a trained text-mining regression model against the validation dataset. Evaluation uses the cleaned validation labels, the precomputed validation TF-IDF matrix, and one saved model per target. Input filenames and output directories come from [.env](../.env.example). Run the commands from the repository root with the project virtual environment.

## Prerequisites

Install the dependencies and prepare the environment as described in [README.md](../README.md). The `data/` directory must contain:

- `valid_limpo.csv` with the four target columns.
- `X_valid_tfidf.npz` with the validation TF-IDF features.

The selected model must already be trained with the same [config/models.yaml](../config/models.yaml) target definitions. By default, the evaluator expects these files under `artifacts/models/<model>/`:

- `formal_register.joblib`
- `thematic_coherence.joblib`
- `narrative_rhetorical_structure.joblib`
- `cohesion.joblib`

Train the models with [train.py](../src/models/train.py) or follow [trainning.md](trainning.md) before evaluating them.

## Run evaluation

Evaluate the Ridge models against `data/` with:

```bash
python -m src.models.evaluate --model ridge
```

The model name must be defined in the YAML configuration and must match the name used during training. If `--config` is omitted, the evaluator reads `config/models.yaml`.

For non-default locations, pass the parent directory containing the model-specific folder and the directory for the metrics file:

```bash
python -m src.models.evaluate \
  --config config/models.yaml \
  --data-dir /path/to/data \
  --model ridge \
  --models-dir artifacts/models \
  --output-dir artifacts/evaluation
```

## Results and validation

The script clips predictions to the valid label range of 1–5, prints RMSE and MAE for each target, and prints the average RMSE and MAE across all four targets. It writes the same values as JSON to `artifacts/evaluation/<model>.json`.

Evaluation stops when a validation input, target column, or model file is missing, or when the CSV row count does not match the TF-IDF matrix row count. The trained models and validation matrix must use compatible TF-IDF features; incompatible features cause prediction to fail.

## Source of truth

- [src/models/evaluate.py](../src/models/evaluate.py) defines the inputs, model paths, prediction clipping, metrics, and JSON output.
- [configuration.md](configuration.md) defines the `.env` path and filename settings.
- [config/models.yaml](../config/models.yaml) defines the model names and target list used by evaluation.
- [requirements.txt](../requirements.txt) defines the Python dependencies.
- [trainning.md](trainning.md) documents the broader training workflow.
