# Training and evaluation

This guide explains how to train and evaluate the text-mining regression models in this repository. The scripts use cleaned CSV labels and precomputed TF-IDF matrices from `data/`; they do not create or update those input files. Run the commands from the repository root with the project virtual environment.

## Prerequisites

Create the environment and install dependencies as described in [README.md](../README.md). The data directory must contain these files:

- `train_limpo.csv` and `X_train_tfidf.npz` for training.
- `valid_limpo.csv` and `X_valid_tfidf.npz` for evaluation.

The CSV files must contain the four target columns used by the scripts: `formal_register`, `thematic_coherence`, `narrative_rhetorical_structure`, and `cohesion`. The corresponding CSV row count must match the TF-IDF matrix row count.

## Train a model

Use the default Ridge model with:

```bash
.venv/bin/python src/train.py --data-dir data --model ridge
```

The available models are `ridge` and `linear_svr`. If `--model` is omitted, training uses `ridge`. By default, each model is saved under `artifacts/models/<model>/` as one `.joblib` file per target. To choose another output location, pass `--output-dir`:

```bash
.venv/bin/python src/train.py --data-dir data --model linear_svr --output-dir artifacts/models
```

Training stops with an error when an input file is missing, a CSV and matrix have different row counts, a target column is missing, or the model name is unsupported.

## Follow up with evaluation

Continue with evaluation using the same model and data by following the [evaluation guide](evaluate.md).

## Source of truth

- [src/train.py](../src/train.py) defines input validation, model options, target names, and model output paths.
- [requirements.txt](../requirements.txt) defines the Python dependencies.
