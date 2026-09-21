# Legacy training

This guide covers the pre-existing model scripts, not the PR0 preprocessing pipeline. Generate or inspect preprocessing artifacts through [preprocessing.md](preprocessing.md) first; these scripts still consume the legacy flat cleaned CSV and TF-IDF filenames configured through [.env](../.env.example). Run commands from the repository root with the project virtual environment.

## Prerequisites

Create the environment and install dependencies as described in [README.md](../README.md). The data directory must contain these files:

- `train_limpo.csv` and `X_train_tfidf.npz` for training.
- `valid_limpo.csv` and `X_valid_tfidf.npz` for evaluation.

The CSV files must contain the four target columns used by the scripts: `formal_register`, `thematic_coherence`, `narrative_rhetorical_structure`, and `cohesion`. The corresponding CSV row count must match the TF-IDF matrix row count.

## Train a model

Use the default Ridge model with the repository's `data/` directory:

```bash
python src/train.py --model ridge
```

The available models are `ridge` and `linear_svr`. If `--model` is omitted, training uses `ridge`. If `--data-dir` is omitted, the script reads from `data` relative to the repository root. By default, each model is saved under `artifacts/models/<model>/` as one `.joblib` file per target. To choose another data or output location, pass `--data-dir` or `--output-dir`:

```bash
python src/train.py --data-dir /path/to/data --model linear_svr --output-dir /path/to/models
```

Model parameters are defined in [config/models.yaml](../config/models.yaml). Edit the `default` value in the selected model's `params` mapping, or pass another YAML file with `--config`:

```bash
python src/train.py \
  --config config/models.yaml \
  --model linear_svr
```

Training stops with an error when an input file is missing, a CSV and matrix have different row counts, a target column is missing, or the model name is unsupported.

## Follow up with evaluation

Continue with evaluation using the same model and data by following the [evaluation guide](evaluate.md).

## Tune hyperparameters

Use the dedicated [tuning guide](tune.md) to compare the supported Ridge and LinearSVR parameter grids before selecting values for training.

## Scope note

PR0 does not train, tune, compare, or select models. The commands below remain available for the existing baseline workflow.

## Source of truth

- [src/train.py](../src/train.py) defines input validation, model options, target names, and model output paths.
- [configuration.md](configuration.md) defines the `.env` path and filename settings.
- [config/models.yaml](../config/models.yaml) defines the targets and fitted model parameters.
- [requirements.txt](../requirements.txt) defines the Python dependencies.
