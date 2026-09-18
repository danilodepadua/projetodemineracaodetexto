# Environment configuration

This guide defines the path and filename settings used by the training, evaluation, and tuning scripts. The scripts load `.env` from the repository root automatically; `.env.example` contains the safe default values and should be copied or adapted for another checkout.

## Loading and precedence

Run commands from the repository root when using relative CLI paths. Relative values in `.env` resolve from the repository root. The precedence order is the process environment, then `.env`, then the built-in defaults; an explicit CLI option such as `--data-dir` or `--output-dir` takes precedence over both environment sources.

## Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_CONFIG_PATH` | `config/models.yaml` | Model and tuning YAML file. |
| `DATA_DIR` | `data` | Directory containing input files. |
| `TRAIN_LABELS_FILENAME` | `train_limpo.csv` | Training labels filename inside `DATA_DIR`. |
| `VALID_LABELS_FILENAME` | `valid_limpo.csv` | Validation labels filename inside `DATA_DIR`. |
| `TRAIN_TFIDF_FILENAME` | `X_train_tfidf.npz` | Training TF-IDF matrix filename inside `DATA_DIR`. |
| `VALID_TFIDF_FILENAME` | `X_valid_tfidf.npz` | Validation TF-IDF matrix filename inside `DATA_DIR`. |
| `MODELS_DIR` | `artifacts/models` | Directory for trained model files. |
| `EVALUATION_DIR` | `artifacts/evaluation` | Directory for evaluation and tuning results. |
| `MODEL_FILENAME_TEMPLATE` | `{target}.joblib` | Template for saved target models. Supports `{model}` and `{target}`. |
| `METRICS_FILENAME_TEMPLATE` | `{model}.json` | Template for evaluation metrics. Supports `{model}`. |
| `TUNING_FILENAME` | `tuning.json` | Default tuning results filename inside `EVALUATION_DIR`. |

## Validation

The shared settings loader is implemented in [src/settings.py](../src/settings.py). Missing data files, invalid model configuration, and incompatible matrices still fail in the command that uses them. Do not put secrets in `.env.example`; the current settings are paths and filenames only.
