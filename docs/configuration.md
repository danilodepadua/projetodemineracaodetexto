# Configuration

Run commands from the repository root. The project has no required environment variables and does not load dotenv settings. A `.env` file is optional local state, not part of the execution contract; do not create one for the default workflow.

## Source of truth

`config/models.yaml` defines:

- the four target columns;
- canonical representation candidates;
- supported model names;
- default model parameters; and
- hyperparameter grids used by tuning.

Do not duplicate these lists in shell configuration or documentation.

## Path defaults

| CLI option | Default | Used by |
| --- | --- | --- |
| `--data-dir` | `data/raw` | preprocessing |
| `--output-dir` | `artifacts/preprocessing` for preprocessing; `artifacts/models` for train; `artifacts/evaluation` for evaluate | preprocessing/train/evaluate |
| `--run-dir` | required for train/evaluate; `latest` for tune | model commands |
| `--runs-dir` | `artifacts/preprocessing` | train/evaluate/tune |
| `--config` | `config/models.yaml` | train/evaluate/tune |
| `--models-dir` | `artifacts/models` | evaluate |
| `--output` | `artifacts/evaluation/tuning.json` | tune |

Paths are repository-relative when commands run from the repository root. Pass an explicit CLI path to use another location. CLI arguments are the only supported override mechanism, and they take precedence over documented defaults.

## Repository-root contract

The supported invocation location is the repository root. This keeps `config/models.yaml`, `data/raw`, and `artifacts/` unambiguous across host, `.venv`, and container environments. Use absolute paths only as explicit CLI arguments; generated manifests and model metadata do not require producer-machine absolute paths.

## Validation

Invalid YAML, unsupported representations/models, missing raw files, missing runs, and incompatible model metadata fail in the command that uses them. Use the error's suggested action, then rerun the command. See [pipeline.md](pipeline.md) and [models.md](models.md) for executable flows.
