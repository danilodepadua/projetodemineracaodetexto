# Current pipeline run

This guide runs every available stage in order. PR0 preprocessing and the existing model scripts are both runnable, but they use different artifact layouts. The final integration from timestamped preprocessing runs into model commands is intentionally deferred to a later representation/model PR.

## 1. Enter the project environment

Run commands from the repository root inside the devcontainer:

```bash
python -m pip install -r requirements.txt
ruff format --check src tests
ruff check src tests
pyright
python -m pytest
```

## 2. Validate, clean, and represent raw competition data

Ensure `data/raw/` contains `train.csv`, `valid.csv`, `test.csv`, and `sample_submission.csv`. Then run:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

The command performs this sequence:

```text
raw CSVs
→ input/schema/split validation
→ conservative text cleaning
→ clean CSVs and reports
→ fit TF and TF-IDF on train only
→ transform valid and test
→ sparse matrices, vectorizers, manifest
```

Record the printed `artifacts/preprocessing/run-...` path. Inspect `manifest.json` and the `reports/` directory before using its artifacts. See [preprocessing.md](preprocessing.md) for formats and cleaning rules.

## 3. Run the current legacy model baseline

The current model commands consume the pre-existing flat legacy files in `data/` (`train_limpo.csv`, `valid_limpo.csv`, `X_train_tfidf.npz`, and `X_valid_tfidf.npz`). They do **not** yet accept the timestamped PR0 preprocessing output.

Run a Ridge baseline into an isolated output directory:

```bash
python -m src.models.train \
  --data-dir data \
  --model ridge \
  --output-dir artifacts/pipeline-check/models
```

Evaluate the same model against the legacy validation split:

```bash
python -m src.models.evaluate \
  --data-dir data \
  --model ridge \
  --models-dir artifacts/pipeline-check/models \
  --output-dir artifacts/pipeline-check/evaluation
```

The evaluator prints RMSE and MAE for each target and writes `artifacts/pipeline-check/evaluation/ridge.json`.

## 4. Run legacy tuning

Tuning evaluates the configured Ridge and LinearSVR grids using the same legacy flat TF-IDF files. It does not overwrite trained models:

```bash
python -m src.models.tune \
  --data-dir data \
  --output artifacts/pipeline-check/tuning.json
```

Review the output JSON before changing `config/models.yaml`. Tuning, model selection, and submission remain outside PR0.

## 5. Verify the run

```bash
ruff format --check src tests
ruff check src tests
pyright
python -m pytest
```

Generated files under `artifacts/` and source files under `data/` are local ignored artifacts. Do not commit them.

## Current boundary

```text
PR0 raw data → cleaning → TF/TF-IDF artifacts
legacy data/ flat TF-IDF → train → evaluate → tune
```

A later PR will make model commands consume a selected timestamped representation run directly. Until then, keep the two paths separate to avoid silently mixing incompatible vectorizers or split artifacts.
