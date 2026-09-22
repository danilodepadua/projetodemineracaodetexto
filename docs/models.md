# Models

## Purpose

Model commands consume an existing preprocessing run through `load_experiment_data`. They never open matrix files directly and never trigger preprocessing. `config/models.yaml` is the source of truth for targets, representation candidates, models, defaults, and tuning grids.

## Tune configured representations

Prerequisite: generate at least one preprocessing run. Tune every available configured representation:

```bash
python -m src.models.tune \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --config config/models.yaml \
  --output artifacts/evaluation/tuning-all.json
```

Tune one representation:

```bash
python -m src.models.tune \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --representation tfidf \
  --config config/models.yaml \
  --output artifacts/evaluation/tuning-tfidf.json
```

Tuning discovers the newest run containing each requested representation. It skips unavailable or invalid representations with a reason and does not regenerate them. Results are grouped as `representation -> model -> target -> hyperparameters and metrics`.

Expected result: a JSON file under `artifacts/evaluation/`. The command does not write trained production models.

## Train a model

```bash
python -m src.models.train \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --representation tfidf \
  --model ridge \
  --config config/models.yaml \
  --output-dir artifacts/models
```

`--run-dir latest` resolves to the newest run containing the requested representation. Use an explicit `run-<timestamp>` path when a run must be pinned. `--runs-dir` is the directory searched for `latest`.

Expected output:

```text
artifacts/models/
└── tfidf/
    └── ridge/
        ├── formal_register.joblib
        ├── thematic_coherence.joblib
        ├── narrative_rhetorical_structure.joblib
        ├── cohesion.joblib
        └── metadata.json
```

`metadata.json` records the representation, source run name, model name, parameters, and target list. The representation directory prevents `ridge + tfidf` and `ridge + bert` from overwriting one another.

## Evaluate a model

```bash
python -m src.models.evaluate \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --representation tfidf \
  --model ridge \
  --models-dir artifacts/models \
  --output-dir artifacts/evaluation
```

Evaluation resolves the requested representation, loads it through the handoff loader, and validates model metadata before prediction. It writes `artifacts/evaluation/tfidf/ridge.json` with per-target and overall RMSE/MAE.

Evaluation fails clearly when the model directory, metadata, target artifact, representation, model name, or source preprocessing run does not match. Retrain against the requested run/representation instead of renaming files or bypassing metadata checks.

## Creating submission file
```bash
python -m src.models.submit --output submission.csv
```

## `latest` behavior

- Training and evaluation: newest preprocessing run containing the requested representation.
- Tuning: newest matching run independently for every configured representation.
- Explicit run paths: the requested representation must be declared in that run's manifest.

Therefore, generating `all`, then `bert`, produces different runs; tuning can use both, while a training command for `bert` selects the BERT run rather than the globally newest unrelated run.

## Common failures

- `No preprocessing run contains representation`: run `src.preprocessing` for that representation and retry.
- `Model metadata not found`: retrain with the current command; older flat model directories are not part of the supported workflow.
- `Model artifact mismatch`: use the same representation and source run for training and evaluation.
- invalid configuration/model: inspect `config/models.yaml` and run `python -m src.models.tune --help`.
