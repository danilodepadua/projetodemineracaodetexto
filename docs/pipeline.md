# Pipeline

Run every command from the repository root with the same Python interpreter used during setup. The pipeline is intentionally staged: raw data is validated before preprocessing, preprocessing creates immutable timestamped runs, and model commands consume those runs through `load_experiment_data`.

## 1. Validate raw data and build lightweight representations

Prerequisite: install dependencies and place the four files described in [setup.md](setup.md) under `data/raw/`.

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

Expected result: a new `artifacts/preprocessing/run-<UTC timestamp>/` containing cleaned splits, validation reports, row IDs, a manifest, and `bow`, `tf`, `tfidf`, and `structural` artifacts. Inputs are not modified.

If files are missing, fix `data/raw/` and rerun. If the schemas or reference row counts are invalid, fix the source files rather than editing generated artifacts.

## 2. Build expensive representations when needed

Each command creates a separate timestamped run:

```bash
python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation word2vec_cbow

python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation word2vec_skipgram

python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation essay_prompt

python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation bert \
  --bert-model neuralmind/bert-base-portuguese-cased \
  --bert-device auto --bert-batch-size 4
```

Word2Vec and essay-prompt use train-only fitting. BERT may download its pretrained model to the standard user cache; weights are not copied into project artifacts.

## 3. Tune existing runs

```bash
python -m src.models.tune \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --config config/models.yaml \
  --output artifacts/evaluation/tuning-all.json
```

Tuning reads representations already present on disk. `latest` means the newest run containing each requested representation, not one combined run. Missing representations are written as explicit `status: skipped`; preprocessing is never triggered automatically.

For one representation:

```bash
python -m src.models.tune \
  --run-dir latest --runs-dir artifacts/preprocessing \
  --representation tfidf --config config/models.yaml \
  --output artifacts/evaluation/tuning-tfidf.json
```

## 4. Train one selected model

```bash
python -m src.models.train \
  --run-dir latest --runs-dir artifacts/preprocessing \
  --representation tfidf --model ridge \
  --config config/models.yaml --output-dir artifacts/models
```

Training resolves the newest run containing `tfidf` and writes `artifacts/models/tfidf/ridge/`. The directory contains one joblib per target and `metadata.json` with the representation, source run name, model, parameters, and targets.

## 5. Evaluate the matching artifacts

```bash
python -m src.models.evaluate \
  --run-dir latest --runs-dir artifacts/preprocessing \
  --representation tfidf --model ridge \
  --models-dir artifacts/models --output-dir artifacts/evaluation
```

Evaluation resolves the same representation-aware run and checks model metadata before prediction. Results are written to `artifacts/evaluation/tfidf/ridge.json`. A mismatched source run, representation, model, target list, or missing artifact fails with an actionable error; it does not fall through to a scikit-learn feature-count error.

## Output locations

```text
artifacts/
├── preprocessing/run-<timestamp>/
├── models/<representation>/<model>/
└── evaluation/<representation>/<model>.json
```

All paths recorded inside preprocessing manifests and model/tuning metadata are run-relative identifiers, not producer-machine absolute paths.
