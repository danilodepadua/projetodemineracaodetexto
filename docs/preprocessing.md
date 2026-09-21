# Preprocessing

## Purpose

`src.preprocessing` validates the competition files, cleans each split, fits train-only representations where needed, and writes a self-contained timestamped run. It does not train downstream regressors or create submissions.

## Prerequisites

Run from the repository root after following [setup.md](setup.md). Put these ignored files in `data/raw/`:

```text
data/raw/
├── train.csv
├── valid.csv
├── test.csv
└── sample_submission.csv
```

`train.csv` and `valid.csv` require `id`, `essay`, `prompt`, and the four targets. `test.csv` requires `id`, `essay`, and `prompt`. Missing files, invalid schemas, duplicate IDs, invalid scores, or incorrect reference split sizes stop the command before usable artifacts are produced.

## Lightweight command

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

`all` is intentionally lightweight and means exactly:

```text
bow, tf, tfidf, structural
```

It does not build Word2Vec, essay-prompt, or BERT features. Expected result: `artifacts/preprocessing/run-<UTC timestamp>/` with cleaned CSVs, reports, row IDs, a manifest, and one directory per representation.

## Explicit representations

Each invocation creates its own run. Run these separately when the representation is needed:

```bash
python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation word2vec_cbow

python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation word2vec_skipgram

python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation essay_prompt
```

`word2vec_cbow` and `word2vec_skipgram` train on training essays only. `essay_prompt` builds six dense essay/prompt relationship features using train-only TF-IDF and Word2Vec state.

BERT is explicit and may download `neuralmind/bert-base-portuguese-cased` to the standard Hugging Face cache:

```bash
python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation bert \
  --bert-model neuralmind/bert-base-portuguese-cased \
  --bert-device auto --bert-batch-size 4
```

Use `--bert-device cpu` when CUDA is unavailable or reproducibility requires CPU execution. A CUDA request on a host without CUDA fails with a device error; rerun with `--bert-device cpu` or `auto`.

## Fit boundary

Frequency and semantic representations follow:

```text
fit(train) -> transform(valid) -> transform(test)
```

Structural features are computed independently for each split. All representations use cleaned `essay_clean` text and preserve row order. Targets are never used to fit representation state.

## Generated run

```text
artifacts/preprocessing/run-<timestamp>/
├── train_clean.csv
├── valid_clean.csv
├── test_clean.csv
├── <representation>/
│   ├── X_train.*
│   ├── X_valid.*
│   └── X_test.*
├── row_ids/{train,valid,test}.json
├── reports/{validation_diagnostics,duplicate_essays,marker_inventory}.csv
└── manifest.json
```

BoW, TF, and TF-IDF use sparse `.npz` matrices and persist their vectorizer. Word2Vec, structural, essay-prompt, and BERT use dense `.npy` matrices. The manifest records shapes, dtypes, feature metadata, hashes, runtime versions, and paths relative to the run directory. Pretrained BERT weights are not copied into project artifacts.

## Common failures

- `Raw data file not found`: place the required files under `data/raw/`, or pass `--data-dir`.
- schema/row/ID validation error: restore the original competition columns and split contents.
- `Representation ... is unavailable`: generate that representation explicitly before tuning or model execution.
- BERT download/device failure: verify network access or rerun with an available device; do not commit the model cache.

Use `python -m src.preprocessing --help` for the complete current option list.
