# Complete pipeline guide

`src.preprocessing` provides repeatable data validation, cleaning, and traditional text representations. It does not train models or create competition submissions.

## 1. Prepare the environment

Run all commands from the repository root inside the devcontainer:

```bash
python -m pip install -r requirements.txt
ruff format --check src tests
ruff check src tests
pyright
python -m pytest
```

## 2. Provide the original competition data

Place unmodified source files in ignored `data/raw/`:

```text
data/raw/
├── train.csv
├── valid.csv
├── test.csv
└── sample_submission.csv
```

`train.csv` and `valid.csv` require `id`, `essay`, `prompt`, and the four competition targets. `test.csv` requires `id`, `essay`, and `prompt`. Validation enforces reference row counts (740/125/370), exact schemas, unique non-empty IDs, scores 1–5, and no IDs shared across splits.

## 3. Run preprocessing

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

Use `--representation bow`, `--representation tf`, or `--representation tfidf` to build one representation. `--representation all` builds all three after cleaning once. Every execution creates `artifacts/preprocessing/run-<UTC timestamp>/` and leaves the input files unchanged.

## 4. Understand the transformations

The cleaning rules reproduce `notebooks/legacy/data_cleaning.ipynb`: NFC Unicode and line-ending normalization; paragraph-marker line breaks; documented title, erasure, symbol, unknown, and out-of-line marker removal; and horizontal whitespace collapse. Accents, capitalization, punctuation, spelling, and undocumented markers remain intact.

Clean CSVs retain competition columns and add `essay_clean`, documented-marker counts, `undocumented_marker_count`, `clean_text_empty`, `character_count`, and `word_count`. The final two fields come from the recomputation branch of the legacy preprocessing notebook.

Every representation uses `essay_clean` with a strict boundary:

```text
fit(train) → transform(valid) → transform(test)
```

All three representations share `lowercase=False`, `strip_accents=None`, `stop_words=None`, `ngram_range=(1, 2)`, and `min_df=2`.

- **BoW** uses raw `CountVectorizer` document-term counts.
- **TF** uses the same counts with L2 row normalization and no IDF weighting (`use_idf=False`). The previous PR0 `tf` implementation produced raw counts; that behavior is now named BoW rather than silently retained as TF.
- **TF-IDF** applies term frequency multiplied by inverse document frequency. It preserves the legacy notebook parameters and learns vocabulary and IDF statistics from train only.

Every representation follows `fit(train) → transform(valid) → transform(test)`; validation and test text never expand the vocabulary or alter learned weights.

## 5. Inspect the generated run

```text
run-<timestamp>/
├── train_clean.csv
├── valid_clean.csv
├── test_clean.csv
├── bow/
│   ├── X_train.npz
│   ├── X_valid.npz
│   ├── X_test.npz
│   └── vectorizer.joblib
├── tf/
│   ├── X_train.npz
│   ├── X_valid.npz
│   ├── X_test.npz
│   └── vectorizer.joblib
├── tfidf/
│   ├── X_train.npz
│   ├── X_valid.npz
│   ├── X_test.npz
│   └── vectorizer.joblib
├── reports/
│   ├── validation_diagnostics.csv
│   ├── duplicate_essays.csv
│   └── marker_inventory.csv
└── manifest.json
```

`manifest.json` records input hashes, row counts, marker definitions, representation names, vectorizer types, configurations, vocabulary sizes, feature dimensions, split shapes, artifact paths, and runtime versions. Matrices remain sparse and CSV/matrix round trips are checked while writing.

## 6. Legacy model commands

Model scripts live in `src.models` and are outside this representation PR. The preprocessing run is the stable boundary that supplies cleaned datasets, sparse matrices, fitted vectorizer state, and manifest metadata to later model work:

```bash
python -m src.models.train --help
python -m src.models.evaluate --help
python -m src.models.tune --help
```

## Reference material

Unmodified source notebooks are retained in `notebooks/legacy/`. Preprocessing output is written only under `artifacts/preprocessing/` and never overwrites legacy artifacts.
