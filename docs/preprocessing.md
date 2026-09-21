# Preprocessing

`src.preprocess` migrates the legacy notebook pipeline into a repeatable command. It processes the competition splits without training models.

## Inputs

Place the original files in ignored `data/raw/`:

```text
data/raw/
├── train.csv
├── valid.csv
├── test.csv
└── sample_submission.csv
```

`train.csv` and `valid.csv` must contain `id`, `essay`, `prompt`, and the four competition targets. `test.csv` must contain `id`, `essay`, and `prompt`. The pipeline enforces the reference split sizes (740/125/370), required files, exact columns, non-empty unique IDs, target values 1–5, and no shared IDs across splits.

## Run

```bash
python -m src.preprocess \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

`--representation` accepts `tf`, `tfidf`, or `all` (default). Each execution creates `artifacts/preprocessing/run-<UTC timestamp>/`.

## Cleaning

The implementation follows `notebooks/legacy/data_cleaning.ipynb`:

- normalizes Unicode to NFC and normalizes line endings;
- converts documented paragraph markers to line breaks;
- removes documented title, erasure, symbol, unknown, and out-of-line markers;
- collapses horizontal whitespace while preserving line breaks;
- preserves accents, capitalization, punctuation, spelling, and undocumented bracketed markers.

Clean datasets retain raw competition columns and add `essay_clean`, documented-marker counts, `undocumented_marker_count`, `clean_text_empty`, `character_count`, and `word_count`. The last two fields originate from the recomputation branch of the legacy preprocessing notebook and are absent from the older checked-in clean CSVs.

## Representations and leakage boundary

Every representation uses `essay_clean` and follows this boundary:

```text
fit(train) → transform(valid) → transform(test)
```

TF-IDF preserves the notebook configuration: `lowercase=False`, `strip_accents=None`, `stop_words=None`, `ngram_range=(1, 2)`, and `min_df=2`. Matrices remain sparse. TF is a user-approved `CountVectorizer` addition with the same lexical parameters; it is not claimed as notebook-equivalent behavior.

## Artifacts

```text
run-<timestamp>/
├── train_clean.csv
├── valid_clean.csv
├── test_clean.csv
├── tf/ or tfidf/
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

The manifest records input SHA-256 hashes, split row counts, marker definitions, vectorizer parameters, feature counts and shapes, and Python/library versions. CSVs and matrices are reloaded during generation to catch failed serialization.

## Reference material

The unmodified source notebooks are retained in `notebooks/legacy/`. Existing flat files in `data/` are legacy artifacts used by older model scripts; preprocessing writes new artifacts under `artifacts/preprocessing/` and does not overwrite them.
