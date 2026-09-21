# Complete pipeline guide

`src.preprocessing` provides repeatable data validation, cleaning, and text representations. BoW, TF, and TF-IDF are frequency-based; Word2Vec provides static distributed semantics; structural provides dense document-level linguistic features; essay-prompt provides a small dense representation of essay↔prompt relationships; BERT provides contextual semantic essay embeddings. It does not train downstream models or create competition submissions.

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

Use canonical `--representation bow`, `--representation tf`, `--representation tfidf`, `--representation word2vec_cbow`, `--representation word2vec_skipgram`, `--representation structural`, `--representation essay_prompt`, or `--representation bert` to build one representation. Structural features are inexpensive and require no fitting. Word2Vec, essay-prompt, and BERT features are explicit so their training or inference costs are predictable:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation word2vec_cbow

python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation word2vec_skipgram
```

`--representation essay_prompt` builds six dense relationship features and internally fits TF-IDF plus CBOW and Skip-Gram Word2Vec models on train essays only:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation essay_prompt
```

`--representation bert` uses the default Portuguese BERTimbau model and exposes only meaningful controls:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation bert \
  --bert-model neuralmind/bert-base-portuguese-cased \
  --bert-device auto \
  --bert-batch-size 4
```

`--representation all` builds BoW, TF, TF-IDF, and structural features after cleaning once; it intentionally excludes the more expensive Word2Vec, essay-prompt, and BERT features. Every execution creates `artifacts/preprocessing/run-<UTC timestamp>/`, writes row-ID alignment records and one manifest, and leaves the input files unchanged.

## 4. Understand the transformations

The cleaning rules reproduce `notebooks/legacy/data_cleaning.ipynb`: NFC Unicode and line-ending normalization; paragraph-marker line breaks; documented title, erasure, symbol, unknown, and out-of-line marker removal; and horizontal whitespace collapse. Accents, capitalization, punctuation, spelling, and undocumented markers remain intact.

Clean CSVs retain competition columns and add `essay_clean`, documented-marker counts, `undocumented_marker_count`, `clean_text_empty`, `character_count`, and `word_count`. The final two fields come from the recomputation branch of the legacy preprocessing notebook. Structural features consume these canonical columns; cleaning remains responsible for marker detection and text normalization.

Every text representation uses `essay_clean` with a strict boundary:

```text
fit(train) → transform(valid) → transform(test)
```

BoW, TF, and TF-IDF share `lowercase=False`, `strip_accents=None`, `stop_words=None`, `ngram_range=(1, 2)`, and `min_df=2`.

- **BoW** uses raw `CountVectorizer` document-term counts.
- **TF** uses the same counts with L2 row normalization and no IDF weighting (`use_idf=False`). The previous PR0 `tf` implementation produced raw counts; that behavior is now named BoW rather than silently retained as TF.
- **TF-IDF** applies term frequency multiplied by inverse document frequency. It preserves the legacy notebook parameters and learns vocabulary and IDF statistics from train only.
- **Word2Vec** uses Gensim with explicit CBOW (`sg=0`) or Skip-Gram (`sg=1`) configuration. Tokenization extracts Unicode word tokens from `essay_clean` without lowercasing, accent stripping, stemming, lemmatization, or stopword removal. The model is trained on train tokens only; document vectors use simple mean pooling over known word vectors. OOV tokens are ignored, and documents with no known tokens receive a deterministic zero vector. The default `workers=1` favors reproducibility over throughput; `vector_size`, `window`, `min_count`, `epochs`, and `seed` are recorded in the manifest.
- **Structural** produces a fixed, explicitly ordered dense vector: `character_count`, `word_count`, `sentence_count`, `paragraph_count`, `unique_word_count`, `lexical_diversity`, `average_word_length`, `average_sentence_length_words`, `average_paragraph_length_words`, `punctuation_count`, `comma_count`, `sentence_terminal_count`, `uppercase_ratio`, `digit_count`, and the seven cleaning marker counts. It counts sentences from terminal-punctuation runs with a one-sentence fallback for non-empty text, paragraphs from non-empty lines in canonical `essay_clean`, and tokens with Unicode `\\b\\w+\\b` without lowercasing, accent stripping, stemming, lemmatization, or stopword removal. Empty denominators produce zero. Features are raw and no scaler is fitted.
- **Essay ↔ Prompt** uses this deterministic feature order: `tfidf_cosine_similarity`, `word2vec_cbow_cosine_similarity`, `word2vec_skipgram_cosine_similarity`, `shared_token_count`, `essay_prompt_token_jaccard`, and `prompt_token_coverage`. Prompts are transformed with `clean_text` in memory; the original `prompt` column is not changed. The first feature uses a TF-IDF vectorizer fit on train `essay_clean` only and transforms both essay and prompt text. The semantic features use separate CBOW and Skip-Gram models fit on train essay tokens only, then mean-pool known tokens for both fields. OOV tokens are ignored; a zero-norm comparison returns `0.0`. Shared-token count uses unique case-preserving Unicode tokens; Jaccard is intersection divided by union; coverage is intersection divided by prompt tokens. Empty denominators produce zero. No target column is read.
- **BERT** uses `AutoTokenizer` and `AutoModel` with configurable `neuralmind/bert-base-portuguese-cased` default. It receives canonical `essay_clean` text and performs transform-only inference on all three splits; the pretrained model is never fitted on project texts or targets. Token lengths are measured before inference. Payloads fit within the model maximum after reserving special tokens; essays over the limit use ordered, non-overlapping chunks. Each chunk uses masked mean pooling over valid non-padding token states, then chunk vectors are mean-pooled into one fixed-size dense vector. Hidden size comes from model configuration. `auto` selects CUDA when visible and otherwise CPU; explicit `cpu` is always supported. Standard Hugging Face cache stores downloads; weights are not copied into project artifacts. Metadata records identifiers/revision, maximum length, pooling/chunk strategy, batch/device configuration, split shapes, token statistics, and chunk counts.

Frequency, semantic, and essay-prompt fitted components use `fit(train) → transform(valid) → transform(test)`. Structural generation has no fitting step; all split rows are computed independently with the same ordered definitions.

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
├── word2vec_cbow/
│   ├── X_train.npy
│   ├── X_valid.npy
│   ├── X_test.npy
│   └── model.model
├── word2vec_skipgram/
│   ├── X_train.npy
│   ├── X_valid.npy
│   ├── X_test.npy
│   └── model.model
├── structural/
│   ├── X_train.npy
│   ├── X_valid.npy
│   ├── X_test.npy
│   └── feature_names.json
├── bert/
│   ├── X_train.npy
│   ├── X_valid.npy
│   └── X_test.npy
├── essay_prompt/
│   ├── X_train.npy
│   ├── X_valid.npy
│   ├── X_test.npy
│   ├── feature_names.json
│   ├── tfidf_vectorizer.joblib
│   ├── word2vec_cbow.model
│   └── word2vec_skipgram.model
├── row_ids/
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── reports/
│   ├── validation_diagnostics.csv
│   ├── duplicate_essays.csv
│   └── marker_inventory.csv
└── manifest.json
```

`manifest.json` records input hashes, row counts, marker definitions, canonical representation names, family/storage/dtype/shape metadata, configurations, feature definitions, row-ID alignment, artifact hashes, artifact paths, and runtime versions. BoW/TF/TF-IDF matrices remain sparse and use `.npz`; Word2Vec, structural, essay-prompt, and BERT matrices are dense `.npy` arrays. Structural and essay-prompt artifacts store ordered `feature_names.json`. Essay-prompt metadata records train-only TF-IDF configuration, train-only CBOW/Skip-Gram configuration, tokenization and prompt-cleaning strategies, feature definitions, shapes, artifact paths, and zero-norm comparison counts. BERT artifacts contain only the three embedding arrays; pretrained weights remain in the standard Hugging Face cache and are identified through manifest metadata rather than copied into the run.

## 6. Legacy model commands

Model scripts live in `src.models` and are outside this representation PR. The preprocessing run is the stable boundary that supplies cleaned datasets, sparse or dense matrices, fitted representation state, and manifest metadata to later model work:

```bash
python -m src.models.train --help
python -m src.models.evaluate --help
python -m src.models.tune --help
```

## Reference material

Unmodified source notebooks are retained in `notebooks/legacy/`. Preprocessing output is written only under `artifacts/preprocessing/` and never overwrites legacy artifacts.
