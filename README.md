# Text Mining

Reproducible preprocessing for the essay-scoring competition.

## Quick start

Open the repository in the devcontainer, then run:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

The command validates the supplied competition files, cleans every split independently, builds the selected representation from cleaned essays, and writes a timestamped artifact run. Canonical names are `bow`, `tf`, `tfidf`, `word2vec_cbow`, `word2vec_skipgram`, `structural`, `essay_prompt`, and `bert`. `all` builds the lightweight `bow`, `tf`, `tfidf`, and `structural` set; the other representations remain explicit because they are more expensive or specialized. See [preprocessing](docs/preprocessing.md) and the [model-team handoff](docs/model-handoff.md).

To build frozen Portuguese BERT contextual essay embeddings explicitly:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation bert \
  --bert-model neuralmind/bert-base-portuguese-cased \
  --bert-device auto \
  --bert-batch-size 4
```

The first run downloads the configured Hugging Face model into the standard local cache. BERT uses the cleaned `essay_clean` text, subword tokenization, non-overlapping chunks when needed, masked token mean pooling, and mean pooling across chunks. It is a frozen feature extractor: no target labels, fine-tuning, regression head, or model weights are written to project artifacts.

## Tune all configured representations

After generating the supported preprocessing artifacts, run the configured model grids across every available representation:

```bash
python -m src.models.tune \
--run-dir latest \
--runs-dir artifacts/preprocessing \
--config config/models.yaml \
--output artifacts/evaluation/tuning-all.json
```

Results are grouped by representation, model, target, and hyperparameter configuration. Missing representation artifacts are reported as skipped; tuning never regenerates or substitutes them. See [model-team handoff](docs/model-handoff.md).

To build the six dense essay↔prompt relationship features:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation essay_prompt
```

## Representation layers

The project supports five representation families:

- **Frequency-based:** BoW, TF, and TF-IDF.
- **Semantic:** Word2Vec CBOW and Skip-Gram.
- **Structural / linguistic:** dense document-level features for length, organization, lexical diversity, punctuation, casing, digits, and cleaning markers.
- **Essay ↔ Prompt:** lexical overlap, train-only TF-IDF cosine similarity, and train-essay-only Word2Vec CBOW/Skip-Gram cosine similarity.
- **Contextual semantic:** frozen Portuguese BERTimbau essay embeddings with model-derived hidden size, CPU-safe automatic device selection, configurable batching, and manifest-recorded chunk statistics.

The progression is frequency-based representations → static Word2Vec semantics → contextual BERT semantics, alongside structural and essay-prompt features. The default `all` command intentionally remains lightweight and does not download or run Word2Vec, essay-prompt, or BERT inference.

Structural features are raw, deterministic, and unscaled. Word2Vec is a dense distributed semantic representation trained with Gensim. It learns word vectors from training essays only and mean-pools known word vectors into one fixed-size vector per essay. Unknown tokens are ignored; an essay with no known tokens receives a zero vector. Essay-prompt cosine similarities use the same tokenization on cleaned prompt views, return `0.0` for zero-norm comparisons, and never use target values.

## Local data layout

```text
data/
├── raw/                         # ignored local competition inputs
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── sample_submission.csv
└── ...                           # legacy local artifacts
notebooks/legacy/                 # copied notebook reference implementation
artifacts/preprocessing/          # ignored generated runs
```

`data/raw/` and generated artifacts are intentionally ignored. Do not commit competition data or generated matrices.

## Environment

The devcontainer installs the dependencies from `requirements.txt`. Outside the container:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Run tests with:

```bash
python -m pytest
```

## Code quality

The repository enforces Python 3.12, 88-character lines, LF endings, four-space indentation, Ruff formatting/linting, and Pyright standard type checking. Run the same checks locally before opening a pull request:

```bash
ruff format --check src tests
ruff check src tests
pyright
python -m pytest
```

Apply safe formatting and lint fixes with `ruff format src tests` and `ruff check --fix src tests`. `.editorconfig` defines editor behavior; `.gitattributes` normalizes Git text files and treats notebooks/artifacts as binary.

## Guides

- [Complete current pipeline](docs/pipeline.md)
- [Model-team handoff](docs/model-handoff.md)
- [Architecture](docs/architecture.md)
- [Testing](docs/testing.md)
- [Preprocessing and artifacts](docs/preprocessing.md)
- [Legacy training](docs/trainning.md)
- [Legacy evaluation](docs/evaluate.md)
- [Legacy tuning](docs/tune.md)
- [Model configuration](config/models.yaml)
- [Environment configuration](docs/configuration.md)
