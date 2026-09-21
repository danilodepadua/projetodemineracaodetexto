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

The command validates the supplied competition files, cleans every split independently, builds each representation from the cleaned essays, and writes a timestamped artifact run. Select `--representation bow`, `--representation tf`, `--representation tfidf`, `--representation structural`, or `--representation word2vec`; choose Word2Vec's architecture with `--word2vec-architecture cbow` or `--word2vec-architecture skipgram`. `all` builds BoW, TF, TF-IDF, and structural features, while Word2Vec remains explicit. See [preprocessing](docs/preprocessing.md) for representation semantics, artifacts, validation, and reproducibility details.

## Representation layers

The project supports three representation families:

- **Frequency-based:** BoW, TF, and TF-IDF.
- **Semantic:** Word2Vec CBOW and Skip-Gram.
- **Structural / linguistic:** dense document-level features for length, organization, lexical diversity, punctuation, casing, digits, and cleaning markers.

Structural features are raw, deterministic, and unscaled. Word2Vec is a dense distributed semantic representation trained with Gensim. It learns word vectors from training essays only and mean-pools known word vectors into one fixed-size vector per essay. Unknown tokens are ignored; an essay with no known tokens receives a zero vector.

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
- [Testing](docs/testing.md)
- [Preprocessing and artifacts](docs/preprocessing.md)
- [Legacy training](docs/trainning.md)
- [Legacy evaluation](docs/evaluate.md)
- [Legacy tuning](docs/tune.md)
- [Model configuration](config/models.yaml)
- [Environment configuration](docs/configuration.md)
