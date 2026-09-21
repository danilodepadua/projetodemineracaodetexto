# Text Mining

Reproducible preprocessing for the essay-scoring competition. PR0 migrates the legacy notebooks into Python modules; model training and submission workflows remain separate legacy tooling.

## Quick start

Open the repository in the devcontainer, then run:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

The command validates the supplied competition files, cleans every split independently, fits each representation on training essays only, and writes a timestamped artifact run. See [preprocessing](docs/preprocessing.md) for inputs, outputs, validation, and reproducibility details.

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
- [Preprocessing and artifacts](docs/preprocessing.md)
- [Legacy training](docs/trainning.md)
- [Legacy evaluation](docs/evaluate.md)
- [Legacy tuning](docs/tune.md)
- [Model configuration](config/models.yaml)
- [Environment configuration](docs/configuration.md)
