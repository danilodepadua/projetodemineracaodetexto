# Text Mining

Reproducible preprocessing and model experiments for the essay-scoring competition.

## Requirements

- Python 3.12.x
- Competition data placed under `data/raw/` (not committed)
- Linux, macOS, Windows PowerShell, or the project Dev Container

Use the host `.venv` workflow in [docs/setup.md](docs/setup.md) unless you intentionally manage a system Python installation.

## Quick start

Run commands from the repository root. On Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The Windows PowerShell equivalent is:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
```

Put the unmodified competition files here:

```text
data/raw/
├── train.csv
├── valid.csv
├── test.csv
└── sample_submission.csv
```

Generate the lightweight preprocessing run:

```bash
.venv/bin/python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

`all` means `bow`, `tf`, `tfidf`, and `structural`. Each invocation creates a timestamped run under `artifacts/preprocessing/run-<timestamp>/`.

Tune every configured representation that already has a run, then train and evaluate one representation:

```bash
.venv/bin/python -m src.models.tune \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --config config/models.yaml \
  --output artifacts/evaluation/tuning-all.json

.venv/bin/python -m src.models.train \
  --run-dir latest \
  --representation tfidf \
  --model ridge \
  --config config/models.yaml \
  --output-dir artifacts/models

.venv/bin/python -m src.models.evaluate \
  --run-dir latest \
  --representation tfidf \
  --model ridge \
  --models-dir artifacts/models \
  --output-dir artifacts/evaluation
```

Training and evaluation resolve `latest` to the newest run containing the requested representation. Tuning does the same independently for each configured representation; it never starts preprocessing automatically. Model artifacts are isolated by representation.

## Guides

- [Setup](docs/setup.md): system Python, `.venv`, Dev Container, and verification.
- [Pipeline](docs/pipeline.md): the complete execution sequence and outputs.
- [Preprocessing](docs/preprocessing.md): data contract and representation commands.
- [Models](docs/models.md): configuration, tuning, training, evaluation, and artifact compatibility.
- [Testing](docs/testing.md): quality checks and environment validation.
- [Configuration](docs/configuration.md): current defaults and precedence.
- [Model handoff](docs/model-handoff.md): loader contract for model code.
- [Architecture](docs/architecture.md): component boundaries.

Generated artifacts, raw data, virtual environments, and caches are ignored by Git. Do not commit competition data, model files, downloaded Hugging Face weights, or evaluation outputs.
