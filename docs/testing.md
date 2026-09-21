# Testing

Run commands from the repository root inside the devcontainer.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the complete suite

```bash
python -m pytest
```

The suite contains unit tests for cleaning, validation, train-only vectorizer fitting, serialization, and end-to-end artifact generation.

## Run focused tests

```bash
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_validation.py
python -m pytest tests/test_pipeline.py
```

## Run notebook-regression checks

The legacy comparison test verifies cleaned text, marker counts, TF-IDF vocabulary, matrix shapes, sparsity, and values against the copied reference inputs and existing legacy artifacts.

```bash
python -m pytest tests/test_legacy_regression.py
```

It runs only when these local ignored files exist:

```text
data/raw/train.csv
data/raw/valid.csv
data/raw/test.csv
data/train_limpo.csv
data/valid_limpo.csv
data/test_limpo.csv
data/tfidf_vectorizer.joblib
data/X_train_tfidf.npz
data/X_valid_tfidf.npz
data/X_test_tfidf.npz
```

A warning about loading the legacy vectorizer with a newer scikit-learn version is expected during this compatibility comparison.

## Run quality checks

```bash
ruff format --check src tests
ruff check src tests
pyright
python -m pytest
```

Apply automatic style fixes before rerunning the checks:

```bash
ruff format src tests
ruff check --fix src tests
```
