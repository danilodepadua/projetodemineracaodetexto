# Testing

Run checks from the repository root with the selected environment interpreter. The test suite does not require competition data for its unit and fixture-based integration tests; `test_legacy_regression.py` skips unless local reference artifacts exist.

## Install and dependency check

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Expected result: no broken requirements.

## Complete quality suite

```bash
ruff format --check src tests
ruff check src tests
pyright
python -m pytest
python -m pip check
git diff --check
```

`requirements.txt` includes runtime, test, Ruff, Pyright, notebook, and representation dependencies, so the same commands work in raw Python, `.venv`, and the Dev Container.

## Focused tests

```bash
python -m pytest tests/test_preprocessing.py tests/test_validation.py
python -m pytest tests/test_pipeline.py tests/test_representation_contract.py
python -m pytest tests/test_model_data.py tests/test_model_config.py
python -m pytest tests/test_model_artifacts.py tests/test_cli.py
python -m pytest tests/test_word2vec.py tests/test_bert.py
```

The portable-execution tests cover repository-relative CLI defaults, representation-specific latest resolution, model artifact isolation, metadata portability, mismatch rejection, and module entrypoint help.

## CLI smoke test

```bash
python -m src.preprocessing --help
python -m src.models.train --help
python -m src.models.evaluate --help
python -m src.models.tune --help
```

## Data and real pipeline checks

If competition data is available, run the lightweight pipeline and record its run name:

```bash
python -m src.preprocessing \
  --data-dir data/raw \
  --output-dir artifacts/preprocessing \
  --representation all
```

Then run tuning and one matching train/evaluate flow from [models.md](models.md). Generate Word2Vec, essay-prompt, or BERT only when validating those representations; never claim a representation passed if it was not generated or loaded.

## Environment matrix

Record `PASS` only after executing the row. Use `N/A — not executed` with a reason otherwise.

| Check | Raw Python 3.12 | Host `.venv` | Fresh Dev Container | Reopen/rebuild |
| --- | --- | --- | --- | --- |
| install requirements |  |  | post-create | N/A or result |
| `pip check` |  |  |  |  |
| Ruff format/lint |  |  |  |  |
| Pyright |  |  |  |  |
| pytest |  |  |  |  |
| CLI help |  |  |  |  |
| lightweight preprocessing |  |  |  |  |
| tuning |  |  |  |  |
| train/evaluate |  |  |  |  |

## Clean-room rules

Use a temporary Python 3.12 virtual environment for raw-host validation; do not reuse `.venv` or a Dev Container environment. For container validation, rebuild and reopen rather than manually repairing `.venv`. Keep `data/`, `artifacts/`, Hugging Face caches, model binaries, and temporary outputs ignored.
