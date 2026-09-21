# Current pipeline run

All model commands consume a timestamped preprocessing run.

```bash
python -m src.preprocessing --data-dir data/raw --output-dir artifacts/preprocessing --representation tfidf
python -m src.models.train --run-dir latest --representation tfidf --model ridge --output-dir artifacts/models
python -m src.models.evaluate --run-dir latest --representation tfidf --model ridge --models-dir artifacts/models --output-dir artifacts/evaluation
python -m src.models.tune --run-dir latest --representation tfidf --output artifacts/evaluation/tuning.json
```

The preprocessing run is the source of cleaned datasets, native sparse or dense matrices, representation state, row-ID alignment, and manifest metadata. Model code should call `load_experiment_data` from `src.models.data` instead of opening artifacts directly; the loader validates the requested canonical representation before returning train/valid/test matrices and independent target arrays.

The full handoff sequence is:

```text
preprocess one canonical representation
  -> load validated run with src.models.data
  -> train/evaluate/tune using config/models.yaml
  -> compare per-target results
```

`config/models.yaml` controls model factories and hyperparameter grids. `src.models.tune` consumes those grids and the selected preprocessing artifact; it does not regenerate representations or use a separate model list.

See [model-handoff.md](model-handoff.md) for the public model-team API and [architecture.md](architecture.md) for ownership and boundaries.
