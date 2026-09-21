# Current pipeline run

All model commands consume a timestamped preprocessing run.

```bash
python -m src.preprocessing --data-dir data/raw --output-dir artifacts/preprocessing --representation tfidf
python -m src.models.train --run-dir latest --representation tfidf --model ridge --output-dir artifacts/models
python -m src.models.evaluate --run-dir latest --representation tfidf --model ridge --models-dir artifacts/models --output-dir artifacts/evaluation
python -m src.models.tune --run-dir latest --runs-dir artifacts/preprocessing --output artifacts/evaluation/tuning-all.json
```

The preprocessing run is the source of cleaned datasets, native sparse or dense matrices, representation state, row-ID alignment, and manifest metadata. Model code should call `load_experiment_data` from `src.models.data` instead of opening artifacts directly; the loader validates the requested canonical representation before returning train/valid/test matrices and independent target arrays.

The full handoff sequence is:

```text
preprocess configured representations
  -> discover available runs per representation
  -> load validated data with src.models.data
  -> tune configured models and grids
  -> compare representation/model/target results
```

`config/models.yaml` controls representation candidates, model factories, and hyperparameter grids. `src.models.tune` consumes those grids and existing preprocessing artifacts; it reports unavailable candidates as skipped and never regenerates or silently substitutes representations.

See [model-handoff.md](model-handoff.md) for the public model-team API and [architecture.md](architecture.md) for ownership and boundaries.
