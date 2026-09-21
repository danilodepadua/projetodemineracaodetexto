# Current pipeline run

All model commands consume a timestamped preprocessing run.

```bash
python -m src.preprocessing --data-dir data/raw --output-dir artifacts/preprocessing --representation tfidf
python -m src.models.train --run-dir artifacts/preprocessing/<run-id> --representation tfidf --model ridge --output-dir artifacts/models
python -m src.models.evaluate --run-dir artifacts/preprocessing/<run-id> --representation tfidf --model ridge --models-dir artifacts/models --output-dir artifacts/evaluation
python -m src.models.tune --run-dir artifacts/preprocessing/<run-id> --representation tfidf --output artifacts/evaluation/tuning.json
```

The preprocessing run is the source of clean datasets, sparse matrices, vectorizers, and manifest metadata. Models validate the requested representation before loading it.
