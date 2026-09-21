# Model handoff

Preprocessing ends at a timestamped run. Model code loads matrices and targets through `load_experiment_data`; it does not open `.npy`, `.npz`, CSV, Joblib, or Gensim artifacts directly. Run commands are documented in [models.md](models.md).

## Catalog

```python
from src.representations import REPRESENTATION_NAMES

print(REPRESENTATION_NAMES)
# bow, tf, tfidf, word2vec_cbow, word2vec_skipgram,
# structural, essay_prompt, bert
```

Generate one representation inside the Dev Container:

```bash
python -m src.preprocessing --data-dir data/raw \
  --output-dir artifacts/preprocessing --representation tfidf
```

`all` intentionally generates the lightweight default set: `bow`, `tf`, `tfidf`, and `structural`. Expensive `word2vec_cbow`, `word2vec_skipgram`, `essay_prompt`, and `bert` runs are explicit.

## Load an experiment dataset

```python
from src.models.data import (
    load_experiment_data,
    resolve_representation_run_or_raise,
)

run = resolve_representation_run_or_raise("latest", "tfidf")
data = load_experiment_data(run, "tfidf")

X_train, y_train = data.X_train, data.y_train
X_valid, y_valid = data.X_valid, data.y_valid
X_test = data.X_test
```

`y_train` and `y_valid` are mappings keyed by the four independent target names: `formal_register`, `thematic_coherence`, `narrative_rhetorical_structure`, and `cohesion`. No synthetic target is created.

## Storage and metadata

| Representation | Family | Matrix type |
| --- | --- | --- |
| `bow`, `tf`, `tfidf` | frequency | SciPy sparse |
| `word2vec_cbow`, `word2vec_skipgram` | static semantic | NumPy dense |
| `structural` | structural | NumPy dense |
| `essay_prompt` | relationship | NumPy dense |
| `bert` | contextual | NumPy dense |

`data.metadata` contains the manifest entry, including family, storage, dtype, feature dimension, shapes, configuration, artifact paths, and representation-specific feature names or model details. Rows are checked against persisted split IDs before loading. Paths inside manifests are relative to the preprocessing run.

Model code must not assume a specific feature dimension, scaling policy, tokenizer, vocabulary, or serialization format. Sparse matrices remain sparse. Combine representations explicitly with `scipy.sparse.hstack` for compatible sparse blocks, `numpy.hstack` for dense blocks, or an intentional sparse/dense conversion at the small-block boundary.

The compatibility input `word2vec` remains accepted for older commands, but new manifests and documentation use `word2vec_cbow` and `word2vec_skipgram`.

## Model configuration and tuning

`config/models.yaml` is the model-team source of truth for the four targets, representation candidates, enabled model names, fixed parameters, and tunable parameter grids. Tuning iterates every configured representation and model, selecting the newest run that contains each representation:

```bash
python -m src.models.tune \
  --run-dir latest \
  --runs-dir artifacts/preprocessing \
  --config config/models.yaml \
  --output artifacts/evaluation/tuning-all.json
```

Results are organized as `representation → model → target → hyperparameters and metrics`. Missing or incomplete representation artifacts are recorded as `status: skipped` with a reason; tuning never falls back to TF-IDF. Use `--representation tfidf` to run one candidate only.

The preprocessing pipeline remains a separate artifact-producing stage because representations have different costs. Generate representations first, then tune existing artifacts. All model commands consume the same validated handoff loader, so sparse and dense representations use the same model boundary.

Training, evaluation, and tuning all use `load_experiment_data`; no model command reads representation files directly. Trained bundles live under `artifacts/models/<representation>/<model>/` and include metadata identifying the source run, representation, model, parameters, and targets. Evaluation rejects a bundle whose metadata does not match the requested run and representation.
