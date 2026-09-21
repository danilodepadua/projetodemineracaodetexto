# Model-team handoff

Preprocessing ends at a timestamped run. Model code loads matrices and targets through the stable handoff API; it does not open `.npy`, `.npz`, CSV, Joblib, or Gensim artifacts directly.

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
from src.models.data import load_experiment_data, resolve_run_dir

run = resolve_run_dir("latest")
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

`data.metadata` contains the manifest entry, including family, storage, dtype, feature dimension, shapes, configuration, artifact paths, and representation-specific feature names or model details. Rows are checked against persisted split IDs before loading.

Model code must not assume a specific feature dimension, scaling policy, tokenizer, vocabulary, or serialization format. Sparse matrices remain sparse. Combine representations explicitly with `scipy.sparse.hstack` for compatible sparse blocks, `numpy.hstack` for dense blocks, or an intentional sparse/dense conversion at the small-block boundary.

The compatibility input `word2vec` remains accepted for older commands, but new manifests and documentation use `word2vec_cbow` and `word2vec_skipgram`.
