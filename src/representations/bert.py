from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ._common import Representation

DEFAULT_BERT_MODEL = "neuralmind/bert-base-portuguese-cased"


@dataclass(frozen=True)
class BertConfig:
    """Configuration for frozen BERT feature extraction."""

    model_name: str = DEFAULT_BERT_MODEL
    revision: str | None = None
    device: str = "auto"
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def parameters(self) -> dict[str, Any]:
        """Return serializable extraction parameters."""
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic CPU/CUDA device policy."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_length_statistics(
    lengths: Sequence[int], max_length: int
) -> dict[str, float | int]:
    """Summarize tokenizer lengths and the percentage over the model limit."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        return {
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
            "over_max": 0.0,
        }
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_max": float(np.mean(values > max_length) * 100),
    }


def chunk_token_ids(token_ids: Sequence[int], payload_size: int) -> list[list[int]]:
    """Split token IDs into ordered, non-overlapping payload chunks."""
    if payload_size < 1:
        raise ValueError("payload_size must be positive")
    if not token_ids:
        return [[]]
    return [
        list(token_ids[start : start + payload_size])
        for start in range(0, len(token_ids), payload_size)
    ]


def masked_mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool valid token states, returning zeros for all-zero masks."""
    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden states must be 3-D and attention masks 2-D")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states and attention mask shapes do not match")
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return summed / counts


def _model_max_length(tokenizer: Any, model: Any) -> int:
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = getattr(
        getattr(model, "config", None), "max_position_embeddings", None
    )
    limits = [
        int(limit)
        for limit in (tokenizer_limit, model_limit)
        if isinstance(limit, int) and 0 < limit < 1_000_000
    ]
    if not limits:
        raise ValueError("Could not determine a finite model maximum sequence length")
    return min(limits)


def _tokenize_documents(
    tokenizer: Any, texts: Sequence[str], max_length: int
) -> tuple[list[list[list[int]]], list[int]]:
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    payload_size = max_length - special_tokens
    if payload_size < 1:
        raise ValueError("Model maximum sequence length leaves no room for tokens")

    chunks_by_document: list[list[list[int]]] = []
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        token_ids = list(encoded["input_ids"])
        lengths.append(len(token_ids))
        chunks_by_document.append(chunk_token_ids(token_ids, payload_size))
    return chunks_by_document, lengths


def _batch_chunks(
    tokenizer: Any,
    chunk_payloads: Sequence[Sequence[int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = [
        {
            "input_ids": tokenizer.build_inputs_with_special_tokens(list(payload)),
            "attention_mask": [1]
            * len(tokenizer.build_inputs_with_special_tokens(list(payload))),
        }
        for payload in chunk_payloads
    ]
    padded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    return {
        key: value.to(device) for key, value in padded.items() if torch.is_tensor(value)
    }


def _embed_split(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    chunks_by_document, lengths = _tokenize_documents(tokenizer, texts, max_length)
    flat_chunks = [chunk for chunks in chunks_by_document for chunk in chunks]
    chunk_vectors: list[np.ndarray] = []
    for start in range(0, len(flat_chunks), batch_size):
        batch = _batch_chunks(
            tokenizer, flat_chunks[start : start + batch_size], device
        )
        with torch.no_grad():
            output = model(**batch)
        hidden = output.last_hidden_state
        pooled = masked_mean_pool(hidden, batch["attention_mask"])
        chunk_vectors.append(pooled.detach().cpu().numpy())

    all_chunks = (
        np.concatenate(chunk_vectors, axis=0)
        if chunk_vectors
        else np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
    )

    vectors_list: list[np.ndarray] = []
    offset = 0
    for chunks in chunks_by_document:
        count = len(chunks)
        vectors_list.append(np.mean(all_chunks[offset : offset + count], axis=0))
        offset += count
    vectors = np.asarray(vectors_list, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise AssertionError("BERT embeddings contain NaN or Inf")
    chunk_counts = [len(chunks) for chunks in chunks_by_document]
    return vectors, {
        "token_lengths": token_length_statistics(lengths, max_length),
        "multi_chunk_documents": int(sum(count > 1 for count in chunk_counts)),
        "max_chunks": max(chunk_counts, default=0),
        "mean_chunks": float(np.mean(chunk_counts)) if chunk_counts else 0.0,
    }


def build_bert(
    train_texts: Iterable[str],
    valid_texts: Iterable[str],
    test_texts: Iterable[str],
    config: BertConfig | None = None,
) -> Representation:
    """Extract frozen contextual essay vectors from a pretrained BERT model."""
    config = config or BertConfig()
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.revision
    )
    model = AutoModel.from_pretrained(config.model_name, revision=config.revision)
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    max_length = _model_max_length(tokenizer, model)
    splits = {
        "train": list(train_texts),
        "valid": list(valid_texts),
        "test": list(test_texts),
    }
    vectors: dict[str, np.ndarray] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    for split, texts in splits.items():
        vectors[split], split_stats[split] = _embed_split(
            model,
            tokenizer,
            texts,
            max_length,
            config.batch_size,
            device,
        )

    hidden_size = int(model.config.hidden_size)
    shapes = {split: list(matrix.shape) for split, matrix in vectors.items()}
    metadata: dict[str, Any] = {
        "representation": "bert",
        "model_identifier": config.model_name,
        "model_revision": config.revision
        or getattr(getattr(model, "config", None), "_commit_hash", None),
        "tokenizer_identifier": config.model_name,
        "hidden_size": hidden_size,
        "pooling": (
            "masked mean over valid non-padding token states, then mean over chunks"
        ),
        "chunk_strategy": "non-overlapping token chunks",
        "maximum_sequence_length": max_length,
        "batch_size": config.batch_size,
        "device_requested": config.device,
        "device_used": str(device),
        "frozen_feature_extractor": True,
        "fit_scope": "pretrained model; transform-only for train, valid, and test",
        "split_statistics": split_stats,
        "shapes": shapes,
        "matrix_shapes": shapes,
        **config.parameters(),
    }
    return Representation(
        vectorizer=config,
        train=vectors["train"],
        valid=vectors["valid"],
        test=vectors["test"],
        metadata=metadata,
    )
