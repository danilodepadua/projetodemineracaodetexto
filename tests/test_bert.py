from types import SimpleNamespace

import numpy as np
import pytest
import torch
from src.representations import bert
from src.representations.bert import (
    BertConfig,
    build_bert,
    chunk_token_ids,
    masked_mean_pool,
    resolve_device,
    token_length_statistics,
)


class FakeTokenizer:
    model_max_length = 8

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": list(range(1, len(text.split()) + 1))}

    def build_inputs_with_special_tokens(self, payload):
        return [101, *payload, 102]

    def pad(self, encoded, padding=True, return_tensors="pt"):
        width = max(len(item["input_ids"]) for item in encoded)
        input_ids = []
        attention_mask = []
        for item in encoded:
            missing = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [0] * missing)
            attention_mask.append(item["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            max_position_embeddings=8,
            hidden_size=3,
            _commit_hash="fake-revision",
        )
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.training = True
        self.grad_enabled = []

    def parameters(self):
        return [self.parameter]

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, **batch):
        self.grad_enabled.append(torch.is_grad_enabled())
        values = batch["input_ids"].to(dtype=torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=values.repeat(1, 1, 3))


def _fake_huggingface(monkeypatch):
    models = []
    monkeypatch.setattr(
        bert.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer()
    )

    def load_model(*args, **kwargs):
        model = FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(bert.AutoModel, "from_pretrained", load_model)
    return models


def test_masked_mean_pool_excludes_padding_and_handles_zero_masks():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))

    zero = masked_mean_pool(hidden, torch.zeros_like(mask))
    assert torch.equal(zero, torch.zeros((1, 2)))


def test_chunk_token_ids_preserves_order_and_handles_empty():
    assert chunk_token_ids([], 3) == [[]]
    chunks = chunk_token_ids(list(range(7)), 3)
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]
    assert [token for chunk in chunks for token in chunk] == list(range(7))
    with pytest.raises(ValueError):
        chunk_token_ids([1], 0)


def test_token_length_statistics_reports_percentiles_and_overflow():
    stats = token_length_statistics([1, 2, 3, 10], max_length=3)
    assert stats["min"] == 1
    assert stats["max"] == 10
    assert stats["over_max"] == 25.0


def test_build_bert_is_batched_frozen_deterministic_and_finite(monkeypatch):
    models = _fake_huggingface(monkeypatch)
    config = BertConfig(model_name="fake", device="cpu", batch_size=2)
    first = build_bert(
        ["one two three four five six seven", ""],
        ["one two"],
        ["one two three"],
        config,
    )
    second = build_bert(
        ["one two three four five six seven", ""],
        ["one two"],
        ["one two three"],
        config,
    )

    assert first.train.shape == (2, 3)
    assert first.valid.shape == (1, 3)
    assert first.test.shape == (1, 3)
    assert np.isfinite(first.train).all()
    assert np.allclose(first.train, second.train)
    assert first.metadata["hidden_size"] == 3
    assert first.metadata["maximum_sequence_length"] == 8
    assert first.metadata["split_statistics"]["train"]["multi_chunk_documents"] == 1
    assert first.metadata["device_used"] == "cpu"
    assert first.metadata["frozen_feature_extractor"] is True
    assert all(model.training is False for model in models)
    assert all(model.grad_enabled == [False, False, False, False] for model in models)


def test_device_policy_uses_cpu_without_cuda():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError):
        if not torch.cuda.is_available():
            resolve_device("cuda")


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        BertConfig(device="tpu")
    with pytest.raises(ValueError):
        BertConfig(batch_size=0)
