import math
import torch
from analysis.metrics import extract_metrics, _compute_sparsity, _compute_l2, _compute_velocity


def test_compute_l2():
    h = torch.tensor([[3.0, 4.0]])
    assert _compute_l2(h) == 5.0


def test_compute_sparsity():
    dim = 4
    h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sparsity = _compute_sparsity(h, dim)
    expected = 1.0 - 1.0 / (1.0 * math.sqrt(4))
    assert abs(sparsity - expected) < 1e-6
    assert 0 < sparsity < 1


def test_compute_sparsity_dense():
    dim = 2
    h = torch.tensor([[1.0, 1.0]])
    sparsity = _compute_sparsity(h, dim)
    expected = 1.0 - (2.0 / (math.sqrt(2) * math.sqrt(2)))
    assert abs(sparsity - expected) < 1e-6


def test_compute_velocity():
    hidden = (
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, 4) * 2,
        torch.ones(1, 1, 4) * 4,
    )
    velocities = _compute_velocity(hidden)
    assert velocities[0] == 0.0
    assert velocities[1] == 4.0
    assert velocities[2] == 4.0


def test_extract_metrics_matched():
    hidden_a = (
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    hidden_b = (
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    payload = extract_metrics(hidden_a, hidden_b, is_matched=True, hidden_dim_a=2, hidden_dim_b=2)

    assert "layer_0" in payload
    assert "layer_1" in payload

    assert payload["layer_0"]["cosine_sim"] == 1.0
    assert payload["layer_0"]["mse"] == 0.0
    assert payload["layer_0"]["model_a_l2"] > 0
    assert payload["layer_1"]["cosine_sim"] == 1.0


def test_extract_metrics_mismatched():
    hidden_a = (
        torch.tensor([[[1.0, 2.0, 3.0]]]),
    )
    hidden_b = (
        torch.tensor([[[4.0, 5.0]]]),
    )
    payload = extract_metrics(hidden_a, hidden_b, is_matched=False, hidden_dim_a=3, hidden_dim_b=2)

    assert "layer_0" in payload
    assert "cosine_sim" not in payload["layer_0"]
    assert "mse" not in payload["layer_0"]
    assert "model_a_l2" in payload["layer_0"]
    assert "model_b_l2" in payload["layer_0"]
    assert "model_a_sparsity" in payload["layer_0"]
    assert "model_b_sparsity" in payload["layer_0"]
    assert "model_a_velocity" in payload["layer_0"]
    assert "model_b_velocity" in payload["layer_0"]
    assert "model_a_last_l2" in payload["layer_0"]
    assert "model_b_last_l2" in payload["layer_0"]


def test_extract_metrics_payload_structure():
    hidden_a = (
        torch.randn(1, 3, 4),
        torch.randn(1, 3, 4),
    )
    hidden_b = (
        torch.randn(1, 3, 4),
        torch.randn(1, 3, 4),
    )
    payload = extract_metrics(hidden_a, hidden_b, is_matched=True, hidden_dim_a=4, hidden_dim_b=4)

    for key in payload:
        if key.startswith("_"):
            continue
        entry = payload[key]
        assert "cosine_sim" in entry
        assert "mse" in entry
        assert "model_a_l2" in entry
        assert "model_b_l2" in entry
        assert "model_a_velocity" in entry
        assert "model_b_velocity" in entry
        assert "model_a_sparsity" in entry
        assert "model_b_sparsity" in entry

    assert "_per_token_cosine" in payload
    assert len(payload["_per_token_cosine"]) == 2
    assert len(payload["_per_token_cosine"][0]) == 3
    assert len(payload) == 3  # 2 layer keys + 1 meta key


class _DummyTokenizer:
    def decode(self, ids):
        return f"tok_{ids}"


def test_extract_metrics_logit_lens():
    hidden = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    lm_head = torch.nn.Linear(4, 10)
    tokenizer = _DummyTokenizer()
    payload = extract_metrics(
        hidden, hidden, is_matched=True,
        hidden_dim_a=4, hidden_dim_b=4,
        lm_head_a=lm_head, lm_head_b=lm_head,
        tokenizer_a=tokenizer, tokenizer_b=tokenizer,
    )
    assert "_logit_lens_a" in payload
    assert "_logit_lens_b" in payload
    assert len(payload["_logit_lens_a"]) == 2
    assert len(payload["_logit_lens_a"][0]) == 5
    assert "token" in payload["_logit_lens_a"][0][0]
    assert "prob" in payload["_logit_lens_a"][0][0]


def test_extract_metrics_logit_lens_skipped():
    hidden = (torch.randn(1, 2, 4),)
    payload = extract_metrics(
        hidden, hidden, is_matched=True,
        hidden_dim_a=4, hidden_dim_b=4,
    )
    assert "_logit_lens_a" not in payload
    assert "_logit_lens_b" not in payload
