import math
import torch
import torch.nn.functional as F


def _compute_sparsity(h: torch.Tensor, dim: int) -> float:
    l1 = h.norm(p=1, dim=-1)
    l2 = h.norm(p=2, dim=-1)
    return (1.0 - l1 / (l2 * math.sqrt(dim))).mean().item()


def _compute_l2(h: torch.Tensor) -> float:
    return h.norm(p=2, dim=-1).mean().item()


def _compute_velocity(
    hidden_states: tuple[torch.Tensor, ...],
) -> list[float]:
    n = len(hidden_states)
    velocities = [0.0] * n
    for i in range(1, n):
        velocities[i] = F.mse_loss(hidden_states[i], hidden_states[i - 1]).item()
    return velocities


def _compute_logit_lens(
    hidden_states: tuple[torch.Tensor, ...],
    lm_head: torch.nn.Module,
    tokenizer,
    top_k: int = 5,
) -> list[list[dict]]:
    result: list[list[dict]] = []
    for h in hidden_states:
        last_token = h[:, -1, :]
        logits = lm_head(last_token)
        probs = F.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, top_k, dim=-1)
        layer_tokens: list[dict] = []
        for idx, val in zip(indices.squeeze(0).tolist(), values.squeeze(0).tolist()):
            token_str = tokenizer.decode([idx], skip_special_tokens=False)
            layer_tokens.append({"token": token_str, "prob": round(val, 4)})
        result.append(layer_tokens)
    return result


def extract_metrics(
    hidden_a: tuple[torch.Tensor, ...],
    hidden_b: tuple[torch.Tensor, ...],
    is_matched: bool,
    hidden_dim_a: int,
    hidden_dim_b: int,
    lm_head_a: torch.nn.Module | None = None,
    lm_head_b: torch.nn.Module | None = None,
    tokenizer_a=None,
    tokenizer_b=None,
) -> dict[str, dict]:
    n_layers = len(hidden_a)
    velocity_a = _compute_velocity(hidden_a)
    velocity_b = _compute_velocity(hidden_b)

    payload: dict[str, dict] = {}

    if is_matched:
        per_token_cosine: list[list[float]] = []

    for i in range(n_layers):
        ha = hidden_a[i]
        hb = hidden_b[i]

        entry: dict = {
            "model_a_l2": _compute_l2(ha),
            "model_b_l2": _compute_l2(hb),
            "model_a_velocity": velocity_a[i],
            "model_b_velocity": velocity_b[i],
            "model_a_sparsity": _compute_sparsity(ha, hidden_dim_a),
            "model_b_sparsity": _compute_sparsity(hb, hidden_dim_b),
        }

        if is_matched:
            cos_per_token = F.cosine_similarity(ha, hb, dim=-1)
            entry["cosine_sim"] = cos_per_token.mean().item()
            entry["mse"] = F.mse_loss(ha, hb).item()
            per_token_cosine.append(cos_per_token.squeeze(0).tolist())
        else:
            ha_lt = ha[:, -1, :]
            hb_lt = hb[:, -1, :]
            entry["model_a_last_l2"] = _compute_l2(ha_lt)
            entry["model_b_last_l2"] = _compute_l2(hb_lt)

        payload[f"layer_{i}"] = entry

    if is_matched:
        payload["_per_token_cosine"] = per_token_cosine

    if lm_head_a is not None and tokenizer_a is not None:
        payload["_logit_lens_a"] = _compute_logit_lens(hidden_a, lm_head_a, tokenizer_a)
    if lm_head_b is not None and tokenizer_b is not None:
        payload["_logit_lens_b"] = _compute_logit_lens(hidden_b, lm_head_b, tokenizer_b)

    return payload
