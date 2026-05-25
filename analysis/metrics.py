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


def extract_metrics(
    hidden_a: tuple[torch.Tensor, ...],
    hidden_b: tuple[torch.Tensor, ...],
    is_matched: bool,
    hidden_dim_a: int,
    hidden_dim_b: int,
) -> dict[str, dict]:
    n_layers = len(hidden_a)
    velocity_a = _compute_velocity(hidden_a)
    velocity_b = _compute_velocity(hidden_b)

    payload: dict[str, dict] = {}

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
            entry["cosine_sim"] = (
                F.cosine_similarity(ha, hb, dim=-1).mean().item()
            )
            entry["mse"] = F.mse_loss(ha, hb).item()
        else:
            ha_lt = ha[:, -1, :]
            hb_lt = hb[:, -1, :]
            entry["model_a_last_l2"] = _compute_l2(ha_lt)
            entry["model_b_last_l2"] = _compute_l2(hb_lt)

        payload[f"layer_{i}"] = entry

    return payload
