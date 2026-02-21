from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class AttnCurveResult:
    length: int
    curve: np.ndarray  # shape: [length]
    sink_positions: List[int]


def _choose_layer_indices(num_hidden_layers: int, num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    if num_layers >= num_hidden_layers:
        return list(range(num_hidden_layers))
    # evenly spaced indices
    return [int(round(x)) for x in np.linspace(0, num_hidden_layers - 1, num_layers).tolist()]


@torch.inference_mode()
def attention_to_position_curve_llama(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: Sequence[int],
    head_indices: Sequence[int],
) -> torch.Tensor:
    """Compute attention-to-position curve for LLaMA-like models.

    Strategy:
    1) Run model once with output_hidden_states=True (no attentions) to get per-layer inputs.
    2) Recompute attention weights only for selected layers/heads using the layer's self_attn module.

    Returns a vector of shape [T] on CPU.
    """

    model.eval()
    device = input_ids.device

    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    hidden_states = out.hidden_states  # tuple length: layers+1

    # Average attention mass assigned to each key position.
    T = int(input_ids.shape[1])
    acc = torch.zeros((T,), dtype=torch.float64, device="cpu")
    denom = 0

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    for li in layer_indices:
        layer = model.model.layers[int(li)]
        attn = layer.self_attn

        x = hidden_states[int(li)].to(device)

        # Build position_ids
        position_ids = torch.arange(T, device=device).unsqueeze(0)

        cfg = model.config
        num_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "num_heads", 0) or getattr(cfg, "n_head", 0))
        if num_heads <= 0:
            raise ValueError("Cannot determine num_attention_heads from model config")

        head_dim = int(getattr(attn, "head_dim", 0) or (int(getattr(cfg, "hidden_size")) // num_heads))
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
        num_kv_groups = int(getattr(attn, "num_key_value_groups", 1) or (num_heads // max(1, num_kv_heads)))

        # Projections
        q = attn.q_proj(x)
        k = attn.k_proj(x)

        bsz = q.shape[0]

        q = q.view(bsz, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]

        k = k.view(bsz, T, num_kv_heads, head_dim).transpose(1, 2)  # [B, H_kv, T, D]

        # Rotary embeddings (LLaMA/Mistral style)
        if hasattr(model.model, "rotary_emb"):
            cos, sin = model.model.rotary_emb(k, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Expand kv heads to full heads if needed
        if num_kv_heads != num_heads:
            k = repeat_kv(k, num_kv_groups)

        # Select heads
        sel = torch.tensor(list(head_indices), device=device, dtype=torch.long)
        qh = q.index_select(dim=1, index=sel)
        kh = k.index_select(dim=1, index=sel)

        # Attention scores [B, h, T, T]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)

        # Causal mask
        causal = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))

        attn_probs = torch.softmax(scores.float(), dim=-1)  # float32 softmax

        # Mean over queries -> attention mass per key position
        mass_per_k = attn_probs.mean(dim=-2)  # [B, h, T]
        mass_per_k = mass_per_k.mean(dim=1).mean(dim=0)  # [T]

        acc += mass_per_k.detach().to("cpu", dtype=torch.float64)
        denom += 1

        # free
        del q, k, qh, kh, scores, attn_probs, mass_per_k

    if denom == 0:
        return acc.to(dtype=torch.float32)

    return (acc / float(denom)).to(dtype=torch.float32)


def find_sink_positions(curve: np.ndarray, topk: int = 5) -> List[int]:
    topk = min(topk, int(curve.shape[0]))
    idx = np.argsort(-curve)[:topk]
    return [int(i) for i in idx.tolist()]
