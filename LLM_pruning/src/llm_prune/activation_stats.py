from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .utils import LinearLike, iter_prunable_linears


@dataclass
class ActivationRMS:
    # per-module input feature RMS (sqrt(mean(x^2)))
    rms: Dict[str, torch.Tensor]


@torch.inference_mode()
def collect_activation_rms(
    model: torch.nn.Module,
    device: torch.device,
    calibration_iter,
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> ActivationRMS:
    """Collect per-layer input RMS for linear-like layers.

    Stores RMS per input feature dimension for each prunable module.
    """

    include = None
    exclude = None
    import re

    if include_regex:
        include = re.compile(include_regex)
    if exclude_regex:
        exclude = re.compile(exclude_regex)

    model.eval()

    sum_sq: Dict[str, torch.Tensor] = {}
    count: Dict[str, int] = {}

    handles = []

    name_to_linear: Dict[str, LinearLike] = {}
    for layer in iter_prunable_linears(model, include=include, exclude=exclude):
        name_to_linear[layer.name] = layer

        def make_hook(layer_name: str, layout: str):
            def hook(module, inputs):
                x = inputs[0]
                if not torch.is_tensor(x):
                    return
                # Expect x shape: [B, T, in]
                if x.dim() < 2:
                    return
                x = x.detach()
                if x.dim() == 2:
                    # [B, in]
                    feat_dim = -1
                    reduce_dims = (0,)
                else:
                    # [B, T, in]
                    feat_dim = -1
                    reduce_dims = tuple(range(0, x.dim() - 1))

                # convert to fp32 for stable stats
                x = x.float()
                v = (x * x).sum(dim=reduce_dims)
                n = 1
                for d in reduce_dims:
                    n *= x.shape[d]

                if layer_name not in sum_sq:
                    sum_sq[layer_name] = v.cpu()
                    count[layer_name] = n
                else:
                    sum_sq[layer_name] += v.cpu()
                    count[layer_name] += n

            return hook

        handles.append(layer.module.register_forward_pre_hook(make_hook(layer.name, layer.weight_layout)))

    for input_ids, attention_mask in calibration_iter:
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in handles:
        h.remove()

    rms: Dict[str, torch.Tensor] = {}
    for name, ss in sum_sq.items():
        denom = max(1, count[name])
        mean_sq = ss / float(denom)
        rms[name] = torch.sqrt(mean_sq + 1e-12)

    return ActivationRMS(rms=rms)


def activation_outlier_summary(act: ActivationRMS) -> Dict[str, float]:
    """Return simple outlier metrics aggregated over layers."""
    max_over_median = []
    frac_top1 = []

    for _, v in act.rms.items():
        v = v.flatten().float()
        if v.numel() == 0:
            continue
        med = torch.median(v)
        mx = torch.max(v)
        max_over_median.append(float((mx / (med + 1e-12)).item()))

        s = torch.sum(v)
        frac_top1.append(float((mx / (s + 1e-12)).item()))

    if not max_over_median:
        return {"layers": 0.0, "max_over_median_mean": 0.0, "frac_top1_mean": 0.0}

    return {
        "layers": float(len(max_over_median)),
        "max_over_median_mean": float(sum(max_over_median) / len(max_over_median)),
        "frac_top1_mean": float(sum(frac_top1) / len(frac_top1)),
    }
