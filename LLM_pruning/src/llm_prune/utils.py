from __future__ import annotations

import dataclasses
import os
import random
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compile_optional_regex(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if not pattern:
        return None
    return re.compile(pattern)


@dataclasses.dataclass
class LinearLike:
    name: str
    module: torch.nn.Module
    weight: torch.nn.Parameter
    weight_layout: str  # "linear" (out,in) or "conv1d" (in,out)


def iter_prunable_linears(
    model: torch.nn.Module,
    include: Optional[re.Pattern[str]] = None,
    exclude: Optional[re.Pattern[str]] = None,
) -> Iterable[LinearLike]:
    """Yield linear-like modules with 2D weights.

    Supports torch.nn.Linear and HF Conv1D-like layers (any module with 2D .weight).
    Excludes embeddings and lm_head via regex by default from caller.
    """

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight")
        if not isinstance(weight, torch.nn.Parameter):
            continue
        if weight.dim() != 2:
            continue
        if include is not None and include.search(name) is None:
            continue
        if exclude is not None and exclude.search(name) is not None:
            continue

        # Heuristic: Linear is (out,in). Many HF Conv1D are (in,out).
        layout = "linear"
        if module.__class__.__name__ == "Conv1D":
            layout = "conv1d"
        else:
            # Fallback heuristic: if module has in_features/out_features, treat like Linear
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                layout = "linear"
            # If shape looks like (in,out) and module lacks Linear attrs, treat as conv1d.
            if not hasattr(module, "in_features") and weight.shape[0] < weight.shape[1]:
                # Not perfect, but avoids transposing wrong in GPT2-like layers.
                layout = "conv1d"

        yield LinearLike(name=name, module=module, weight=weight, weight_layout=layout)


def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def maybe_reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_cuda_peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))
