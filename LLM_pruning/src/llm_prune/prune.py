from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import math
import random

import torch

from .activation_stats import ActivationRMS
from .utils import LinearLike, compile_optional_regex, iter_prunable_linears


@dataclass
class PruneReport:
    method: str
    target_sparsity: float
    total_weights: int
    nonzero_weights_before: int
    nonzero_weights_after: int

    @property
    def achieved_sparsity(self) -> float:
        if self.total_weights <= 0:
            return 0.0
        return 1.0 - (self.nonzero_weights_after / self.total_weights)


def _weight_score_magnitude(layer: LinearLike) -> torch.Tensor:
    return layer.weight.detach().abs().float().cpu()


def _weight_score_wanda(layer: LinearLike, act_rms: torch.Tensor) -> torch.Tensor:
    w = layer.weight.detach().abs().float().cpu()
    a = act_rms.detach().float().cpu()

    if layer.weight_layout == "linear":
        # W: (out, in), a: (in,)
        if w.shape[1] != a.numel():
            raise ValueError(f"Activation dim mismatch for {layer.name}: W {tuple(w.shape)} vs a {tuple(a.shape)}")
        return w * a.view(1, -1)

    # conv1d-like: W: (in, out), a: (in,)
    if w.shape[0] != a.numel():
        raise ValueError(f"Activation dim mismatch for {layer.name}: W {tuple(w.shape)} vs a {tuple(a.shape)}")
    return w * a.view(-1, 1)


def _apply_unstructured_prune_by_threshold(weight: torch.nn.Parameter, threshold: float) -> Tuple[int, int]:
    """Zero out entries with |w| <= threshold. Returns (nonzero_before, nonzero_after)."""
    with torch.no_grad():
        w = weight.data
        before = int((w != 0).sum().item())
        mask = w.abs() > threshold
        w.mul_(mask)
        after = int((w != 0).sum().item())
    return before, after


def _prune_layer_by_sparsity_magnitude(layer: LinearLike, target_sparsity: float) -> Tuple[int, int, int]:
    """Prune a single layer by magnitude to a target sparsity (per-layer).

    Returns (total, nonzero_before, nonzero_after) counted on the layer weight.
    """
    with torch.no_grad():
        w = layer.weight.data
        total = int(w.numel())
        nonzero_before = int((w != 0).sum().item())
        k = int(total * target_sparsity)
        if k <= 0:
            return total, nonzero_before, nonzero_before

        # kthvalue expects 1 <= k <= n
        k = min(max(1, k), total)
        score = w.abs().to(dtype=torch.float32)
        threshold = float(torch.kthvalue(score.flatten(), k).values.item())
        w.mul_(score > threshold)
        nonzero_after = int((w != 0).sum().item())
        return total, nonzero_before, nonzero_after


def _prune_layer_by_sparsity_wanda(layer: LinearLike, act_rms: torch.Tensor, target_sparsity: float) -> Tuple[int, int, int]:
    """Prune a single layer by WANDA score to a target sparsity (per-layer)."""
    with torch.no_grad():
        w = layer.weight.data
        total = int(w.numel())
        nonzero_before = int((w != 0).sum().item())
        k = int(total * target_sparsity)
        if k <= 0:
            return total, nonzero_before, nonzero_before

        k = min(max(1, k), total)
        a = act_rms.to(w.device, dtype=torch.float32)
        absw = w.abs().to(dtype=torch.float32)
        if layer.weight_layout == "linear":
            score = absw * a.view(1, -1)
        else:
            score = absw * a.view(-1, 1)
        threshold = float(torch.kthvalue(score.flatten(), k).values.item())
        w.mul_(score > threshold)
        nonzero_after = int((w != 0).sum().item())
        return total, nonzero_before, nonzero_after


def prune_magnitude(
    model: torch.nn.Module,
    target_sparsity: float,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> PruneReport:
    include = compile_optional_regex(include_regex)
    exclude = compile_optional_regex(exclude_regex)

    layers = list(iter_prunable_linears(model, include=include, exclude=exclude))
    if not layers:
        return PruneReport("magnitude", target_sparsity, 0, 0, 0)

    # Per-layer pruning avoids materializing a massive global score vector (saves RAM).
    total_weights = 0
    before_total = 0
    after_total = 0
    for layer in layers:
        t, b, a = _prune_layer_by_sparsity_magnitude(layer, target_sparsity)
        total_weights += t
        before_total += b
        after_total += a

    return PruneReport(
        method="magnitude",
        target_sparsity=target_sparsity,
        total_weights=total_weights,
        nonzero_weights_before=before_total,
        nonzero_weights_after=after_total,
    )


def prune_wanda(
    model: torch.nn.Module,
    act: ActivationRMS,
    target_sparsity: float,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> PruneReport:
    include = compile_optional_regex(include_regex)
    exclude = compile_optional_regex(exclude_regex)

    layers = list(iter_prunable_linears(model, include=include, exclude=exclude))
    if not layers:
        return PruneReport("wanda", target_sparsity, 0, 0, 0)

    # Per-layer pruning avoids building a global score vector (saves RAM).
    total_weights = 0
    before_total = 0
    after_total = 0

    for layer in layers:
        if layer.name not in act.rms:
            continue
        t, b, a = _prune_layer_by_sparsity_wanda(layer, act.rms[layer.name], target_sparsity)
        total_weights += t
        before_total += b
        after_total += a

    return PruneReport(
        method="wanda",
        target_sparsity=target_sparsity,
        total_weights=total_weights,
        nonzero_weights_before=before_total,
        nonzero_weights_after=after_total,
    )


@torch.inference_mode()
def wanda_score_summary(
    model: torch.nn.Module,
    act: ActivationRMS,
    *,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    sample_size: int = 200_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute a whole-model summary of WANDA scores.

    WANDA score per-weight is defined as: score = |W| * act_rms (broadcast over input features).
    This function aggregates streaming stats (mean/std/min/max) over all scored weights,
    and also estimates quantiles from a uniform random sample of score elements.

    Notes:
    - This does not prune; it only measures the score distribution.
    - Sampling is approximate but cheap; the mean/std are exact given the dense score tensor
      computed per layer.
    """

    include = compile_optional_regex(include_regex)
    exclude = compile_optional_regex(exclude_regex)

    layers = [layer for layer in iter_prunable_linears(model, include=include, exclude=exclude) if layer.name in act.rms]
    total_weights = int(sum(int(layer.weight.numel()) for layer in layers))
    if total_weights <= 0:
        return {
            "total_weights": 0,
            "layers": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "sample_size": 0,
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
        }

    rng = random.Random(int(seed))
    # Streaming stats
    sum_score = 0.0
    sumsq_score = 0.0
    min_score = float("inf")
    max_score = float("-inf")

    # Sampled stats (uniform by allocating sample per layer proportional to numel)
    remaining = int(max(0, sample_size))
    samples_cpu: list[torch.Tensor] = []

    for layer in layers:
        w = layer.weight.detach()
        a = act.rms[layer.name].to(w.device, dtype=torch.float32)
        absw = w.abs().to(dtype=torch.float32)
        if layer.weight_layout == "linear":
            score = absw * a.view(1, -1)
        else:
            score = absw * a.view(-1, 1)

        # Exact aggregates
        s = float(score.sum(dtype=torch.float64).item())
        ss = float((score * score).sum(dtype=torch.float64).item())
        sum_score += s
        sumsq_score += ss
        mn = float(score.min().item())
        mx = float(score.max().item())
        if mn < min_score:
            min_score = mn
        if mx > max_score:
            max_score = mx

        # Sampling
        if remaining > 0:
            layer_n = int(score.numel())
            # proportional allocation with a bit of randomness to avoid systematic rounding bias
            want = int(round((sample_size * (layer_n / float(total_weights))) + rng.random() - 0.5))
            want = max(0, min(remaining, want))
            if want > 0:
                flat = score.flatten()
                # Use torch.randint for speed; seed via Python RNG.
                gen_seed = int(rng.randrange(0, 2**31 - 1))
                g = torch.Generator(device=flat.device)
                g.manual_seed(gen_seed)
                idx = torch.randint(0, layer_n, (want,), device=flat.device, generator=g)
                samp = flat.index_select(0, idx).detach().to("cpu", dtype=torch.float32)
                samples_cpu.append(samp)
                remaining -= want

        # free
        del score, absw, a

    n = float(total_weights)
    mean = sum_score / n
    # Var = E[x^2] - mean^2
    ex2 = sumsq_score / n
    var = max(0.0, ex2 - mean * mean)
    std = math.sqrt(var)

    if samples_cpu:
        sample = torch.cat(samples_cpu, dim=0)
        # Guard against empty
        if int(sample.numel()) > 0:
            p50 = float(torch.quantile(sample, 0.50).item())
            p90 = float(torch.quantile(sample, 0.90).item())
            p99 = float(torch.quantile(sample, 0.99).item())
            used = int(sample.numel())
        else:
            p50 = p90 = p99 = float("nan")
            used = 0
    else:
        p50 = p90 = p99 = float("nan")
        used = 0

    return {
        "total_weights": int(total_weights),
        "layers": int(len(layers)),
        "mean": float(mean),
        "std": float(std),
        "min": float(min_score if min_score != float("inf") else float("nan")),
        "max": float(max_score if max_score != float("-inf") else float("nan")),
        "sample_size": int(used),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
    }
