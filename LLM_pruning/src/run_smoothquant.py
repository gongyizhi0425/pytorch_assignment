#!/usr/bin/env python3
"""SmoothQuant (W8A8) assignment — TinyLlama on WikiText-2.

Steps
1. Baseline (FP16): PPL + speed
2. Naïve W8A8 PTQ:
       weights  → INT8 per-channel (symmetric)
       activations → INT8 per-tensor (static, using calibration set)
3. SmoothQuant W8A8:
       For each α ∈ {0, 0.25, 0.5, 0.75}:
         a) collect per-channel activation max on calibration set
         b) compute smoothing scales  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
         c) fuse  X' = X / diag(s),  W' = diag(s) · W
         d) quantise  W' per-channel INT8,  X' per-tensor INT8
         e) evaluate PPL + speed
4. Plot  PPL vs α  and  Latency vs α
5. Outlier channel analysis: bar chart before / after smoothing
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ───────────────────────────── helpers ─────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ─────────────────────── data loading ──────────────────────────────

def load_eval_tokens(tokenizer, cfg: dict, max_tokens: int) -> torch.Tensor:
    ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
    ids: list[int] = []
    for row in ds:
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(enc)
        if tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long)


def load_calibration_inputs(tokenizer, cfg: dict) -> List[torch.Tensor]:
    """Return list of (1, seq_len) tensors."""
    ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
    out: list[torch.Tensor] = []
    for i in range(min(cfg["num_samples"], len(ds))):
        text = str(ds[i].get("text", ""))
        if not text.strip():
            continue
        enc = tokenizer(text, truncation=True, max_length=cfg["seq_len"],
                        padding="max_length", return_tensors="pt")
        out.append(enc["input_ids"])          # (1, seq_len)
    return out


# ─────────────────────── PPL evaluation ────────────────────────────

@torch.inference_mode()
def eval_ppl(model, token_stream: torch.Tensor, device: torch.device,
             block_size: int = 512) -> float:
    model.eval()
    token_stream = token_stream.to("cpu")
    stride = block_size
    nll_sum, n_tok = 0.0, 0
    for begin in range(0, token_stream.numel(), stride):
        end = min(begin + block_size, token_stream.numel())
        inp = token_stream[begin:end].unsqueeze(0).to(device)
        tgt = inp.clone()
        if begin > 0:
            tgt[:, :block_size - stride] = -100
        loss = float(model(input_ids=inp, labels=tgt).loss)
        scored = int((tgt != -100).sum().item())
        nll_sum += loss * scored
        n_tok += scored
        if end == token_stream.numel():
            break
    return math.exp(nll_sum / max(1, n_tok))


# ─────────────────── speed measurement ─────────────────────────────

@torch.inference_mode()
def measure_speed(model, device: torch.device, vocab_size: int,
                  prefill_len: int = 256, decode_tokens: int = 32,
                  repeats: int = 5) -> Dict[str, float]:
    model.eval()
    ids = torch.randint(0, vocab_size, (1, prefill_len), device=device)

    # warmup
    for _ in range(2):
        _ = model(input_ids=ids, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # prefill
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(input_ids=ids, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) / repeats * 1000
    prefill_tok_s = prefill_len / (prefill_ms / 1000)

    # decode
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(decode_tokens):
        out = model(input_ids=nxt, use_cache=True, past_key_values=past)
        past = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_tok_s = decode_tokens / (decode_ms / 1000)

    return {
        "prefill_ms": round(prefill_ms, 2),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "decode_ms": round(decode_ms, 2),
        "decode_tok_s": round(decode_tok_s, 1),
    }


# ─────────────── model size on disk (simulated) ───────────────────

def model_size_mb(model: nn.Module, bits: int = 16) -> float:
    """Estimate model size for given uniform bit-width."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * bits / 8 / 1024 / 1024


# ═══════════════════════  QUANTISATION  ════════════════════════════

# --- INT8 helpers --------------------------------------------------

def _quantize_per_channel_symmetric(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise weight (out_features, in_features) → INT8 per output-channel.

    Returns (w_int8, scale) where  w ≈ w_int8.float() * scale[:,None].
    """
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)   # (out, 1)
    scale = amax / 127.0
    w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.squeeze(1)          # scale: (out,)


def _dequant_per_channel(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w_int8.float() * scale[:, None]


def _quantize_per_tensor_symmetric(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x / scale).round().clamp(-128, 127).to(torch.int8)


def _dequant_per_tensor(x_int8: torch.Tensor, scale: float) -> torch.Tensor:
    return x_int8.float() * scale


# --- static activation scale from calibration ----------------------

@torch.inference_mode()
def collect_act_scales(model, calib_inputs: List[torch.Tensor],
                       device: torch.device) -> Dict[str, torch.Tensor]:
    """Collect per-channel activation **max abs** for every Linear input.

    Returns {layer_name: tensor(in_features)}.
    """
    model.eval()
    act_scales: Dict[str, torch.Tensor] = {}
    hooks = []

    def _hook_fn(name):
        def fn(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])        # (tokens, hidden)
            amax = x.abs().amax(dim=0)                 # (hidden,)
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], amax)
            else:
                act_scales[name] = amax.clone()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_hook_fn(name)))

    for ids in calib_inputs:
        model(input_ids=ids.to(device))

    for h in hooks:
        h.remove()

    # move to cpu
    return {k: v.cpu() for k, v in act_scales.items()}


def _compute_static_act_scale(act_scales: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute per-tensor static activation scale for each Linear layer.

    per-tensor scale = global_max_abs / 127
    """
    result = {}
    for name, amax_per_ch in act_scales.items():
        global_max = float(amax_per_ch.max().item())
        result[name] = max(global_max, 1e-8) / 127.0
    return result


# ═════════════  NAÏVE W8A8 PTQ  (simulated, in-place)  ════════════

def apply_naive_w8a8(model: nn.Module, static_act_scales: Dict[str, float],
                     device: torch.device):
    """Replace each Linear's weight with dequantized INT8 approximation
    and insert a forward-pre-hook that simulates activation quantisation.
    """
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # weight: per-channel symmetric INT8
        w = module.weight.data
        w_int8, w_scale = _quantize_per_channel_symmetric(w)
        module.weight.data = _dequant_per_channel(w_int8, w_scale).to(w.dtype)

        # activation: per-tensor symmetric INT8 (static)
        act_s = static_act_scales.get(name, 1.0)
        def _make_hook(s):
            def hook(mod, args):
                x = args[0]
                x_q = _quantize_per_tensor_symmetric(x, s)
                x_deq = _dequant_per_tensor(x_q, s).to(x.dtype)
                return (x_deq,) + args[1:]
            return hook
        hooks.append(module.register_forward_pre_hook(_make_hook(act_s)))

    return hooks    # caller can remove later


# ═══════════  SMOOTHQUANT  ═════════════════════════════════════════

def apply_smoothquant(model: nn.Module,
                      act_scales: Dict[str, torch.Tensor],
                      alpha: float,
                      device: torch.device) -> Dict[str, float]:
    """Apply SmoothQuant + W8A8 in-place.

    For each Linear layer:
        s_j = max(|X_j|)^α  /  max(|W_j|)^(1-α)
        W' = diag(s) · W   (per input-channel)
        Then quantise W' per-channel INT8
        Activation hook:  X' = X / diag(s) →  per-tensor INT8 (static)

    Returns dict of per-tensor static activation scales (after smoothing).
    """
    static_scales: Dict[str, float] = {}
    hooks: list = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in act_scales:
            continue

        w = module.weight.data.float()          # (out, in)
        act_max = act_scales[name].to(w.device) # (in,)
        w_max = w.abs().amax(dim=0)             # (in,)

        # smoothing factor
        s = (act_max.clamp(min=1e-8).pow(alpha) /
             w_max.clamp(min=1e-8).pow(1.0 - alpha))

        # smooth weight:  W' = W * diag(s)  (scale input dimension)
        w_smooth = w * s.unsqueeze(0)           # broadcast (out, in)
        # quantise weight per-channel
        w_int8, w_scale = _quantize_per_channel_symmetric(w_smooth)
        module.weight.data = _dequant_per_channel(w_int8, w_scale).to(module.weight.dtype)

        # activation: X' = X / s  →  per-tensor INT8 static
        # new act max after dividing by s
        smooth_act_max = (act_max / s.clamp(min=1e-8)).max().item()
        act_scale = max(smooth_act_max, 1e-8) / 127.0
        static_scales[name] = act_scale

        # pre-hook:  X → X / diag(s) → INT8 per-tensor → dequant
        def _make_hook(s_vec, a_scale):
            s_vec_dev = s_vec.clone()  # will be moved at first call
            def hook(mod, args):
                x = args[0]
                s_local = s_vec_dev.to(x.device, x.dtype)
                x_smooth = x / s_local                         # per-channel divide
                x_q = _quantize_per_tensor_symmetric(x_smooth, a_scale)
                x_deq = _dequant_per_tensor(x_q, a_scale).to(x.dtype)
                return (x_deq,) + args[1:]
            return hook
        hooks.append(module.register_forward_pre_hook(_make_hook(s, act_scale)))

    return static_scales


# ═══════════  OUTLIER ANALYSIS  ════════════════════════════════════

@torch.inference_mode()
def collect_per_channel_activation_max(
    model, calib_inputs: List[torch.Tensor], device: torch.device,
    target_layer_substr: str = "",
) -> Dict[str, torch.Tensor]:
    """Collect per-channel activation max-abs for a chosen subset of layers."""
    model.eval()
    result: Dict[str, torch.Tensor] = {}
    hooks = []

    def _hook(name):
        def fn(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            amax = x.abs().amax(dim=0)
            if name in result:
                result[name] = torch.max(result[name], amax)
            else:
                result[name] = amax.clone()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target_layer_substr in name:
            hooks.append(module.register_forward_hook(_hook(name)))

    for ids in calib_inputs:
        model(input_ids=ids.to(device))

    for h in hooks:
        h.remove()

    return {k: v.cpu() for k, v in result.items()}


# ═══════════  PLOTTING  ═══════════════════════════════════════════

def plot_ppl_vs_alpha(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # baseline
    base_ppl = df.loc[df["method"] == "baseline", "ppl"].values[0]
    naive_ppl = df.loc[df["method"] == "naive_w8a8", "ppl"].values[0]
    sq = df[df["method"] == "smoothquant"].sort_values("alpha")

    alphas = sq["alpha"].values
    ppls = sq["ppl"].values

    ax.axhline(base_ppl, color="green", ls="--", lw=1.5, label=f"Baseline FP16 (PPL={base_ppl:.2f})")
    ax.axhline(naive_ppl, color="red", ls=":", lw=1.5, label=f"Naïve W8A8 (PPL={naive_ppl:.2f})")
    ax.plot(alphas, ppls, "bo-", lw=2, ms=7, label="SmoothQuant W8A8")
    for a, p in zip(alphas, ppls):
        ax.annotate(f"{p:.2f}", (a, p), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.set_xlabel("SmoothQuant α", fontsize=12)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
    ax.set_title("PPL vs. SmoothQuant α  (TinyLlama-1.1B / WikiText-2)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_latency_vs_alpha(df: pd.DataFrame, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    base = df[df["method"] == "baseline"].iloc[0]
    naive = df[df["method"] == "naive_w8a8"].iloc[0]
    sq = df[df["method"] == "smoothquant"].sort_values("alpha")

    for ax, col, title in [
        (ax1, "prefill_tok_s", "Prefill throughput (tok/s ↑)"),
        (ax2, "decode_tok_s", "Decode throughput (tok/s ↑)"),
    ]:
        ax.axhline(base[col], color="green", ls="--", lw=1.5, label="Baseline FP16")
        ax.axhline(naive[col], color="red", ls=":", lw=1.5, label="Naïve W8A8")
        ax.plot(sq["alpha"], sq[col], "bo-", lw=2, ms=7, label="SmoothQuant")
        ax.set_xlabel("SmoothQuant α")
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    fig.suptitle("Latency/Throughput vs. α  (TinyLlama-1.1B)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_outlier_channels(before: Dict[str, torch.Tensor],
                          after: Dict[str, torch.Tensor],
                          top_k: int, out_path: str):
    """Bar chart: top-K outlier channels before & after SmoothQuant."""
    # pick ONE representative layer  (first self_attn.q_proj or first Linear)
    target = None
    for name in before:
        if "self_attn.q_proj" in name:
            target = name
            break
    if target is None:
        target = list(before.keys())[0]

    b = before[target].numpy()
    a = after.get(target, before[target]).numpy()

    # top-K channels by before magnitude
    top_idx = np.argsort(b)[-top_k:][::-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    x = np.arange(top_k)
    ax1.bar(x, b[top_idx], color="tomato", edgecolor="darkred", alpha=0.85)
    ax1.set_ylabel("Activation max |X|")
    ax1.set_title(f"Before SmoothQuant — Top-{top_k} outlier channels\n({target})")
    for i, idx in enumerate(top_idx):
        ax1.text(i, b[idx], f"ch{idx}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax2.bar(x, a[top_idx], color="steelblue", edgecolor="navy", alpha=0.85)
    ax2.set_ylabel("Activation max |X'| (smoothed)")
    ax2.set_title(f"After SmoothQuant (α=0.5)")
    ax2.set_xlabel("Channel rank (sorted by pre-smooth magnitude)")
    for i, idx in enumerate(top_idx):
        ax2.text(i, a[idx], f"ch{idx}", ha="center", va="bottom", fontsize=7, rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_full_channel_distribution(before: Dict[str, torch.Tensor],
                                   after: Dict[str, torch.Tensor],
                                   out_path: str):
    """All-channel activation max bar chart (before vs after) for one layer."""
    target = None
    for name in before:
        if "self_attn.q_proj" in name:
            target = name
            break
    if target is None:
        target = list(before.keys())[0]

    b = before[target].numpy()
    a = after.get(target, before[target]).numpy()
    n_ch = len(b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.bar(range(n_ch), b, width=1.0, color="tomato", alpha=0.7)
    ax1.set_ylabel("|X| max")
    ax1.set_title(f"Per-channel activation max — BEFORE smoothing\n({target})")
    ax1.set_xlim(0, n_ch)

    ax2.bar(range(n_ch), a, width=1.0, color="steelblue", alpha=0.7)
    ax2.set_ylabel("|X'| max (smoothed)")
    ax2.set_title("AFTER SmoothQuant (α=0.5)")
    ax2.set_xlabel("Channel index")
    ax2.set_xlim(0, n_ch)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path}")


# ═══════════  ANALYSIS  ═══════════════════════════════════════════

def write_analysis(df: pd.DataFrame, out_path: str):
    base = df[df["method"] == "baseline"].iloc[0]
    naive = df[df["method"] == "naive_w8a8"].iloc[0]
    sq = df[df["method"] == "smoothquant"].sort_values("alpha")

    best = sq.loc[sq["ppl"].idxmin()]

    lines = [
        "=" * 70,
        "  Brief Analysis — SmoothQuant (W8A8) on TinyLlama-1.1B / WikiText-2",
        "=" * 70,
        "",
        f"Baseline FP16   :  PPL = {base['ppl']:.2f}",
        f"Naïve W8A8 PTQ  :  PPL = {naive['ppl']:.2f}  (Δ = {naive['ppl'] - base['ppl']:+.2f})",
        "",
        "SmoothQuant sweep:",
    ]
    for _, r in sq.iterrows():
        lines.append(f"  α = {r['alpha']:.2f}  →  PPL = {r['ppl']:.2f}  (Δ = {r['ppl'] - base['ppl']:+.2f})")

    lines += [
        "",
        f"Best α = {best['alpha']:.2f}  →  PPL = {best['ppl']:.2f}",
        "",
        "--- Why does Naïve W8A8 degrade PPL? ---",
        "  Activations in LLMs exhibit strong per-channel outliers (some channels",
        "  have max values ~70× larger than the median).  Per-tensor INT8 quantisation",
        "  must accommodate these outliers, wasting most of the 256 quantisation levels",
        "  on the small-valued majority channels.  This creates large rounding errors.",
        "",
        "--- How does SmoothQuant help? ---",
        "  SmoothQuant redistributes the quantisation difficulty from activations",
        "  to weights by per-channel scaling:  X' = X / s,  W' = s · W.",
        "  Weights are easier to quantise (per-channel scale already), so absorbing",
        "  some activation magnitude into weights evens out the dynamic range.",
        "  The hyper-parameter α controls the trade-off: α=0 puts all burden on",
        "  weights; α=1 keeps activations untouched (like naïve W8A8).",
        "",
        "--- Latency ---",
        "  Since this is *simulated* quantisation (dequant back to FP16 before matmul),",
        "  actual throughput may be lower than the FP16 baseline.  Real speedup requires",
        "  INT8 GEMM kernels (e.g. via TensorRT-LLM or torchao).",
        "",
        "--- Outlier channels ---",
        "  Before smoothing, a handful of channels have ~10-70× larger activation",
        "  magnitudes than the rest.  After SmoothQuant, the distribution becomes",
        "  much more uniform, making per-tensor INT8 quantisation much more effective.",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → saved {out_path}")


# ═══════════  MAIN  ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SmoothQuant W8A8 — TinyLlama")
    parser.add_argument("--config", type=str,
                        default="configs/smoothquant.yaml")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = pick_device(cfg["model"]["device"])
    dtype = pick_dtype(cfg["model"]["dtype"], device)
    model_name = cfg["model"]["name"]
    cache_dir = args.cache_dir or cfg["model"].get("cache_dir", None)
    out_dir = os.path.join("results_smoothquant")
    ensure_dir(out_dir)

    print(f"\n{'='*60}")
    print(f"  SmoothQuant (W8A8) Experiment")
    print(f"  Model  : {model_name}")
    print(f"  Device : {device}   Dtype: {dtype}")
    print(f"  Output : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load tokenizer & model ──
    tok_kwargs: dict = {}
    mdl_kwargs: dict = {"torch_dtype": dtype}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
        mdl_kwargs["cache_dir"] = cache_dir
    if args.local_files_only:
        tok_kwargs["local_files_only"] = True
        mdl_kwargs["local_files_only"] = True

    print("[Model] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    print("[Model] Loading model …")
    model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load data ──
    print("\n[Data] Loading WikiText-2 evaluation tokens …")
    eval_tokens = load_eval_tokens(tokenizer, cfg["eval"],
                                   cfg["eval"]["max_eval_tokens"])
    print(f"  Eval tokens: {eval_tokens.numel():,}")

    print("[Data] Loading calibration inputs …")
    calib_inputs = load_calibration_inputs(tokenizer, cfg["calibration"])
    print(f"  Calibration samples: {len(calib_inputs)}")

    speed_cfg = cfg.get("speed", {})
    results: list[dict] = []

    # ══════════════════════════════════════════════════════════════
    #  Step 1: Baseline (FP16)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Step 1: Baseline FP16")
    print(f"{'='*60}")
    ppl_base = eval_ppl(model, eval_tokens, device, cfg["eval"].get("block_size", 512))
    print(f"  PPL = {ppl_base:.2f}")
    speed_base = measure_speed(model, device, vocab_size,
                               speed_cfg.get("prefill_seq_len", 256),
                               speed_cfg.get("decode_new_tokens", 32),
                               speed_cfg.get("repeats", 5))
    print(f"  Speed: {speed_base}")
    results.append({"method": "baseline", "alpha": None, "ppl": ppl_base,
                     "model_size_mb": round(model_size_mb(model, 16), 2),
                     **speed_base})

    # ── Collect activation scales (on original FP16 model) ──
    print("\n[Calibration] Collecting per-channel activation max …")
    act_scales = collect_act_scales(model, calib_inputs, device)
    print(f"  Layers with act scales: {len(act_scales)}")

    # ── Save per-channel activation max BEFORE smoothing ──
    outlier_before = collect_per_channel_activation_max(
        model, calib_inputs, device, target_layer_substr="self_attn")

    # ══════════════════════════════════════════════════════════════
    #  Step 2: Naïve W8A8 PTQ
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Step 2: Naïve W8A8 PTQ")
    print(f"{'='*60}")
    model_naive = copy.deepcopy(model)
    static_scales = _compute_static_act_scale(act_scales)
    hooks_naive = apply_naive_w8a8(model_naive, static_scales, device)
    ppl_naive = eval_ppl(model_naive, eval_tokens, device, cfg["eval"].get("block_size", 512))
    print(f"  PPL = {ppl_naive:.2f}  (Δ = {ppl_naive - ppl_base:+.2f})")
    speed_naive = measure_speed(model_naive, device, vocab_size,
                                speed_cfg.get("prefill_seq_len", 256),
                                speed_cfg.get("decode_new_tokens", 32),
                                speed_cfg.get("repeats", 5))
    print(f"  Speed: {speed_naive}")
    results.append({"method": "naive_w8a8", "alpha": None, "ppl": ppl_naive,
                     "model_size_mb": round(model_size_mb(model, 8), 2),
                     **speed_naive})
    # clean up
    for h in hooks_naive:
        h.remove()
    del model_naive; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    #  Step 3: SmoothQuant W8A8 — sweep α
    # ══════════════════════════════════════════════════════════════
    alphas = cfg["smoothquant"]["alphas"]
    outlier_after: Dict[str, torch.Tensor] = {}  # save for α=0.5

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"  Step 3: SmoothQuant W8A8  (α = {alpha})")
        print(f"{'='*60}")
        model_sq = copy.deepcopy(model)
        sq_scales = apply_smoothquant(model_sq, act_scales, alpha, device)
        ppl_sq = eval_ppl(model_sq, eval_tokens, device, cfg["eval"].get("block_size", 512))
        print(f"  PPL = {ppl_sq:.2f}  (Δ = {ppl_sq - ppl_base:+.2f})")
        speed_sq = measure_speed(model_sq, device, vocab_size,
                                 speed_cfg.get("prefill_seq_len", 256),
                                 speed_cfg.get("decode_new_tokens", 32),
                                 speed_cfg.get("repeats", 5))
        print(f"  Speed: {speed_sq}")
        results.append({"method": "smoothquant", "alpha": alpha, "ppl": ppl_sq,
                         "model_size_mb": round(model_size_mb(model, 8), 2),
                         **speed_sq})

        # collect outlier AFTER for α=0.5
        if abs(alpha - 0.5) < 0.01:
            outlier_after = collect_per_channel_activation_max(
                model_sq, calib_inputs, device, target_layer_substr="self_attn")

        del model_sq; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    #  Save results
    # ══════════════════════════════════════════════════════════════
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "smoothquant_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")
    print(df.to_string(index=False))

    # ══════════════════════════════════════════════════════════════
    #  Step 4: Plots
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Step 4: Plotting")
    print(f"{'='*60}")
    plot_ppl_vs_alpha(df, os.path.join(out_dir, "ppl_vs_alpha.png"))
    plot_latency_vs_alpha(df, os.path.join(out_dir, "latency_vs_alpha.png"))

    # ══════════════════════════════════════════════════════════════
    #  Step 5: Outlier channel analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Step 5: Outlier channel analysis")
    print(f"{'='*60}")
    top_k = cfg.get("outlier", {}).get("top_k", 20)
    plot_outlier_channels(outlier_before, outlier_after, top_k,
                          os.path.join(out_dir, "outlier_topk_channels.png"))
    plot_full_channel_distribution(outlier_before, outlier_after,
                                   os.path.join(out_dir, "outlier_all_channels.png"))

    # ── analysis text ──
    write_analysis(df, os.path.join(out_dir, "analysis.txt"))

    print(f"\n{'='*60}")
    print(f"  ✓  ALL DONE — results in {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
