#!/usr/bin/env python3
"""AWQ (W4A16) assignment — TinyLlama on WikiText-2.

Steps
1. Baseline (FP16): PPL + speed + memory
2. Naïve 4-bit weight-only PTQ (W4A16, group_size=128):
       groupwise symmetric INT4 weight quantisation, activations in FP16
3. AWQ with group-size sweep (32 / 64 / 128):
       AutoAWQ activation-aware scaling → 4-bit groupwise quantisation
4. Quantisation error analysis:
       per-output-channel MAE before / after AWQ scaling
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import sys
import time
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

# ───────────────────────── helpers ─────────────────────────

def set_seed(seed: int = 42):
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


# ───────────────────────── data ────────────────────────────

def load_eval_tokens(tokenizer, cfg) -> torch.Tensor:
    """Load WikiText-2 test split → single flat token tensor."""
    ds = load_dataset(cfg["eval"]["dataset"], cfg["eval"]["config"],
                      split=cfg["eval"]["split"])
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    max_tok = cfg["eval"].get("max_tokens", 60000)
    return tokens[:max_tok]


# ───────────────────────── PPL evaluation ──────────────────

@torch.no_grad()
def eval_ppl(model, tokenizer, eval_tokens: torch.Tensor,
             block_size: int = 512, device="cuda") -> float:
    """Sliding-window perplexity on a flat token stream."""
    model.eval()
    n = eval_tokens.numel()
    nlls = []
    for i in range(0, n - 1, block_size):
        j = min(i + block_size, n - 1)
        inp = eval_tokens[i : j + 1].unsqueeze(0).to(device)
        target = inp[:, 1:].contiguous()
        logits = model(inp[:, :-1]).logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), reduction="sum"
        )
        nlls.append(loss.item())
    ppl = math.exp(sum(nlls) / (n - 1))
    return ppl


# ───────────────────────── speed measurement ───────────────

@torch.no_grad()
def measure_speed(model, tokenizer, cfg, device) -> Dict[str, float]:
    """Measure prefill + decode throughput."""
    model.eval()
    prompt_len = cfg["speed"]["prompt_len"]
    new_tokens = cfg["speed"]["new_tokens"]
    warmup = cfg["speed"]["warmup"]
    repeats = cfg["speed"]["repeats"]

    dummy = torch.randint(0, tokenizer.vocab_size, (1, prompt_len), device=device)

    # warmup
    for _ in range(warmup):
        out = model.generate(dummy, max_new_tokens=2, do_sample=False)

    # prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(dummy)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) / repeats * 1000
    prefill_tok_s = round(prompt_len / (prefill_ms / 1000), 1)

    # decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model.generate(dummy, max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_decode_ms = (t1 - t0) / repeats * 1000
    decode_ms = total_decode_ms - prefill_ms  # approx decode-only
    decode_tok_s = round(new_tokens / (decode_ms / 1000), 1) if decode_ms > 0 else 0

    return dict(
        prefill_ms=round(prefill_ms, 2),
        prefill_tok_s=prefill_tok_s,
        decode_ms=round(decode_ms, 2),
        decode_tok_s=decode_tok_s,
    )


# ───────────────────────── memory helpers ──────────────────

def model_size_mb(model) -> float:
    """Total parameter bytes in MB."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return round(total / (1024 ** 2), 2)


def peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    return 0.0


# ───────────────────────── Naïve W4 groupwise quant ───────

def _naive_int4_group_quant_dequant(weight: torch.Tensor,
                                     group_size: int = 128) -> torch.Tensor:
    """Simulate groupwise 4-bit symmetric quantisation on a weight tensor.

    Steps: group → compute scale → quantise → clamp → dequantise
    Returns dequantised FP16 tensor (same shape as input).
    """
    out_feat, in_feat = weight.shape
    # pad in_feat to multiple of group_size
    pad = (group_size - in_feat % group_size) % group_size
    if pad:
        weight = torch.nn.functional.pad(weight, (0, pad))
    w = weight.reshape(-1, group_size)  # (n_groups, group_size)

    # symmetric: qmax = 2^(b-1) - 1 = 7
    qmax = 7
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    wq = (w / scale).round().clamp(-qmax - 1, qmax)
    wdq = (wq * scale).reshape(out_feat, -1)[:, :in_feat]
    return wdq.to(weight.dtype)


def apply_naive_w4a16(model, group_size: int = 128):
    """Replace Linear weights with naïve INT4 group-quantised (dequantised) weights."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.copy_(
                    _naive_int4_group_quant_dequant(module.weight.data, group_size)
                )


# ──────── Quantisation error analysis ─────────────────────

def compute_per_channel_mae(
    original_weight: torch.Tensor,
    quantised_weight: torch.Tensor,
) -> np.ndarray:
    """Mean Absolute Error per output channel."""
    err = (original_weight - quantised_weight).abs().float()
    mae = err.mean(dim=1).cpu().numpy()
    return mae


def collect_error_analysis(
    model_fp16, cfg, device
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compare per-channel MAE: naïve W4 vs AWQ W4 on a target layer."""
    target_name = cfg["error_analysis"]["target_layer"]
    naive_gs = cfg["error_analysis"]["naive_group_size"]

    # get original weight
    target_module = dict(model_fp16.named_modules())[target_name]
    w_orig = target_module.weight.data.clone()

    # naïve quant error
    w_naive = _naive_int4_group_quant_dequant(w_orig, naive_gs)
    mae_naive = compute_per_channel_mae(w_orig, w_naive)

    return mae_naive, w_orig, target_name


def get_awq_weight_for_layer(
    awq_model, target_name: str
) -> torch.Tensor:
    """Extract dequantised weight from AWQ model for a target layer."""
    target_module = dict(awq_model.model.named_modules())[target_name]
    return target_module.weight.data.clone()


def compute_awq_equiv_error(
    w_orig: torch.Tensor,
    w_awq: torch.Tensor,
    group_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equivalent per-channel quantisation error for AWQ.

    AWQ applies per-channel scaling  W' = W * diag(s)  before quantisation.
    With export_compatible=True, w_awq = Q(W * s)  (quantised & dequantised
    in the scaled space).  The per-channel scaling factor can be recovered as:
        s_j ≈ ‖w_awq[:,j]‖ / ‖w_orig[:,j]‖
    Then the equivalent error in the original space is:
        err_j = W[:,j] − Q(W * s)[:,j] / s_j

    Returns (mae_naive, mae_awq_equiv) as per-output-channel arrays.
    """
    # --- Naïve: quantise original weight directly ---
    w_naive_dq = _naive_int4_group_quant_dequant(w_orig, group_size)
    mae_naive = compute_per_channel_mae(w_orig, w_naive_dq)

    # --- AWQ equivalent error ---
    # Recover per-input-channel scale: s_j ≈ ||w_awq_col_j|| / ||w_orig_col_j||
    col_norm_orig = w_orig.float().abs().mean(dim=0).clamp(min=1e-12)   # (in,)
    col_norm_awq  = w_awq.float().abs().mean(dim=0).clamp(min=1e-12)   # (in,)
    s_approx = col_norm_awq / col_norm_orig                            # (in,)

    # Undo scaling: W_equiv = w_awq / diag(s)  (back to original space)
    w_awq_unscaled = w_awq.float() / s_approx.unsqueeze(0)

    # Per-output-channel MAE in original space
    mae_awq_equiv = compute_per_channel_mae(
        w_orig.to(w_awq_unscaled.device),
        w_awq_unscaled.to(w_orig.dtype),
    )

    return mae_naive, mae_awq_equiv


# ───────────────────────── plotting ────────────────────────

def plot_ppl_vs_groupsize(results: pd.DataFrame, out_dir: str):
    """PPL vs group size for AWQ, with baseline and naïve references."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # AWQ points
    awq = results[results["method"] == "awq"].sort_values("group_size")
    ax.plot(awq["group_size"], awq["ppl"], "o-", color="tab:green",
            markersize=8, linewidth=2, label="AWQ W4A16")

    # reference lines
    bl = results[results["method"] == "baseline"]["ppl"].values[0]
    naive = results[results["method"] == "naive_w4a16"]["ppl"].values[0]
    ax.axhline(bl, color="tab:blue", ls="--", lw=1.5, label=f"Baseline FP16 (PPL={bl:.2f})")
    ax.axhline(naive, color="tab:red", ls="--", lw=1.5, label=f"Naïve W4A16 (PPL={naive:.2f})")

    for _, r in awq.iterrows():
        ax.annotate(f"{r['ppl']:.2f}", (r["group_size"], r["ppl"]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10)

    ax.set_xlabel("Group Size", fontsize=12)
    ax.set_ylabel("Perplexity (↓)", fontsize=12)
    ax.set_title("AWQ W4A16 — PPL vs Group Size (TinyLlama / WikiText-2)", fontsize=13)
    ax.set_xticks(awq["group_size"].tolist())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "ppl_vs_groupsize.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


def plot_latency_vs_groupsize(results: pd.DataFrame, out_dir: str):
    """Latency (prefill / decode tok/s) vs group size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    awq = results[results["method"] == "awq"].sort_values("group_size")
    bl = results[results["method"] == "baseline"].iloc[0]
    naive = results[results["method"] == "naive_w4a16"].iloc[0]

    for ax, metric, ylabel, title in [
        (ax1, "prefill_tok_s", "Prefill (tokens/s ↑)", "Prefill Throughput"),
        (ax2, "decode_tok_s", "Decode (tokens/s ↑)", "Decode Throughput"),
    ]:
        ax.plot(awq["group_size"], awq[metric], "s-", color="tab:green",
                markersize=8, linewidth=2, label="AWQ W4A16")
        ax.axhline(bl[metric], color="tab:blue", ls="--", lw=1.5,
                   label=f"Baseline ({bl[metric]:.1f})")
        ax.axhline(naive[metric], color="tab:red", ls="--", lw=1.5,
                   label=f"Naïve ({naive[metric]:.1f})")

        for _, r in awq.iterrows():
            ax.annotate(f"{r[metric]:.1f}", (r["group_size"], r[metric]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

        ax.set_xlabel("Group Size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(awq["group_size"].tolist())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("AWQ W4A16 — Latency vs Group Size", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "latency_vs_groupsize.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def plot_quant_error(mae_naive: np.ndarray, mae_awq: np.ndarray,
                     layer_name: str, out_dir: str):
    """Per-output-channel MAE: before vs after AWQ scaling (equiv. error)."""
    n = len(mae_naive)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, mae_naive, width, label="Before AWQ (Naïve W4)",
           color="tab:red", alpha=0.7)
    ax.bar(x + width / 2, mae_awq, width, label="After AWQ (equiv. error)",
           color="tab:green", alpha=0.7)
    ax.set_xlabel("Output Channel Index", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (original weight space)", fontsize=11)
    ax.set_title(f"Per-Channel Quantisation Error: Before vs After AWQ — {layer_name}",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # show every 64th tick if too many
    if n > 128:
        step = n // 8
        ax.set_xticks(range(0, n, step))

    fig.tight_layout()
    path = os.path.join(out_dir, "quant_error_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


def plot_quant_error_hist(mae_naive: np.ndarray, mae_awq: np.ndarray,
                          layer_name: str, out_dir: str):
    """Histogram of per-channel MAE distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mae_naive, bins=40, alpha=0.6, color="tab:red", label="Before AWQ (Naïve W4)")
    ax.hist(mae_awq, bins=40, alpha=0.6, color="tab:green", label="After AWQ (equiv.)")
    ax.set_xlabel("Mean Absolute Error per Channel (original weight space)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"MAE Distribution: Before vs After AWQ — {layer_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "quant_error_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


# ───────────────────────── analysis text ───────────────────

def write_analysis(results: pd.DataFrame, mae_naive, mae_awq, out_dir: str,
                   target_name: str = ""):
    bl = results[results["method"] == "baseline"].iloc[0]
    naive = results[results["method"] == "naive_w4a16"].iloc[0]
    awq_rows = results[results["method"] == "awq"].sort_values("group_size")
    best = awq_rows.loc[awq_rows["ppl"].idxmin()]

    lines = [
        "=" * 70,
        "  Brief Analysis — AWQ (W4A16) on TinyLlama-1.1B / WikiText-2",
        "=" * 70,
        "",
        f"Baseline FP16       :  PPL = {bl['ppl']:.2f}  |  size = {bl['model_size_mb']:.0f} MB",
        f"Naïve W4A16 (g=128) :  PPL = {naive['ppl']:.2f}  (Δ = +{naive['ppl'] - bl['ppl']:.2f})  |  size = {naive['model_size_mb']:.0f} MB",
        "",
        "AWQ W4A16 sweep:",
    ]
    for _, r in awq_rows.iterrows():
        lines.append(
            f"  g = {int(r['group_size']):>3d}  →  PPL = {r['ppl']:.2f}  "
            f"(Δ = +{r['ppl'] - bl['ppl']:.2f})  |  size = {r['model_size_mb']:.0f} MB"
        )
    lines += [
        "",
        f"Best group size = {int(best['group_size'])}  →  PPL = {best['ppl']:.2f}",
        "",
        "--- Why does Naïve W4 degrade PPL? ---",
        "  4-bit quantisation has only 16 levels (-8 to +7). Groupwise",
        "  quantisation (group_size=128) means 128 weights share one scale.",
        "  Important weight channels that correspond to large activations",
        "  receive the same rough treatment as unimportant ones → large error.",
        "",
        "--- How does AWQ help? ---",
        "  AWQ observes which weight channels are 'important' by looking at",
        "  activation magnitudes on a calibration set. It applies per-channel",
        "  scaling so that important channels have finer quantisation granularity.",
        "  This reduces error on the channels that matter most for accuracy.",
        "",
        "--- Group size trade-off ---",
        "  Smaller group size → more scales to store (slight overhead) but finer",
        "  quantisation → better PPL. Larger group → fewer scales, smaller model,",
        "  but coarser quantisation → PPL degrades.",
        "",
        "--- Quantisation error analysis (original weight space) ---",
        f"  Target layer       : {target_name}",
        f"  Before AWQ (Naïve) : mean MAE = {mae_naive.mean():.6f}",
        f"  After AWQ (equiv.) : mean MAE = {mae_awq.mean():.6f}",
        f"  Error reduction    = {(1 - mae_awq.mean() / mae_naive.mean()) * 100:.1f}%",
        "",
        "  Note: AWQ equivalent error is computed by recovering the per-channel",
        "  scaling factor s from the AWQ output, then mapping the quantised",
        "  weights back to the original weight space:  W_equiv = Q(W*s) / s",
        "  This gives a fair comparison of quantisation error before vs after",
        "  AWQ scaling, both measured in the same coordinate system.",
        "",
    ]

    path = os.path.join(out_dir, "analysis.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → saved {path}")


# ═══════════════════════════ main ══════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/awq.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    device = pick_device(cfg["model"]["device"])
    dtype = getattr(torch, cfg["model"]["dtype"]) if "dtype" in cfg["model"] else torch.float16
    model_name = cfg["model"]["name"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  AWQ (W4A16) Experiment")
    print(f"  Model  : {model_name}")
    print(f"  Device : {device}   Dtype: {dtype}")
    print(f"  Output : {out_dir}")
    print("=" * 60)

    # ── load tokenizer ──
    print("\n[Model] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── load eval tokens ──
    print("[Data] Loading WikiText-2 evaluation tokens …")
    eval_tokens = load_eval_tokens(tokenizer, cfg)
    print(f"  Eval tokens: {eval_tokens.numel():,}")

    # ── results collector ──
    rows: List[Dict[str, Any]] = []

    # ═════════════════════════════════════════════════════════
    #  Step 1: Baseline FP16
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 1: Baseline FP16")
    print("=" * 60)

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=str(device)
    )
    model_fp16.eval()
    n_params = sum(p.numel() for p in model_fp16.parameters())
    print(f"  Parameters: {n_params:,}")

    torch.cuda.reset_peak_memory_stats()
    ppl_bl = eval_ppl(model_fp16, tokenizer, eval_tokens,
                      cfg["eval"]["block_size"], device)
    speed_bl = measure_speed(model_fp16, tokenizer, cfg, device)
    size_bl = model_size_mb(model_fp16)
    peak_bl = peak_gpu_mb()
    print(f"  PPL = {ppl_bl:.2f}")
    print(f"  Size = {size_bl:.0f} MB  |  Peak GPU = {peak_bl:.0f} MB")
    print(f"  Speed: {speed_bl}")

    rows.append(dict(method="baseline", group_size=None, ppl=ppl_bl,
                     model_size_mb=size_bl, peak_gpu_mb=peak_bl, **speed_bl))

    # ═════════════════════════════════════════════════════════
    #  Step 2: Naïve W4A16 (group_size=128)
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 2: Naïve 4-bit Weight-only PTQ (W4A16, g=128)")
    print("=" * 60)

    naive_gs = cfg["error_analysis"]["naive_group_size"]
    model_naive = copy.deepcopy(model_fp16)
    apply_naive_w4a16(model_naive, group_size=naive_gs)

    # model size: simulated, so report theoretical 4-bit size
    # each param = 0.5 bytes (4 bit) + scale overhead
    n_groups = sum(
        p.numel() // naive_gs for p in model_naive.parameters() if p.ndim == 2
    )
    # 4-bit params + FP16 scales + FP16 biases
    bits_weights = sum(p.numel() for p in model_naive.parameters() if p.ndim == 2) * 4
    bits_other = sum(p.numel() * 16 for p in model_naive.parameters() if p.ndim != 2)
    bits_scales = n_groups * 16
    size_naive = round((bits_weights + bits_other + bits_scales) / 8 / (1024 ** 2), 2)

    torch.cuda.reset_peak_memory_stats()
    ppl_naive = eval_ppl(model_naive, tokenizer, eval_tokens,
                         cfg["eval"]["block_size"], device)
    speed_naive = measure_speed(model_naive, tokenizer, cfg, device)
    peak_naive = peak_gpu_mb()

    print(f"  PPL = {ppl_naive:.2f}  (Δ = +{ppl_naive - ppl_bl:.2f})")
    print(f"  Theoretical size = {size_naive:.0f} MB  |  Peak GPU = {peak_naive:.0f} MB")
    print(f"  Speed: {speed_naive}")

    rows.append(dict(method="naive_w4a16", group_size=naive_gs, ppl=ppl_naive,
                     model_size_mb=size_naive, peak_gpu_mb=peak_naive, **speed_naive))

    # collect naïve error for Step 4
    mae_naive, w_orig, target_name = collect_error_analysis(model_fp16, cfg, device)

    del model_naive
    gc.collect()
    torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════
    #  Step 3: AWQ with group-size sweep
    # ═════════════════════════════════════════════════════════
    from awq import AutoAWQForCausalLM

    group_sizes = cfg["awq"]["group_sizes"]
    mae_awq_best = None
    best_ppl = float("inf")

    for gs in group_sizes:
        print("\n" + "=" * 60)
        print(f"  Step 3: AWQ W4A16 (group_size = {gs})")
        print("=" * 60)

        quant_config = {
            "zero_point": cfg["awq"]["zero_point"],
            "q_group_size": gs,
            "w_bit": cfg["awq"]["w_bit"],
            "version": cfg["awq"]["version"],
        }

        # Load fresh model via AutoAWQ
        print("  Loading model via AutoAWQ …")
        awq_model = AutoAWQForCausalLM.from_pretrained(
            model_name, safetensors=True, device_map=str(device)
        )

        # Pre-load calibration data as list of strings (AutoAWQ needs this
        # for datasets that require a config name like wikitext)
        print(f"  Quantizing with AWQ (group_size={gs}) …")
        calib_ds = load_dataset(
            cfg["calibration"]["dataset"],
            cfg["calibration"]["config"],
            split=cfg["calibration"]["split"],
        )
        calib_texts = [t for t in calib_ds["text"] if len(t.strip()) > 0]
        calib_texts = calib_texts[: cfg["calibration"]["max_samples"]]

        awq_model.quantize(
            tokenizer=tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts,       # pass pre-loaded list of strings
            duo_scaling=cfg["awq"]["duo_scaling"],
            apply_clip=cfg["awq"]["apply_clip"],
            max_calib_samples=cfg["calibration"]["max_samples"],
            max_calib_seq_len=cfg["calibration"]["max_seq_len"],
            export_compatible=True,   # keep weights in FP16 (dequant) for eval
        )

        # save & reload for proper measurement
        save_path = os.path.join(out_dir, f"awq_g{gs}")
        print(f"  Saving quantised model to {save_path} …")
        awq_model.save_quantized(save_path)

        # Evaluate using the export_compatible (FP16 dequant) model
        hf_model = awq_model.model
        hf_model.to(device)
        hf_model.eval()

        torch.cuda.reset_peak_memory_stats()
        ppl_awq = eval_ppl(hf_model, tokenizer, eval_tokens,
                           cfg["eval"]["block_size"], device)
        speed_awq = measure_speed(hf_model, tokenizer, cfg, device)
        size_awq = model_size_mb(hf_model)
        peak_awq = peak_gpu_mb()

        # compute theoretical 4-bit size (same formula as naïve)
        n_groups_awq = sum(
            p.numel() // gs for p in hf_model.parameters() if p.ndim == 2
        )
        bits_w = sum(p.numel() for p in hf_model.parameters() if p.ndim == 2) * 4
        bits_o = sum(p.numel() * 16 for p in hf_model.parameters() if p.ndim != 2)
        bits_s = n_groups_awq * 16
        size_awq_theory = round((bits_w + bits_o + bits_s) / 8 / (1024 ** 2), 2)

        print(f"  PPL = {ppl_awq:.2f}  (Δ = +{ppl_awq - ppl_bl:.2f})")
        print(f"  Theoretical size = {size_awq_theory:.0f} MB  |  Peak GPU = {peak_awq:.0f} MB")
        print(f"  Speed: {speed_awq}")

        rows.append(dict(method="awq", group_size=gs, ppl=ppl_awq,
                         model_size_mb=size_awq_theory, peak_gpu_mb=peak_awq,
                         **speed_awq))

        # collect AWQ error for the best group_size
        if ppl_awq < best_ppl:
            best_ppl = ppl_awq
            w_awq = get_awq_weight_for_layer(awq_model, target_name)
            _, mae_awq_best = compute_awq_equiv_error(
                w_orig, w_awq, group_size=gs
            )

        del awq_model, hf_model
        gc.collect()
        torch.cuda.empty_cache()

    # ── save results CSV ──
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "awq_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")
    print(df.to_string(index=False))

    # ═════════════════════════════════════════════════════════
    #  Step 4: Plotting
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 4: Plotting")
    print("=" * 60)
    plot_ppl_vs_groupsize(df, out_dir)
    plot_latency_vs_groupsize(df, out_dir)

    # ═════════════════════════════════════════════════════════
    #  Step 5: Quantisation error analysis
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 5: Quantisation error analysis")
    print("=" * 60)
    if mae_awq_best is not None:
        plot_quant_error(mae_naive, mae_awq_best, target_name, out_dir)
        plot_quant_error_hist(mae_naive, mae_awq_best, target_name, out_dir)
        write_analysis(df, mae_naive, mae_awq_best, out_dir,
                       target_name=target_name)
    else:
        print("  ⚠ No AWQ results to compare")

    print("\n" + "=" * 60)
    print(f"  ✓  ALL DONE — results in {out_dir}/")
    print("=" * 60 + "\n")

    # cleanup FP16 model
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
