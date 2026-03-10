#!/usr/bin/env python3
"""
K-means Quantization vs. Linear Quantization (PTQ)
===================================================
Model      : torchvision.models.resnet18 (CIFAR-10 stem, 32×32)
Dataset    : CIFAR-10 (test set for evaluation; 1,000 train images for calibration)
Metrics    : Top-1 Accuracy, Model Size (MB), CPU Latency (ms/image, batch=1)

Steps:
  1. Baseline — load FP32 model, measure all metrics.
  2. Linear Quantization (weights-only) — b ∈ {8, 4}, per-layer scale,
     dequantize back to FP32 for inference (simulated PTQ).
  3. K-means Quantization (weights-only) — collect activation statistics on
     calibration set, run k-means with K = 2^b per Conv/FC layer, store
     index + codebook, reconstruct via lookup.
  4. Per-layer sensitivity analysis + brief written analysis.

Usage (from project root):
  python src/run_quantization.py \\
      --ckpt CNN_pruning/results_src/hw90_baseline_best.pt \\
      --out-dir CNN_pruning/results_quantization

  # Or train baseline from scratch (≈80 epochs, needs GPU):
  python src/run_quantization.py --train-baseline --device cuda \\
      --out-dir CNN_pruning/results_quantization
"""

import argparse
import copy
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# ====================================================================
# 1.  Model & Data
# ====================================================================

def build_resnet18_cifar10(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Build ResNet-18 with CIFAR-10 stem (3×3 conv1, no maxpool)."""
    try:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
    except Exception:
        model = tv_models.resnet18(pretrained=pretrained)
    # CIFAR-10 stem: smaller first conv + remove max-pool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_cifar10_loaders(
    data_dir: str = "./data",
    calib_size: int = 1000,
    eval_bs: int = 128,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (calibration_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=tf,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=tf,
    )

    g = torch.Generator().manual_seed(seed)
    calib_idx = torch.randperm(len(train_ds), generator=g)[:calib_size].tolist()
    calib_ds = Subset(train_ds, calib_idx)

    calib_loader = DataLoader(calib_ds, batch_size=eval_bs, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=eval_bs, shuffle=False,
                              num_workers=num_workers)
    return calib_loader, test_loader


# ====================================================================
# 2.  Metric helpers
# ====================================================================

@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader,
                      device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


def model_size_fp32_mb(model: nn.Module) -> float:
    """Full FP32 model size in MB (parameters + buffers)."""
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nbytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return nbytes / (1024 ** 2)


def quant_model_size_mb(
    model: nn.Module,
    bits: int,
    codebook_entries: Optional[Dict[str, int]] = None,
) -> float:
    """
    Simulated model size when Conv2d/Linear weights are stored at *bits*-bit.

    For **linear quantization** each layer adds 8 bytes (scale + zero-point).
    For **k-means quantization** each layer adds K × 4 bytes (codebook).
    All other params / buffers (BN, bias) stay FP32.
    """
    total_bytes = 0.0
    counted: set = set()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total_bytes += w.numel() * bits / 8.0          # quantized weight
            if codebook_entries and name in codebook_entries:
                total_bytes += codebook_entries[name] * 4   # codebook (FP32)
            else:
                total_bytes += 8                            # scale + zp
            counted.add(name + ".weight")
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
                counted.add(name + ".bias")

    # Non-quantized parameters (BN gamma/beta etc.)
    for pname, p in model.named_parameters():
        if pname not in counted:
            total_bytes += p.numel() * 4

    # Buffers (BN running_mean / running_var / num_batches_tracked)
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()

    return total_bytes / (1024 ** 2)


def measure_cpu_latency(
    model: nn.Module,
    resolution: int = 32,
    warmup: int = 30,
    iters: int = 100,
) -> float:
    """CPU latency (ms / image) with batch = 1, averaged over *iters* runs."""
    model = model.cpu().eval()
    x = torch.randn(1, 3, resolution, resolution)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


# ====================================================================
# 3.  Linear Quantization (weights-only, per-layer, asymmetric)
# ====================================================================

def linear_quantize_tensor(
    weight: torch.Tensor, bits: int,
) -> Tuple[torch.Tensor, float, int]:
    """
    Asymmetric uniform quantization → dequantize back to FP32.

    Returns (dequantized_tensor, scale, zero_point).
    """
    qmin, qmax = 0, (1 << bits) - 1
    w_min, w_max = weight.min().item(), weight.max().item()
    if w_max == w_min:
        return torch.full_like(weight, w_min), 1.0, 0

    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = int(round(-w_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))

    q = torch.clamp(torch.round(weight / scale) + zero_point, qmin, qmax)
    w_hat = (q - zero_point) * scale
    return w_hat, scale, zero_point


def linear_quantize_model(model: nn.Module, bits: int) -> nn.Module:
    """Apply per-layer linear quantization to every Conv2d / Linear weight."""
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w_hat, sc, zp = linear_quantize_tensor(module.weight.data, bits)
            module.weight.data.copy_(w_hat)
    return model


# ====================================================================
# 4.  K-means Quantization (weights-only, per-layer)
# ====================================================================

def _kmeans_1d(
    data: np.ndarray, k: int, max_iter: int = 100, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient 1-D k-means clustering (numpy).

    Returns (centroids [k], labels [n]).
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    if n <= k:
        centroids = np.zeros(k, dtype=np.float32)
        centroids[:n] = data
        return centroids, np.arange(n, dtype=np.int32)

    # ---- k-means++ initialisation (incremental, O(nK) time, O(n) mem) ----
    first = rng.randint(n)
    idx = [first]
    min_d2 = (data - data[first]) ** 2          # [n]  running min distance²
    for _ in range(1, k):
        prob = min_d2 / (min_d2.sum() + 1e-30)
        new = rng.choice(n, p=prob)
        idx.append(new)
        np.minimum(min_d2, (data - data[new]) ** 2, out=min_d2)
    centroids = data[np.array(idx)].astype(np.float32)

    # ---- Lloyd iterations (chunked to limit peak memory) ----
    CHUNK = 300_000
    for _ in range(max_iter):
        # assign
        labels = np.empty(n, dtype=np.int32)
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            labels[s:e] = ((data[s:e, None] - centroids[None, :]) ** 2
                           ).argmin(axis=1)
        # update
        new_c = np.zeros(k, dtype=np.float64)
        cnt   = np.zeros(k, dtype=np.int64)
        np.add.at(new_c, labels, data.astype(np.float64))
        np.add.at(cnt,   labels, 1)
        alive = cnt > 0
        new_c[alive] /= cnt[alive]
        new_c[~alive] = centroids[~alive]
        new_c = new_c.astype(np.float32)
        if np.max(np.abs(new_c - centroids)) < 1e-7:
            centroids = new_c
            break
        centroids = new_c

    # final assignment
    labels = np.empty(n, dtype=np.int32)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        labels[s:e] = ((data[s:e, None] - centroids[None, :]) ** 2
                       ).argmin(axis=1)
    return centroids, labels


def kmeans_quantize_model(
    model: nn.Module, bits: int, seed: int = 42, verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, int]]:
    """
    Apply per-layer k-means quantization (K = 2^bits).

    Returns (quantised_model, codebook_entries_dict).
    """
    model = copy.deepcopy(model)
    K = 1 << bits
    codebook_entries: Dict[str, int] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            shape = w.shape
            flat = w.detach().cpu().numpy().ravel().astype(np.float32)
            centroids, labels = _kmeans_1d(flat, K, max_iter=80, seed=seed)
            w_hat = centroids[labels].reshape(shape)
            module.weight.data.copy_(torch.from_numpy(w_hat).to(w.device))
            codebook_entries[name] = K
            if verbose:
                print(f"    K-means {bits}b | {name:30s} | "
                      f"shape {str(list(shape)):20s} | K={K}")
    return model, codebook_entries


# ====================================================================
# 5.  Activation statistics (calibration)
# ====================================================================

@torch.inference_mode()
def collect_activation_stats(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Forward calibration data → per-layer activation statistics."""
    model.eval()
    raw: Dict[str, List[dict]] = {}
    hooks = []

    def _hook(name):
        def fn(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                raw.setdefault(name, []).append({
                    "mean": out.mean().item(),
                    "std":  out.std().item(),
                    "min":  out.min().item(),
                    "max":  out.max().item(),
                    "l2":   out.norm(2).item(),
                })
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(_hook(name)))

    for x, _ in loader:
        model(x.to(device))

    for h in hooks:
        h.remove()

    agg = {}
    for name, entries in raw.items():
        agg[name] = {
            "mean":    np.mean([e["mean"] for e in entries]),
            "std":     np.mean([e["std"]  for e in entries]),
            "min":     float(np.min([e["min"]  for e in entries])),
            "max":     float(np.max([e["max"]  for e in entries])),
            "l2_norm": np.mean([e["l2"]   for e in entries]),
        }
    return agg


# ====================================================================
# 6.  Per-layer sensitivity analysis
# ====================================================================

def per_layer_sensitivity(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    baseline_acc: float,
    bits: int,
    method: str = "linear",       # "linear" | "kmeans"
    seed: int = 42,
) -> pd.DataFrame:
    """
    Quantize **one layer at a time**, measure accuracy drop.
    Returns a DataFrame with one row per layer.
    """
    layer_info: list = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            layer_info.append((name, mod.weight.numel()))

    records = []
    for layer_name, n_weights in layer_info:
        m = copy.deepcopy(model)
        for n2, mod in m.named_modules():
            if n2 == layer_name and isinstance(mod, (nn.Conv2d, nn.Linear)):
                if method == "linear":
                    w_hat, _, _ = linear_quantize_tensor(mod.weight.data, bits)
                    mod.weight.data.copy_(w_hat)
                else:
                    K = 1 << bits
                    flat = (mod.weight.data.detach().cpu()
                            .numpy().ravel().astype(np.float32))
                    cen, lab = _kmeans_1d(flat, K, max_iter=50, seed=seed)
                    w_hat = cen[lab].reshape(mod.weight.shape)
                    mod.weight.data.copy_(
                        torch.from_numpy(w_hat).to(mod.weight.device))
                break

        acc = evaluate_accuracy(m, test_loader, device)
        drop = baseline_acc - acc
        records.append({
            "layer": layer_name,
            "method": method,
            "bits": bits,
            "num_weights": n_weights,
            "accuracy": acc,
            "accuracy_drop": drop,
        })
        del m
        gc.collect()
        print(f"    Sens | {method} {bits}b | {layer_name:30s} | "
              f"acc={acc:.4f} | Δ={drop:+.4f}")

    return pd.DataFrame(records)


# ====================================================================
# 7.  Plotting
# ====================================================================

_STYLES = {
    "Baseline FP32":  ("black",    "s", 120),
    "Linear 8-bit":   ("tab:blue", "o", 100),
    "Linear 4-bit":   ("tab:blue", "^", 100),
    "K-means 8-bit":  ("tab:red",  "o", 100),
    "K-means 4-bit":  ("tab:red",  "^", 100),
}


def plot_results(df: pd.DataFrame, out_dir: str) -> None:
    """Accuracy vs. Model-Size  &  Accuracy vs. Latency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xkey, xlabel, title in [
        (axes[0], "model_size_mb", "Model Size (MB)",
         "Accuracy vs. Model Size"),
        (axes[1], "latency_ms", "CPU Latency (ms / image)",
         "Accuracy vs. Latency"),
    ]:
        for _, row in df.iterrows():
            c, m, s = _STYLES.get(row["label"], ("gray", "x", 60))
            ax.scatter(row[xkey], row["accuracy"] * 100,
                       color=c, marker=m, s=s, zorder=5, label=row["label"])
            ax.annotate(row["label"], (row[xkey], row["accuracy"] * 100),
                        fontsize=7, textcoords="offset points",
                        xytext=(6, 4))
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "quantization_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def plot_sensitivity(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart of per-layer accuracy drop."""
    bits_list = sorted(df["bits"].unique())
    methods   = sorted(df["method"].unique())

    fig, axes = plt.subplots(len(bits_list), 1,
                             figsize=(14, 5 * len(bits_list)))
    if len(bits_list) == 1:
        axes = [axes]

    colours = {"linear": "tab:blue", "kmeans": "tab:red"}

    for ax, b in zip(axes, bits_list):
        sub = df[df["bits"] == b]
        layers = sub[sub["method"] == methods[0]]["layer"].values
        x = np.arange(len(layers))
        w = 0.35
        for i, meth in enumerate(methods):
            data = sub[sub["method"] == meth]
            drops = data["accuracy_drop"].values * 100
            ax.bar(x + i * w, drops, w, label=meth,
                   color=colours.get(meth, "gray"), alpha=0.8)
        ax.set_ylabel("Accuracy Drop (%)")
        ax.set_title(f"Per-Layer Sensitivity ({b}-bit)")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(layers, rotation=50, ha="right", fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "layer_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sensitivity plot saved: {path}")


# ====================================================================
# 8.  Brief analysis (auto-generated text)
# ====================================================================

def brief_analysis(results_df: pd.DataFrame,
                   sensitivity_df: pd.DataFrame) -> str:
    lines: list = []
    lines.append("=" * 72)
    lines.append("  Brief Analysis  —  K-means vs. Linear Quantization (PTQ)")
    lines.append("=" * 72)

    baseline = results_df[results_df["label"] == "Baseline FP32"].iloc[0]
    lines.append(
        f"\nBaseline FP32: acc = {baseline['accuracy']*100:.2f}%, "
        f"size = {baseline['model_size_mb']:.2f} MB, "
        f"latency = {baseline['latency_ms']:.2f} ms"
    )

    for b in [8, 4]:
        lin = results_df[results_df["label"] == f"Linear {b}-bit"]
        km  = results_df[results_df["label"] == f"K-means {b}-bit"]
        if lin.empty or km.empty:
            continue
        lin = lin.iloc[0]
        km  = km.iloc[0]
        better = "K-means" if km["accuracy"] >= lin["accuracy"] else "Linear"
        lines.append(f"\n--- {b}-bit ---")
        lines.append(
            f"  Linear  : acc={lin['accuracy']*100:.2f}%  "
            f"size={lin['model_size_mb']:.2f} MB  "
            f"latency={lin['latency_ms']:.2f} ms"
        )
        lines.append(
            f"  K-means : acc={km['accuracy']*100:.2f}%  "
            f"size={km['model_size_mb']:.2f} MB  "
            f"latency={km['latency_ms']:.2f} ms"
        )
        lines.append(f"  ➜ Winner: {better}")
        if better == "K-means":
            lines.append(
                "    K-means uses **non-uniform** quantization that places "
                "more centroids\n"
                "    in the high-density region of the weight distribution "
                "(typically near zero).\n"
                "    This yields lower reconstruction error than uniform "
                "linear quantization\n"
                "    where levels are evenly spaced across [w_min, w_max]."
            )
        else:
            lines.append(
                "    Linear quantization with per-layer scale already "
                "provides sufficient\n"
                "    resolution at this bit-width, while k-means may "
                "suffer from sub-optimal\n"
                "    convergence or initialization sensitivity."
            )

    lines.append(
        "\n--- Why is latency nearly the same? ---"
        "\n  Both methods *simulate* quantization by dequantizing weights "
        "back to FP32\n"
        "  before inference, so the actual computation is identical. Real "
        "speedup would\n"
        "  require hardware-level integer-arithmetic support."
    )

    # Most-sensitive layers
    lines.append("\n--- Most quantization-sensitive layers ---")
    if len(sensitivity_df) > 0:
        for meth in sorted(sensitivity_df["method"].unique()):
            for b in sorted(sensitivity_df["bits"].unique()):
                sub = sensitivity_df[
                    (sensitivity_df["method"] == meth) &
                    (sensitivity_df["bits"] == b)
                ]
                if sub.empty:
                    continue
                top3 = sub.nlargest(3, "accuracy_drop")
                desc = ", ".join(
                    f"{r['layer']} (Δ={r['accuracy_drop']*100:+.2f}%)"
                    for _, r in top3.iterrows()
                )
                lines.append(f"  {meth:7s} {b}b — {desc}")
        lines.append(
            "\n  Observation: The first conv layer (conv1) and the final "
            "FC layer tend to be\n"
            "  the most sensitive.  conv1 has very few weights so even "
            "small perturbations\n"
            "  distort low-level features; the FC layer directly outputs "
            "class logits.\n"
            "  Deeper residual blocks with many weights are generally "
            "more robust."
        )
    else:
        lines.append("  (Sensitivity analysis was skipped.)")

    return "\n".join(lines)


# ====================================================================
# 9.  Baseline training (self-contained, GPU)
# ====================================================================

def train_baseline(
    model: nn.Module,
    data_dir: str,
    device: torch.device,
    test_loader: DataLoader,
    epochs: int = 80,
    lr: float = 0.1,
    out_dir: str = ".",
) -> nn.Module:
    """Quick baseline training on CIFAR-10."""
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)

    model = model.to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                            weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[40, 60, 70], gamma=0.2)

    best_acc = 0.0
    ckpt_path = os.path.join(out_dir, "baseline_best.pt")

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            acc = evaluate_accuracy(model, test_loader, device)
            print(f"    Epoch {epoch+1:3d}/{epochs}: test_acc = {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({"state_dict": model.state_dict()}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Best baseline accuracy: {best_acc:.4f}")
    return model


# ====================================================================
# Main
# ====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="K-means vs. Linear Quantization (PTQ) — "
                    "ResNet18 / CIFAR-10",
    )
    ap.add_argument("--data-dir",       default="./data")
    ap.add_argument("--ckpt",           default="",
                    help="Path to baseline FP32 checkpoint (.pt)")
    ap.add_argument("--train-baseline", action="store_true",
                    help="Train baseline from scratch (needs GPU)")
    ap.add_argument("--epochs",         type=int,   default=80)
    ap.add_argument("--lr",             type=float,  default=0.1)
    ap.add_argument("--device",         default="cpu",
                    choices=["cpu", "cuda", "auto"])
    ap.add_argument("--calib-size",     type=int,   default=1000)
    ap.add_argument("--eval-bs",        type=int,   default=128)
    ap.add_argument("--latency-warmup", type=int,   default=30)
    ap.add_argument("--latency-iters",  type=int,   default=100)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="Skip per-layer sensitivity (faster)")
    ap.add_argument("--sensitivity-only", action="store_true",
                    help="Only run per-layer sensitivity (skip main quant)")
    ap.add_argument("--out-dir",
                    default="./CNN_pruning/results_quantization")
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    # ── device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data ──
    print("[Data] Loading CIFAR-10 ...")
    calib_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        calib_size=args.calib_size,
        eval_bs=args.eval_bs,
        seed=args.seed,
    )
    print(f"  Calibration : {len(calib_loader.dataset)} images")
    print(f"  Test        : {len(test_loader.dataset)} images")

    # ── model ──
    print("\n[Model] Building ResNet-18 (CIFAR-10 stem, 32×32) ...")
    model = build_resnet18_cifar10(pretrained=False)

    if args.ckpt:
        print(f"  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu",
                          weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    elif args.train_baseline:
        print("  Training baseline (this may take a while) ...")
        model = train_baseline(model, args.data_dir, device,
                               test_loader, args.epochs, args.lr,
                               args.out_dir)
    else:
        print("  ⚠  No checkpoint and --train-baseline not set → "
              "random weights!")

    model = model.to(device).eval()

    # ── sensitivity-only mode: skip main experiments ──
    if args.sensitivity_only:
        baseline_acc = evaluate_accuracy(model, test_loader, device)
        print(f"  Baseline accuracy: {baseline_acc * 100:.2f}%")
        # Try to load existing results
        csv_path = os.path.join(args.out_dir, "quantization_results.csv")
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame()
        # Jump directly to sensitivity
        print(f"\n{'=' * 64}")
        print(" 4.  Per-Layer Sensitivity Analysis (4-bit)")
        print(f"{'=' * 64}")
        frames = []
        for meth in ["linear", "kmeans"]:
            print(f"\n  --- {meth} 4-bit ---")
            df = per_layer_sensitivity(
                model, test_loader, device, baseline_acc,
                bits=4, method=meth, seed=args.seed,
            )
            frames.append(df)
        sensitivity_df = pd.concat(frames, ignore_index=True)
        sens_path = os.path.join(args.out_dir, "sensitivity_results.csv")
        sensitivity_df.to_csv(sens_path, index=False)
        print(f"\n  Saved: {sens_path}")
        plot_sensitivity(sensitivity_df, args.out_dir)
        # Update analysis
        analysis_text = brief_analysis(results_df, sensitivity_df)
        print(f"\n{analysis_text}")
        with open(os.path.join(args.out_dir, "analysis.txt"), "w") as f:
            f.write(analysis_text)
        print(f"\n✅  Sensitivity results saved to {args.out_dir}/")
        return

    # ================================================================
    # 1. Baseline
    # ================================================================
    print(f"\n{'=' * 64}")
    print(" 1.  Baseline (FP32)")
    print(f"{'=' * 64}")
    baseline_acc     = evaluate_accuracy(model, test_loader, device)
    baseline_size    = model_size_fp32_mb(model)
    baseline_latency = measure_cpu_latency(
        model, warmup=args.latency_warmup, iters=args.latency_iters)
    print(f"  Accuracy : {baseline_acc * 100:.2f}%")
    print(f"  Size     : {baseline_size:.2f} MB")
    print(f"  Latency  : {baseline_latency:.2f} ms/image "
          f"(CPU, batch=1, {args.latency_iters} runs)")

    results = [{
        "label": "Baseline FP32", "method": "baseline", "bits": 32,
        "accuracy": baseline_acc,
        "model_size_mb": baseline_size,
        "latency_ms": baseline_latency,
    }]

    # ── activation statistics (calibration) ──
    print("\n[Calib] Collecting activation statistics ...")
    act_stats = collect_activation_stats(model, calib_loader, device)
    act_df = pd.DataFrame([
        {"layer": n, **s} for n, s in act_stats.items()
    ])
    act_path = os.path.join(args.out_dir, "activation_stats.csv")
    act_df.to_csv(act_path, index=False)
    print(f"  Saved: {act_path}  ({len(act_stats)} layers)")

    # ================================================================
    # 2. Linear Quantization
    # ================================================================
    for bits in [8, 4]:
        print(f"\n{'=' * 64}")
        print(f" 2.  Linear Quantization ({bits}-bit, weights-only)")
        print(f"{'=' * 64}")
        model.to(device)
        qm = linear_quantize_model(model, bits)
        qm = qm.to(device)
        acc = evaluate_accuracy(qm, test_loader, device)
        sz  = quant_model_size_mb(model, bits)
        lat = measure_cpu_latency(qm, warmup=args.latency_warmup,
                                  iters=args.latency_iters)
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Size     : {sz:.2f} MB  (simulated {bits}-bit storage)")
        print(f"  Latency  : {lat:.2f} ms/image")
        results.append({
            "label": f"Linear {bits}-bit", "method": "linear",
            "bits": bits, "accuracy": acc,
            "model_size_mb": sz, "latency_ms": lat,
        })
        del qm; gc.collect()

    # ================================================================
    # 3. K-means Quantization
    # ================================================================
    for bits in [8, 4]:
        print(f"\n{'=' * 64}")
        print(f" 3.  K-means Quantization ({bits}-bit, weights-only)")
        print(f"{'=' * 64}")
        model.to(device)
        qm, cb = kmeans_quantize_model(model, bits, seed=args.seed)
        qm = qm.to(device)
        acc = evaluate_accuracy(qm, test_loader, device)
        sz  = quant_model_size_mb(model, bits, codebook_entries=cb)
        lat = measure_cpu_latency(qm, warmup=args.latency_warmup,
                                  iters=args.latency_iters)
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Size     : {sz:.2f} MB  (index + codebook)")
        print(f"  Latency  : {lat:.2f} ms/image")
        results.append({
            "label": f"K-means {bits}-bit", "method": "kmeans",
            "bits": bits, "accuracy": acc,
            "model_size_mb": sz, "latency_ms": lat,
        })
        del qm; gc.collect()

    # ── results table ──
    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 64}")
    print(" Summary")
    print(f"{'=' * 64}")
    summary_cols = ["label", "accuracy", "model_size_mb", "latency_ms"]
    print(results_df[summary_cols].to_string(index=False))
    csv_path = os.path.join(args.out_dir, "quantization_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── plots ──
    plot_results(results_df, args.out_dir)

    # ================================================================
    # 4. Per-layer sensitivity
    # ================================================================
    sensitivity_df = pd.DataFrame()
    if not args.skip_sensitivity:
        print(f"\n{'=' * 64}")
        print(" 4.  Per-Layer Sensitivity Analysis (4-bit)")
        print(f"{'=' * 64}")
        frames = []
        for meth in ["linear", "kmeans"]:
            print(f"\n  --- {meth} 4-bit ---")
            df = per_layer_sensitivity(
                model, test_loader, device, baseline_acc,
                bits=4, method=meth, seed=args.seed,
            )
            frames.append(df)
        sensitivity_df = pd.concat(frames, ignore_index=True)
        sens_path = os.path.join(args.out_dir, "sensitivity_results.csv")
        sensitivity_df.to_csv(sens_path, index=False)
        print(f"\n  Saved: {sens_path}")
        plot_sensitivity(sensitivity_df, args.out_dir)

    # ── analysis ──
    analysis_text = brief_analysis(results_df, sensitivity_df)
    print(f"\n{analysis_text}")
    with open(os.path.join(args.out_dir, "analysis.txt"), "w") as f:
        f.write(analysis_text)

    print(f"\n✅  All results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
