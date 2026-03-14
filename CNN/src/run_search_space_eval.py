#!/usr/bin/env python3
"""
Search Space Quality Evaluation for MobileNetV2-like CNNs on CIFAR-10.

Evaluates the quality of different architecture search spaces by:
  1. Defining a flexible MobileNetV2 backbone with 4 configurable dimensions:
     input resolution, kernel size, width multiplier, depth multiplier
  2. Creating sub-search spaces that fix 2 dimensions and vary the other 2
  3. Sampling architectures and measuring validation accuracy, FLOPs, peak activation memory
  4. Analyzing search space quality under a memory constraint via CDF plot

Usage (from CNN/src/ or CNN/):
    python run_search_space_eval.py --config ../configs/search_space_config.yaml
"""

import argparse
import copy
import csv
import gc
import itertools
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import yaml
except ImportError:
    yaml = None


# ========================================================================
# Config
# ========================================================================

def load_config(path: str) -> dict:
    if yaml is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    import json
    with open(path, "r") as f:
        return json.load(f)


# ========================================================================
# Utility
# ========================================================================

def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ========================================================================
# Flexible MobileNetV2 for CIFAR-10
# ========================================================================

class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int,
                 expand_ratio: int, kernel_size: int = 3):
        super().__init__()
        self.use_res_connect = (stride == 1 and inp == oup)
        hidden_dim = int(round(inp * expand_ratio))
        padding = (kernel_size - 1) // 2

        layers: list = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class FlexMobileNetV2(nn.Module):
    """
    MobileNetV2 with 4 configurable dimensions (adapted for CIFAR-10):
      - width_mult  : scales channel counts in every layer
      - kernel_size : depthwise convolution kernel size
      - depth_mult  : scales number of inverted-residual blocks per stage
    Input resolution is handled externally via data transforms.
    """

    # Base inverted-residual settings for CIFAR-10
    # (expansion, out_channels, num_blocks, stride)
    # Strides adjusted for 32×32: first conv stride=1, fewer down-samples
    BASE_CFGS = [
        (1,  16,  1, 1),
        (6,  24,  2, 1),   # keep spatial for small images
        (6,  32,  3, 2),
        (6,  64,  4, 2),
        (6,  96,  3, 1),
        (6, 160,  3, 2),
        (6, 320,  1, 1),
    ]

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0,
                 kernel_size: int = 3, depth_mult: float = 1.0):
        super().__init__()
        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * max(1.0, width_mult))

        # First convolution (stride=1 for CIFAR-10)
        features: list = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )]

        for t, c, n, s in self.BASE_CFGS:
            output_channel = _make_divisible(c * width_mult)
            n_blocks = max(1, round(n * depth_mult))
            for i in range(n_blocks):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    input_channel, output_channel, stride, t, kernel_size
                ))
                input_channel = output_channel

        features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        ))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ========================================================================
# Metrics
# ========================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_flops(model: nn.Module, resolution: int) -> Optional[float]:
    """Return GFLOPs for a single image, or None if thop unavailable."""
    if thop_profile is None:
        return None
    m = copy.deepcopy(model).eval().cpu()
    x = torch.randn(1, 3, resolution, resolution)
    with torch.inference_mode():
        macs, _ = thop_profile(m, inputs=(x,), verbose=False)
    del m
    return float(macs) * 2.0 / 1e9


def compute_peak_activation_memory_mb(
    model: nn.Module, resolution: int, device: torch.device,
) -> float:
    """
    Estimate peak activation memory (MB) during a single-image inference.

    CUDA path : max_memory_allocated minus model-parameter footprint.
    CPU  path : hook-based estimation (max consecutive-pair output sizes × 1.5).
    """
    model = model.eval().to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        base_mem = torch.cuda.memory_allocated(device)   # model params on GPU

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        x = torch.randn(1, 3, resolution, resolution, device=device)
        input_bytes = x.nelement() * x.element_size()
        with torch.inference_mode():
            _ = model(x)
        torch.cuda.synchronize(device)

        peak = torch.cuda.max_memory_allocated(device)
        act_bytes = peak - base_mem - input_bytes
        del x
        torch.cuda.empty_cache()
        return max(0.0, act_bytes / (1024 ** 2))

    # CPU fallback ---------------------------------------------------------
    output_sizes: list = []
    hooks: list = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            output_sizes.append(out.nelement() * out.element_size())

    for m in model.modules():
        if not list(m.children()):
            hooks.append(m.register_forward_hook(hook_fn))

    x = torch.randn(1, 3, resolution, resolution)
    with torch.inference_mode():
        _ = model(x)
    for h in hooks:
        h.remove()

    if not output_sizes:
        return 0.0

    peak = max(output_sizes)
    for i in range(len(output_sizes) - 1):
        peak = max(peak, output_sizes[i] + output_sizes[i + 1])
    # 1.5× overhead accounts for skip-connection tensors co-existing in memory
    return peak * 1.5 / (1024 ** 2)


# ========================================================================
# Data
# ========================================================================

def make_cifar10_loaders(
    data_dir: str, train_bs: int, eval_bs: int, num_workers: int,
    train_limit: int, test_limit: int, seed: int, resolution: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    if resolution == 32:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomCrop(resolution, padding=max(1, resolution // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    if 0 < train_limit < len(train_ds):
        idx = torch.randperm(len(train_ds), generator=g)[:train_limit].tolist()
        train_ds = Subset(train_ds, idx)
    if 0 < test_limit < len(test_ds):
        idx = torch.randperm(len(test_ds), generator=g)[:test_limit].tolist()
        test_ds = Subset(test_ds, idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=eval_bs, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


# ========================================================================
# Training
# ========================================================================

def train_and_evaluate(
    model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
    device: torch.device, epochs: int, lr: float,
) -> float:
    """Train with SGD + cosine LR for *epochs*, return final test accuracy."""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # ---- evaluate ---------------------------------------------------------
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0


# ========================================================================
# Search Space Definition
# ========================================================================

@dataclass
class ArchConfig:
    resolution: int
    kernel_size: int
    width_mult: float
    depth_mult: float


@dataclass
class SearchResult:
    search_space: str
    arch_id: int
    resolution: int
    kernel_size: int
    width_mult: float
    depth_mult: float
    num_parameters: int
    gflops: float
    peak_activation_memory_mb: float
    accuracy: float


def define_search_spaces(cfg: dict) -> Dict[str, List[ArchConfig]]:
    """
    Build 3 sub-search spaces, each fixing 2 dimensions and varying the other 2.
    Samples without replacement from the Cartesian product of the two free dims.
    """
    n_samples    = cfg.get("n_samples_per_space", 20)
    seed         = cfg.get("seed", 42)
    rng          = random.Random(seed)

    resolutions  = cfg.get("resolutions",  [24, 28, 32, 36, 40, 48])
    kernel_sizes = cfg.get("kernel_sizes",  [3, 5, 7])
    width_mults  = cfg.get("width_mults",   [0.35, 0.5, 0.75, 1.0, 1.25, 1.5])
    depth_mults  = cfg.get("depth_mults",   [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

    default_res    = cfg.get("default_resolution", 32)
    default_kernel = cfg.get("default_kernel", 3)
    default_width  = cfg.get("default_width", 1.0)
    default_depth  = cfg.get("default_depth", 1.0)

    def _sample(combos):
        return rng.sample(combos, min(n_samples, len(combos)))

    spaces: Dict[str, List[ArchConfig]] = {}

    # SS1: Resolution × Width   (fix kernel, depth)
    combos = list(itertools.product(resolutions, width_mults))
    spaces["SS1_ResWidth"] = [
        ArchConfig(r, default_kernel, w, default_depth) for r, w in _sample(combos)
    ]

    # SS2: Kernel × Depth       (fix resolution, width)
    combos = list(itertools.product(kernel_sizes, depth_mults))
    spaces["SS2_KernelDepth"] = [
        ArchConfig(default_res, k, default_width, d) for k, d in _sample(combos)
    ]

    # SS3: Resolution × Depth   (fix kernel, width)
    combos = list(itertools.product(resolutions, depth_mults))
    spaces["SS3_ResDepth"] = [
        ArchConfig(r, default_kernel, default_width, d) for r, d in _sample(combos)
    ]

    return spaces


# ========================================================================
# Single Architecture Evaluation
# ========================================================================

def evaluate_architecture(
    arch: ArchConfig, arch_id: int, space_name: str,
    data_dir: str, device: torch.device, train_cfg: dict,
) -> Optional[SearchResult]:
    model = FlexMobileNetV2(
        num_classes=10,
        width_mult=arch.width_mult,
        kernel_size=arch.kernel_size,
        depth_mult=arch.depth_mult,
    )
    n_params = count_parameters(model)
    gflops   = compute_flops(model, arch.resolution) or 0.0
    peak_mem = compute_peak_activation_memory_mb(model, arch.resolution, device)

    # Adapt batch size for large models to avoid OOM
    base_train_bs = train_cfg.get("train_batch_size", 128)
    base_eval_bs  = train_cfg.get("eval_batch_size", 256)
    # Scale down for large resolution × width combinations
    scale = (arch.resolution / 32) ** 2 * arch.width_mult
    if scale > 1.5:
        base_train_bs = max(16, base_train_bs // int(scale))
        base_eval_bs  = max(16, base_eval_bs  // int(scale))

    train_loader, test_loader = make_cifar10_loaders(
        data_dir=data_dir,
        train_bs=base_train_bs,
        eval_bs=base_eval_bs,
        num_workers=train_cfg.get("num_workers", 2),
        train_limit=train_cfg.get("train_limit", 10000),
        test_limit=train_cfg.get("test_limit", 2000),
        seed=train_cfg.get("seed", 42),
        resolution=arch.resolution,
    )

    try:
        accuracy = train_and_evaluate(
            model, train_loader, test_loader, device,
            epochs=train_cfg.get("epochs", 10),
            lr=train_cfg.get("lr", 0.05),
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] Skipping arch (res={arch.resolution}, w={arch.width_mult})")
            accuracy = -1.0  # mark as failed
        else:
            raise
    finally:
        del model, train_loader, test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if accuracy < 0:
        return None

    return SearchResult(
        search_space=space_name, arch_id=arch_id,
        resolution=arch.resolution, kernel_size=arch.kernel_size,
        width_mult=arch.width_mult, depth_mult=arch.depth_mult,
        num_parameters=n_params, gflops=gflops,
        peak_activation_memory_mb=peak_mem, accuracy=accuracy,
    )


# ========================================================================
# Plotting & Analysis
# ========================================================================

def plot_cdf_flops(results: Dict[str, List[SearchResult]],
                   memory_limit_mb: float, output_path: str):
    """CDF of FLOPs for each search space (models within memory constraint)."""
    if plt is None:
        print("[WARN] matplotlib unavailable – skip CDF plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, rows) in enumerate(results.items()):
        filtered = [r for r in rows if r.peak_activation_memory_mb <= memory_limit_mb]
        if not filtered:
            print(f"  {name}: no models within memory constraint")
            continue

        flops = sorted(r.gflops for r in filtered)
        n = len(flops)
        cdf = [(i + 1) / n for i in range(n)]
        color = colors[idx % len(colors)]

        ax.step(flops, cdf, where="post", label=f"{name} (n={n})",
                color=color, linewidth=2)

        p80_idx = int(math.ceil(0.8 * n)) - 1
        p80_val = flops[p80_idx]
        ax.axvline(p80_val, color=color, linestyle="--", alpha=0.6)
        ax.plot(p80_val, 0.8, "o", color=color, markersize=8)
        ax.annotate(f"p80={p80_val:.3f}", xy=(p80_val, 0.8),
                    xytext=(p80_val + 0.003, 0.83), fontsize=9, color=color)

    ax.set_xlabel("GFLOPs", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(f"CDF of FLOPs  (peak activation memory ≤ {memory_limit_mb:.1f} MB)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  CDF plot saved → {output_path}")


def plot_scatter_metrics(results: Dict[str, List[SearchResult]], output_path: str):
    """Three scatter sub-plots: FLOPs-Acc, Memory-Acc, FLOPs-Memory."""
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, rows) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        flops = [r.gflops for r in rows]
        accs  = [r.accuracy * 100 for r in rows]
        mems  = [r.peak_activation_memory_mb for r in rows]

        axes[0].scatter(flops, accs, c=c, label=name, alpha=0.7, s=40)
        axes[1].scatter(mems,  accs, c=c, label=name, alpha=0.7, s=40)
        axes[2].scatter(flops, mems, c=c, label=name, alpha=0.7, s=40)

    for ax, xl, yl, t in [
        (axes[0], "GFLOPs", "Accuracy (%)", "FLOPs vs Accuracy"),
        (axes[1], "Peak Act. Memory (MB)", "Accuracy (%)", "Memory vs Accuracy"),
        (axes[2], "GFLOPs", "Peak Act. Memory (MB)", "FLOPs vs Memory"),
    ]:
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved → {output_path}")


def plot_accuracy_boxplot(results: Dict[str, List[SearchResult]], output_path: str):
    """Box-plot of accuracy distribution per search space."""
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for name, rows in results.items():
        data.append([r.accuracy * 100 for r in rows])
        labels.append(name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Distribution per Search Space")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Boxplot saved → {output_path}")


def generate_analysis(results: Dict[str, List[SearchResult]],
                      memory_limit_mb: float, output_path: str):
    lines: list = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("Search Space Quality Evaluation – Analysis Report")
    lines.append(sep)
    lines.append(f"\nMemory constraint: peak activation memory ≤ {memory_limit_mb:.1f} MB\n")

    space_stats: dict = {}
    for name, rows in results.items():
        filtered = [r for r in rows if r.peak_activation_memory_mb <= memory_limit_mb]
        accs  = [r.accuracy * 100 for r in rows]
        flops = [r.gflops for r in rows]
        mems  = [r.peak_activation_memory_mb for r in rows]

        lines.append(f"\n--- {name} ---")
        lines.append(f"  Total architectures sampled : {len(rows)}")
        lines.append(f"  Meet memory constraint      : {len(filtered)} / {len(rows)}")
        lines.append(f"  Accuracy range              : {min(accs):.1f}% – {max(accs):.1f}%")
        lines.append(f"  FLOPs range                 : {min(flops):.4f} – {max(flops):.4f} GFLOPs")
        lines.append(f"  Activation memory range     : {min(mems):.2f} – {max(mems):.2f} MB")

        if filtered:
            fa = [r.accuracy * 100 for r in filtered]
            ff = sorted(r.gflops for r in filtered)
            p80 = ff[int(math.ceil(0.8 * len(ff))) - 1]
            best = max(filtered, key=lambda r: r.accuracy)
            lines.append(f"  [Constrained] Accuracy      : {min(fa):.1f}% – {max(fa):.1f}%  "
                         f"(mean {sum(fa)/len(fa):.1f}%)")
            lines.append(f"  [Constrained] FLOPs p=80%   : {p80:.4f} GFLOPs")
            lines.append(f"  [Constrained] Best model    : "
                         f"acc={best.accuracy*100:.1f}%, "
                         f"FLOPs={best.gflops:.4f}, "
                         f"mem={best.peak_activation_memory_mb:.2f} MB")
            space_stats[name] = dict(
                n_feasible=len(filtered),
                ratio=len(filtered) / len(rows),
                mean_acc=sum(fa) / len(fa),
                best_acc=max(fa),
                p80_flops=p80,
            )

    # ---- comparison summary ------------------------------------------------
    lines.append(f"\n{sep}")
    lines.append("Comparison Summary")
    lines.append(sep)

    if space_stats:
        best_by_acc   = max(space_stats, key=lambda k: space_stats[k]["mean_acc"])
        lowest_p80    = min(space_stats, key=lambda k: space_stats[k]["p80_flops"])
        most_feasible = max(space_stats, key=lambda k: space_stats[k]["ratio"])

        lines.append(f"\n  Highest mean accuracy  : {best_by_acc}  "
                     f"({space_stats[best_by_acc]['mean_acc']:.1f}%)")
        lines.append(f"  Lowest FLOPs at p=80%  : {lowest_p80}  "
                     f"({space_stats[lowest_p80]['p80_flops']:.4f} GFLOPs)")
        lines.append(f"  Highest feasibility    : {most_feasible}  "
                     f"({space_stats[most_feasible]['ratio']*100:.0f}%)")
        lines.append(f"\n  Conclusion:")
        lines.append(f"  Under the memory constraint of {memory_limit_mb:.1f} MB,")
        lines.append(f"  '{best_by_acc}' produces the highest-quality architectures on average,")
        lines.append(f"  while '{lowest_p80}' reaches 80% of its quality at the lowest FLOPs cost.")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nAnalysis saved → {output_path}")


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="Search Space Quality Evaluation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- device -----------------------------------------------------------
    dev_str = cfg.get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    torch.backends.cudnn.benchmark = False  # deterministic algo selection

    data_dir   = cfg.get("data_dir", "../data")
    output_dir = cfg.get("output_dir", "../CNN_pruning/results_search_space")
    os.makedirs(output_dir, exist_ok=True)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    train_cfg       = cfg.get("training", {})
    search_cfg      = cfg.get("search_space", {})
    memory_limit_mb = cfg.get("memory_limit_mb", 5.0)

    print(f"Device          : {device}")
    print(f"Output dir      : {output_dir}")
    print(f"Memory limit    : {memory_limit_mb} MB")

    # ==== Step 1: Baseline =================================================
    print("\n" + "=" * 55)
    print(" Step 1  Baseline Evaluation")
    print("=" * 55)

    bl = cfg.get("baseline", {})
    baseline_arch = ArchConfig(
        resolution=bl.get("resolution", 32),
        kernel_size=bl.get("kernel_size", 3),
        width_mult=bl.get("width_mult", 1.0),
        depth_mult=bl.get("depth_mult", 1.0),
    )

    t0 = time.time()
    baseline_result = evaluate_architecture(
        baseline_arch, arch_id=0, space_name="baseline",
        data_dir=data_dir, device=device, train_cfg=train_cfg,
    )
    bl_time = time.time() - t0

    print(f"\n  Baseline MobileNetV2 (trained in {bl_time:.0f}s):")
    print(f"    Resolution        : {baseline_result.resolution}")
    print(f"    Parameters        : {baseline_result.num_parameters:,}")
    print(f"    GFLOPs            : {baseline_result.gflops:.4f}")
    print(f"    Peak Act. Memory  : {baseline_result.peak_activation_memory_mb:.2f} MB")
    print(f"    Test Accuracy     : {baseline_result.accuracy*100:.2f}%")

    # ==== Step 2: Search Space Evaluation ==================================
    print("\n" + "=" * 55)
    print(" Step 2  Search Space Evaluation")
    print("=" * 55)

    spaces = define_search_spaces(search_cfg)
    all_results: Dict[str, List[SearchResult]] = {}

    for space_name, arch_list in spaces.items():
        print(f"\n--- {space_name}  ({len(arch_list)} architectures) ---")
        space_results: list = []

        for i, arch in enumerate(arch_list):
            tag = (f"  [{i+1:>2}/{len(arch_list)}]  "
                   f"res={arch.resolution:<3} k={arch.kernel_size} "
                   f"w={arch.width_mult:.2f} d={arch.depth_mult:.2f}")
            print(tag, end="  …  ", flush=True)
            t0 = time.time()

            result = evaluate_architecture(
                arch, arch_id=i + 1, space_name=space_name,
                data_dir=data_dir, device=device, train_cfg=train_cfg,
            )

            dt = time.time() - t0
            if result is not None:
                space_results.append(result)
                print(f"acc={result.accuracy*100:5.1f}%  "
                      f"FLOPs={result.gflops:.4f}  "
                      f"mem={result.peak_activation_memory_mb:.2f}MB  "
                      f"[{dt:.0f}s]")
            else:
                print(f"SKIPPED (OOM)  [{dt:.0f}s]")

        all_results[space_name] = space_results

    # ---- save all results to CSV ------------------------------------------
    csv_path = os.path.join(output_dir, "search_space_results.csv")
    all_rows = [r for r in [baseline_result] if r is not None]
    for sr in all_results.values():
        all_rows.extend(sr)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    print(f"\nCSV saved → {csv_path}")

    # ==== Step 3: Quality Analysis =========================================
    print("\n" + "=" * 55)
    print(" Step 3  Search Space Quality Analysis")
    print("=" * 55)

    # ---- auto-adjust memory limit if too tight / too loose ----------------
    all_mems = [r.peak_activation_memory_mb
                for rs in all_results.values() for r in rs]
    if all_mems:
        all_mems_sorted = sorted(all_mems)
        n_within = sum(1 for m in all_mems if m <= memory_limit_mb)
        pct = n_within / len(all_mems) * 100
        print(f"\n  Measured activation-memory range: "
              f"{all_mems_sorted[0]:.2f} – {all_mems_sorted[-1]:.2f} MB")
        print(f"  Models within {memory_limit_mb:.1f} MB constraint: "
              f"{n_within}/{len(all_mems)} ({pct:.0f}%)")
        if n_within == 0:
            new_limit = all_mems_sorted[int(0.8 * len(all_mems_sorted))]
            print(f"  [AUTO] Raising memory limit to p80 = {new_limit:.2f} MB")
            memory_limit_mb = new_limit
        elif pct > 95:
            new_limit = all_mems_sorted[int(0.5 * len(all_mems_sorted))]
            print(f"  [AUTO] Tightening memory limit to median = {new_limit:.2f} MB")
            memory_limit_mb = new_limit

    plot_cdf_flops(all_results, memory_limit_mb,
                   os.path.join(output_dir, "cdf_flops.png"))
    plot_scatter_metrics(all_results,
                        os.path.join(output_dir, "scatter_metrics.png"))
    plot_accuracy_boxplot(all_results,
                         os.path.join(output_dir, "accuracy_boxplot.png"))
    generate_analysis(all_results, memory_limit_mb,
                      os.path.join(output_dir, "analysis.txt"))

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
