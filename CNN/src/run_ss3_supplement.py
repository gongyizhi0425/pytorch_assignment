#!/usr/bin/env python3
"""
Quick supplementary run for SS3 (Resolution × Depth) search space.
Runs with aggressive batch size scaling and a timeout per model.
"""
import copy
import gc
import itertools
import os
import random
import signal
import sys
import time

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

# ---- import model from main script ----
sys.path.insert(0, os.path.dirname(__file__))
from run_search_space_eval import (
    FlexMobileNetV2, count_parameters, compute_flops,
    compute_peak_activation_memory_mb, make_cifar10_loaders, load_config,
)

OUT_DIR = "../results_search_space"
DATA_DIR = "../data"


def train_and_eval_with_timeout(model, train_loader, test_loader, device,
                                 epochs, lr, timeout_s=300):
    """Train with a per-model wall-time timeout."""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    t_start = time.time()

    for ep in range(epochs):
        if time.time() - t_start > timeout_s:
            print(f"  [TIMEOUT after {ep} epochs]", end=" ")
            break
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0


def main():
    device = torch.device("cpu")  # CUDA unavailable after crash, use CPU
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    rng = random.Random(seed)
    # Skip the first two random.sample calls (SS1 and SS2) to match seeds
    resolutions = [24, 28, 32, 36, 40, 48]
    width_mults = [0.35, 0.5, 0.75, 1.0, 1.25, 1.5]
    kernel_sizes = [3, 5, 7]
    depth_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Reproduce first two samples to advance RNG
    rng.sample(list(itertools.product(resolutions, width_mults)), 20)  # SS1
    rng.sample(list(itertools.product(kernel_sizes, depth_mults)), 18)  # SS2

    # SS3: Resolution × Depth (fix kernel=3, width=1.0)
    combos = list(itertools.product(resolutions, depth_mults))
    chosen = rng.sample(combos, 20)

    # Faster settings for CPU: fewer epochs, smaller subset
    train_limit = 5000
    test_limit = 1000
    epochs = 5

    print(f"Device: {device}")
    print(f"SS3_ResDepth: {len(chosen)} architectures")
    print(f"Training: {epochs} epochs, {train_limit} train samples, {test_limit} test samples\n")

    results = []
    for i, (res, depth) in enumerate(chosen):
        # Aggressive batch size scaling
        scale = (res / 32) ** 2 * 1.0  # width=1.0 fixed
        train_bs = max(16, int(128 / max(1, scale)))
        eval_bs = max(16, int(256 / max(1, scale)))

        model = FlexMobileNetV2(num_classes=10, width_mult=1.0, kernel_size=3, depth_mult=depth)
        n_params = count_parameters(model)
        gflops = compute_flops(model, res) or 0.0
        peak_mem = compute_peak_activation_memory_mb(model, res, device)

        train_loader, test_loader = make_cifar10_loaders(
            data_dir=DATA_DIR, train_bs=train_bs, eval_bs=eval_bs,
            num_workers=2, train_limit=train_limit, test_limit=test_limit,
            seed=42, resolution=res,
        )

        print(f"  [{i+1:>2}/{len(chosen)}]  res={res:<3} k=3 w=1.00 d={depth:.2f}  "
              f"(bs={train_bs})", end="  …  ", flush=True)
        t0 = time.time()

        try:
            accuracy = train_and_eval_with_timeout(
                model, train_loader, test_loader, device,
                epochs=epochs, lr=0.05, timeout_s=300,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"SKIPPED (OOM)")
                accuracy = -1
            else:
                raise
        finally:
            del model, train_loader, test_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        dt = time.time() - t0
        if accuracy >= 0:
            results.append({
                "resolution": res, "kernel_size": 3, "width_mult": 1.0,
                "depth_mult": depth, "accuracy": accuracy, "gflops": gflops,
                "peak_activation_memory_mb": peak_mem, "num_parameters": n_params,
            })
            print(f"acc={accuracy*100:5.1f}%  FLOPs={gflops:.4f}  "
                  f"mem={peak_mem:.2f}MB  [{dt:.0f}s]")

    # Save SS3 results
    import csv
    csv_path = os.path.join(OUT_DIR, "ss3_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nSS3 results ({len(results)} models) → {csv_path}")
    else:
        print("\nNo SS3 results collected!")


if __name__ == "__main__":
    main()
