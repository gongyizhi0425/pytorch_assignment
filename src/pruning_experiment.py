import argparse
import csv
import gc
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Subset

try:
    import psutil
except Exception:
    psutil = None

try:
    from thop import profile as thop_profile  # type: ignore
except Exception:
    thop_profile = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import torch.nn.utils.prune as prune
except Exception:
    prune = None


# -----------------
# Helpers
# -----------------

def _mb(nbytes: int) -> float:
    return nbytes / (1024**2)


def _device_from_str(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_nm_list(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        n_str, m_str = part.split(":", 1)
        out.append((int(n_str), int(m_str)))
    return out


def _count_nonzero_params(model: nn.Module) -> Tuple[int, int, float]:
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    sparsity = 0.0 if total == 0 else 1.0 - (nonzero / total)
    return nonzero, total, sparsity


def _param_bytes_assuming_dtype(model: nn.Module, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    total_numel = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    return int(total_numel * bytes_per)


def _effective_nonzero_weight_bytes(model: nn.Module, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    nonzero = 0
    for p in model.parameters():
        nonzero += int(torch.count_nonzero(p).item())
    return int(nonzero * bytes_per)


def _compute_macs_flops_per_image(model: nn.Module, resolution: int) -> Tuple[Optional[float], Optional[float]]:
    """Return (GMACs/img, GFLOPs/img) using thop if available."""
    if thop_profile is None:
        return None, None
    model = model.eval().cpu()
    x = torch.randn(1, 3, resolution, resolution)
    with torch.inference_mode():
        macs, _params = thop_profile(model, inputs=(x,), verbose=False)
    gmacs = float(macs) / 1e9
    gflops = float(macs) * 2.0 / 1e9
    return gmacs, gflops


def _measure_latency_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    resolution: int,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    """Return (ms_per_image, images_per_s) measured on random input."""
    model = model.eval().to(device)
    x = torch.randn(batch_size, 3, resolution, resolution, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    total_s = t1 - t0
    ms_per_image = (total_s / (iters * batch_size)) * 1000.0
    images_per_s = (iters * batch_size) / max(1e-9, total_s)
    return ms_per_image, images_per_s


def _peak_rss_during(fn, poll_s: float = 0.005) -> Optional[float]:
    if psutil is None:
        return None
    proc = psutil.Process()
    peak = proc.memory_info().rss
    done = False

    def poll_loop():
        nonlocal peak, done
        while not done:
            try:
                peak = max(peak, proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(poll_s)

    import threading

    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    try:
        fn()
    finally:
        done = True
        t.join(timeout=1)

    return _mb(int(peak))


def _measure_peak_gpu_allocated(fn, device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return _mb(int(torch.cuda.max_memory_allocated()))


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# -----------------
# CIFAR-10 pipeline
# -----------------

def _make_cifar10_loaders(
    data_dir: str,
    train_bs: int,
    eval_bs: int,
    num_workers: int,
    train_limit: int,
    calib_limit: int,
    test_limit: int,
    seed: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Resize CIFAR10 32x32 to 224x224 to fit ResNet18 default stem.
    # Use ImageNet stats to match torchvision pretrained behavior.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    test_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    # Deterministic subset
    g = torch.Generator().manual_seed(seed)

    if train_limit > 0 and train_limit < len(train_ds):
        idx = torch.randperm(len(train_ds), generator=g)[:train_limit].tolist()
        train_ds = Subset(train_ds, idx)

    # Calibration subset (for activation sparsity) from training set
    calib_ds = train_ds
    if calib_limit > 0 and calib_limit < len(train_ds):
        if isinstance(train_ds, Subset):
            base_idx = train_ds.indices
            idx = torch.randperm(len(base_idx), generator=g)[:calib_limit].tolist()
            calib_ds = Subset(train_ds.dataset, [base_idx[i] for i in idx])
        else:
            idx = torch.randperm(len(train_ds), generator=g)[:calib_limit].tolist()
            calib_ds = Subset(train_ds, idx)

    # Optional test subset (for quick runs)
    if test_limit > 0 and test_limit < len(test_ds):
        idx = torch.randperm(len(test_ds), generator=g)[:test_limit].tolist()
        test_ds = Subset(test_ds, idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    calib_loader = DataLoader(
        calib_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, calib_loader, test_loader


def _build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    try:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
    except Exception:
        model = tv_models.resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, opt: torch.optim.Optimizer) -> float:
    model.train()
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        total += yb.size(0)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
    return 0.0 if total == 0 else (correct / total)


@torch.inference_mode()
def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        total += yb.size(0)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
    return 0.0 if total == 0 else (correct / total)


# -----------------
# Pruning methods
# -----------------

def _clone_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)


def _apply_fine_grained_magnitude_pruning(model: nn.Module, amount: float) -> nn.Module:
    if prune is None:
        raise RuntimeError("torch.nn.utils.prune not available")

    # Apply unstructured pruning to conv/linear weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, name="weight", amount=amount)

    # Make pruning permanent (convert masked weights to zeros in weight param)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")

    return model


def _apply_channel_pruning_ln_structured(model: nn.Module, amount: float) -> nn.Module:
    if prune is None:
        raise RuntimeError("torch.nn.utils.prune not available")

    # Structured pruning on output channels of conv/linear
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
        elif isinstance(m, nn.Linear):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")

    return model


def _apply_nm_pruning(model: nn.Module, n: int, m: int) -> nn.Module:
    """Apply N:M pruning by zeroing N smallest magnitudes in each contiguous group of M.

    This is a simple mask-based implementation. It creates structured sparsity but does NOT
    change tensor shapes.
    """

    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern {n}:{m}")

    with torch.no_grad():
        for mod in model.modules():
            if not isinstance(mod, (nn.Conv2d, nn.Linear)):
                continue
            w = mod.weight
            # Flatten per-output-channel
            w2 = w.view(w.size(0), -1)
            # pad to multiple of m
            pad = (m - (w2.size(1) % m)) % m
            if pad:
                w2p = torch.cat(
                    [w2, torch.zeros(w2.size(0), pad, device=w2.device, dtype=w2.dtype)],
                    dim=1,
                )
            else:
                w2p = w2
            w3 = w2p.view(w2.size(0), -1, m)
            # Compute mask in each group
            absw = w3.abs()
            # indices of N smallest
            idx = torch.topk(absw, k=n, dim=2, largest=False).indices
            mask = torch.ones_like(w3)
            mask.scatter_(2, idx, 0.0)
            w3.mul_(mask)
            # restore original shape (drop pad)
            w2_new = w3.view(w2p.size(0), -1)
            if pad:
                w2_new = w2_new[:, : w2.size(1)]
            mod.weight.copy_(w2_new.view_as(w))
            # bias untouched

    return model


def _collect_activation_sparsity_resnet18(
    model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict[Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]], torch.Tensor]:
    """Collect per-channel activation sparsity (zero fraction) for ResNet18 convs.

    We measure sparsity on ReLU outputs (post-activation), not raw conv outputs.
    For torchvision ResNet BasicBlock, a single `relu` module is called twice:
    - 1st call: after (conv1+bn1)
    - 2nd call: after (conv2+bn2 + residual)

    Returns mapping: (conv, bn_or_none) -> sparsity_per_out_channel tensor on CPU.
    """

    # Local import to avoid hard dependency on torchvision internals.
    try:
        from torchvision.models.resnet import BasicBlock  # type: ignore
    except Exception:
        BasicBlock = None  # type: ignore

    if BasicBlock is None:
        raise RuntimeError("Unable to import torchvision.models.resnet.BasicBlock")

    stats_sum: Dict[Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]], torch.Tensor] = {}
    stats_cnt: Dict[Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]], torch.Tensor] = {}
    handles = []

    def add_stats(key: Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]], relu_out: torch.Tensor) -> None:
        if relu_out.dim() != 4:
            return
        # relu_out: [N,C,H,W]
        zc = (relu_out == 0).to(torch.float32).sum(dim=(0, 2, 3)).detach().cpu()
        total = float(relu_out.numel() / relu_out.size(1))
        if key not in stats_sum:
            stats_sum[key] = zc
            stats_cnt[key] = torch.full_like(zc, total)
        else:
            stats_sum[key] += zc
            stats_cnt[key] += total

    for block in model.modules():
        if not isinstance(block, BasicBlock):
            continue

        # The same ReLU module is invoked twice; we disambiguate by call order.
        call_idx = {"i": 0}

        def hook_fn(_m, _inp, out, *, blk=block, state=call_idx):
            if not torch.is_tensor(out):
                return
            state["i"] += 1
            if state["i"] == 1:
                add_stats((blk.conv1, blk.bn1), out)
            elif state["i"] == 2:
                add_stats((blk.conv2, blk.bn2), out)
            # ignore further calls if any

        handles.append(block.relu.register_forward_hook(hook_fn))

    model.eval().to(device)
    with torch.inference_mode():
        for i, (xb, _yb) in enumerate(calib_loader):
            if max_batches > 0 and i >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)

    for h in handles:
        h.remove()

    out: Dict[Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]], torch.Tensor] = {}
    for key in stats_sum:
        out[key] = (stats_sum[key] / stats_cnt[key].clamp_min(1.0)).clamp(0.0, 1.0)
    return out


def _apply_activation_channel_pruning(
    model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    ratio: float,
    max_calib_batches: int = 10,
) -> nn.Module:
    """Activation pruning (channel): prune channels with highest ReLU-output zero fraction.

    Note: this zeros weights/BN params but does not change tensor shapes.
    """

    if ratio <= 0:
        return model

    sparsity = _collect_activation_sparsity_resnet18(
        model,
        calib_loader,
        device=device,
        max_batches=max_calib_batches,
    )

    with torch.no_grad():
        for (conv, bn), s in sparsity.items():
            C = int(s.numel())
            k = int(math.floor(ratio * C))
            if k <= 0:
                continue
            idx = torch.topk(s, k=k, largest=True).indices

            conv.weight[idx] = 0
            if conv.bias is not None:
                conv.bias[idx] = 0

            if bn is not None:
                if bn.weight is not None:
                    bn.weight[idx] = 0
                if bn.bias is not None:
                    bn.bias[idx] = 0
                if hasattr(bn, "running_mean") and bn.running_mean is not None:
                    bn.running_mean[idx] = 0
                if hasattr(bn, "running_var") and bn.running_var is not None:
                    bn.running_var[idx] = 1

    return model


# -----------------
# Experiment runner
# -----------------

@dataclass
class ResultRow:
    variant: str  # baseline / method@ratio
    method: str
    ratio: float
    nm_pattern: str

    accuracy: float

    nonzero_params: int
    total_params: int
    weight_sparsity: float

    dense_weight_mb: float
    effective_nonzero_weight_mb: float

    gmacs_per_image: Optional[float]
    gflops_per_image: Optional[float]

    infer_ms_per_image: float
    infer_images_per_s: float
    effective_gflops_per_s: Optional[float]

    peak_cpu_rss_mb: Optional[float]
    peak_gpu_allocated_mb: Optional[float]


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_results(rows: List[ResultRow], out_prefix: str) -> None:
    if plt is None:
        print("[note] 未安装 matplotlib，跳过绘图。可选安装：pip install matplotlib")
        return

    # group by method
    methods = sorted({r.method for r in rows if r.method != "baseline"})
    baseline = next((r for r in rows if r.method == "baseline"), None)

    def plot_metric(title: str, y_label: str, y_get, filename: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
        ax.set_title(title)
        ax.set_xlabel("pruning ratio")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        if baseline is not None:
            ax.scatter([0.0], [y_get(baseline)], marker="*", s=120, color="black", label="baseline")

        for m in methods:
            pts = [r for r in rows if r.method == m]
            # For N:M, use ratio encoded from n/m
            pts = sorted(pts, key=lambda r: r.ratio)
            xs = [r.ratio for r in pts]
            ys = [y_get(r) for r in pts]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=m)

        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = f"{out_prefix}_{filename}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    plot_metric("CIFAR-10 accuracy vs pruning ratio", "accuracy", lambda r: r.accuracy, "acc")
    plot_metric("Weight sparsity vs pruning ratio", "sparsity", lambda r: r.weight_sparsity, "sparsity")
    plot_metric("Inference latency vs pruning ratio", "ms/image", lambda r: r.infer_ms_per_image, "latency")
    plot_metric("Throughput vs pruning ratio", "images/s", lambda r: r.infer_images_per_s, "throughput")

    # Memory
    # peak_cpu_rss_mb might be None
    if any(r.peak_cpu_rss_mb is not None for r in rows):
        plot_metric("Peak CPU RSS vs pruning ratio", "MB", lambda r: float(r.peak_cpu_rss_mb or 0.0), "peak_rss")
    if any(r.peak_gpu_allocated_mb is not None for r in rows):
        plot_metric("Peak GPU allocated vs pruning ratio", "MB", lambda r: float(r.peak_gpu_allocated_mb or 0.0), "peak_gpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="CNN pruning experiments on ResNet-18 (CIFAR-10)")
    ap.add_argument(
        "--config",
        default="",
        help="Optional YAML config file to override CLI args (requires pyyaml)",
    )
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pretrained", action="store_true", help="Start from ImageNet weights")
    ap.add_argument("--epochs", type=int, default=1, help="Quick finetune epochs on CIFAR-10")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--train-bs", type=int, default=128)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--train-limit", type=int, default=5000, help="Limit train samples for speed (0=full)")
    ap.add_argument("--calib-limit", type=int, default=1000, help="Calibration samples for activation pruning")
    ap.add_argument("--test-limit", type=int, default=0, help="Limit test samples for speed (0=full)")

    ap.add_argument("--ratios", default="0.3,0.5,0.8")
    ap.add_argument(
        "--nm-patterns",
        default="3:10,5:10,8:10",
        help="Comma-separated N:M patterns, e.g. 2:4,3:10. Default matches 30/50/80% ratios.",
    )

    ap.add_argument("--infer-bs", type=int, default=16)
    ap.add_argument("--infer-warmup", type=int, default=5)
    ap.add_argument("--infer-iters", type=int, default=20)

    ap.add_argument("--out-prefix", default="prune_resnet18")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # Optional YAML config overrides
    if args.config:
        try:
            import yaml  # type: ignore

            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise ValueError("YAML root must be a mapping")
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    print(f"[note] config key ignored (unknown arg): {k}")
        except ModuleNotFoundError:
            print("[note] 未安装 pyyaml，忽略 --config。可选安装：pip install pyyaml")
        except Exception as e:
            raise RuntimeError(f"Failed to load config {args.config}: {e}")

    device = _device_from_str(args.device)
    _seed_all(args.seed, device=device)

    ratios = _parse_float_list(args.ratios)
    nm_patterns = _parse_nm_list(args.nm_patterns)

    if thop_profile is None:
        print("[note] 未安装 thop，MACs/FLOPs 将为 NA。可选安装：pip install thop")
    if prune is None:
        print("[note] 未能导入 torch.nn.utils.prune，fine/channel pruning 将不可用")

    train_loader, calib_loader, test_loader = _make_cifar10_loaders(
        data_dir=args.data_dir,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        calib_limit=args.calib_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
    )

    # Baseline
    base = _build_resnet18(pretrained=args.pretrained).to(device)
    opt = torch.optim.SGD(base.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs):
        acc_train = _train_one_epoch(base, train_loader, device, opt)
        print(f"epoch {epoch+1}/{args.epochs} train_acc={acc_train:.3f}")

    base_acc = _eval_accuracy(base, test_loader, device)
    print(f"baseline test_acc={base_acc:.3f}")

    # Compute-only metrics
    gmacs, gflops = _compute_macs_flops_per_image(_clone_model(base), resolution=224)

    rows: List[ResultRow] = []

    def measure_variant(name: str, method: str, ratio: float, nm_pattern: str, model: nn.Module) -> None:
        _cleanup(device)
        nonzero, total, sparsity = _count_nonzero_params(model)
        dense_mb = _mb(_param_bytes_assuming_dtype(model, dtype=torch.float32))
        eff_mb = _mb(_effective_nonzero_weight_bytes(model, dtype=torch.float32))

        model = model.to(device)

        # Accuracy on CIFAR-10
        acc = _eval_accuracy(model, test_loader, device)

        # Inference latency/throughput on random input
        ms_img, ips = _measure_latency_throughput(model, device, args.infer_bs, 224, args.infer_warmup, args.infer_iters)

        eff_gflops_s = None
        if gflops is not None:
            # per-image GFLOPs * images/s
            eff_gflops_s = gflops * ips

        # Peak memory during a short measured loop
        def _infer_loop():
            _ = _measure_latency_throughput(model, device, args.infer_bs, 224, warmup=0, iters=5)

        peak_rss = _peak_rss_during(_infer_loop)
        peak_gpu = _measure_peak_gpu_allocated(_infer_loop, device)

        rows.append(
            ResultRow(
                variant=name,
                method=method,
                ratio=ratio,
                nm_pattern=nm_pattern,
                accuracy=acc,
                nonzero_params=nonzero,
                total_params=total,
                weight_sparsity=sparsity,
                dense_weight_mb=dense_mb,
                effective_nonzero_weight_mb=eff_mb,
                gmacs_per_image=gmacs,
                gflops_per_image=gflops,
                infer_ms_per_image=ms_img,
                infer_images_per_s=ips,
                effective_gflops_per_s=eff_gflops_s,
                peak_cpu_rss_mb=peak_rss,
                peak_gpu_allocated_mb=peak_gpu,
            )
        )

        print(
            f"{name}: acc={acc:.3f} sparsity={sparsity:.2%} "
            f"ms/img={ms_img:.3f} imgs/s={ips:.1f} peakRSS={peak_rss} peakGPU={peak_gpu}"
        )

        _cleanup(device)

    # Baseline measurement as ratio=0
    measure_variant("baseline", "baseline", 0.0, "", _clone_model(base))

    # 1) Fine-grained magnitude pruning
    if prune is not None:
        for r in ratios:
            m = _apply_fine_grained_magnitude_pruning(_clone_model(base), amount=r)
            measure_variant(f"fine@{r}", "fine", r, "", m)

        # 2) Channel pruning (structured masks)
        for r in ratios:
            m = _apply_channel_pruning_ln_structured(_clone_model(base), amount=r)
            measure_variant(f"channel@{r}", "channel", r, "", m)

    # 3) N:M pruning
    for (n, mval) in nm_patterns:
        r = n / mval
        mm = _apply_nm_pruning(_clone_model(base), n=n, m=mval)
        measure_variant(f"nm@{n}:{mval}", "nm", r, f"{n}:{mval}", mm)

    # 4) Activation-based channel pruning
    for r in ratios:
        mm = _apply_activation_channel_pruning(_clone_model(base), calib_loader, device=device, ratio=r)
        measure_variant(f"act_channel@{r}", "act_channel", r, "", mm)

    # Save
    out_csv = f"{args.out_prefix}_results.csv"
    _write_csv(out_csv, [asdict(x) for x in rows])
    print(f"Saved: {out_csv}")

    if not args.no_plot:
        _plot_results(rows, args.out_prefix)


if __name__ == "__main__":
    main()
