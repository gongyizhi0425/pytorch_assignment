import csv
import gc
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


# -----------------
# Basics
# -----------------

def mb(nbytes: int) -> float:
    return nbytes / (1024**2)


def device_from_str(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def require_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("This script is configured to run on GPU. Use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")


def seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_nm_list(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        n_str, m_str = part.split(":", 1)
        out.append((int(n_str), int(m_str)))
    return out


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# -----------------
# Data / Model
# -----------------

def make_cifar10_loaders(
    data_dir: str,
    train_bs: int,
    eval_bs: int,
    num_workers: int,
    train_limit: int,
    calib_limit: int,
    test_limit: int,
    seed: int,
    pin_memory: bool,
    resolution: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if resolution == 32:
        train_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )
        test_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )
    else:
        train_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(resolution),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )
        test_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(resolution),
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

    # Calibration subset from training set
    calib_ds = train_ds
    if calib_limit > 0 and calib_limit < len(train_ds):
        if isinstance(train_ds, Subset):
            base_idx = train_ds.indices
            idx = torch.randperm(len(base_idx), generator=g)[:calib_limit].tolist()
            calib_ds = Subset(train_ds.dataset, [base_idx[i] for i in idx])
        else:
            idx = torch.randperm(len(train_ds), generator=g)[:calib_limit].tolist()
            calib_ds = Subset(train_ds, idx)

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


def build_resnet18(num_classes: int = 10, pretrained: bool = False, cifar10_stem: bool = True) -> nn.Module:
    # For CIFAR-10, a standard and effective tweak is:
    # - conv1: 3x3, stride=1, padding=1
    # - remove maxpool
    # This preserves spatial detail for 32x32 inputs and improves accuracy.
    try:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
    except Exception:
        model = tv_models.resnet18(pretrained=pretrained)

    if cifar10_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    amp: bool = False,
    scaler: Optional[Any] = None,
    post_step_fn: Optional[Callable[[], None]] = None,
) -> float:
    model.train()
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if amp:
            if scaler is None:
                raise ValueError("amp=True requires a GradScaler")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if post_step_fn is not None:
                post_step_fn()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            if post_step_fn is not None:
                post_step_fn()
        total += yb.size(0)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
    return 0.0 if total == 0 else (correct / total)


@torch.inference_mode()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
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
# Metrics
# -----------------

def iter_prunable_weights(model: nn.Module) -> Iterable[torch.Tensor]:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield m.weight


def count_nonzero_prunable_weights(model: nn.Module) -> Tuple[int, int, float]:
    total = 0
    nonzero = 0
    for w in iter_prunable_weights(model):
        total += w.numel()
        nonzero += int(torch.count_nonzero(w).item())
    sparsity = 0.0 if total == 0 else 1.0 - (nonzero / total)
    return nonzero, total, sparsity


def param_bytes_assuming_dtype(model: nn.Module, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    total_numel = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    return int(total_numel * bytes_per)


def effective_nonzero_weight_bytes(model: nn.Module, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    nonzero = 0
    for w in iter_prunable_weights(model):
        nonzero += int(torch.count_nonzero(w).item())
    return int(nonzero * bytes_per)


def compute_macs_flops_per_image(model: nn.Module, resolution: int) -> Tuple[Optional[float], Optional[float]]:
    if thop_profile is None:
        return None, None
    # thop may attach temporary buffers like `total_ops/total_params` into modules.
    # Run it on a deepcopy to avoid polluting the caller's model/state_dict.
    import copy

    model = copy.deepcopy(model).eval().cpu()
    x = torch.randn(1, 3, resolution, resolution)
    with torch.inference_mode():
        macs, _params = thop_profile(model, inputs=(x,), verbose=False)
    gmacs = float(macs) / 1e9
    gflops = float(macs) * 2.0 / 1e9
    return gmacs, gflops


def measure_latency_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    resolution: int,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
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


def peak_rss_during(fn, poll_s: float = 0.005) -> Optional[float]:
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

    return mb(int(peak))


def measure_peak_gpu_allocated(fn, device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return mb(int(torch.cuda.max_memory_allocated()))


# -----------------
# Result row / IO
# -----------------


@dataclass
class ResultRow:
    variant: str
    method: str
    ratio: float
    nm_pattern: str

    accuracy: float

    nonzero_prunable: int
    total_prunable: int
    weight_sparsity: float

    dense_model_mb: float
    effective_nonzero_weight_mb: float

    gmacs_per_image: Optional[float]
    gflops_per_image: Optional[float]

    infer_ms_per_image: float
    infer_images_per_s: float
    effective_gflops_per_s: Optional[float]

    peak_cpu_rss_mb: Optional[float]
    peak_gpu_allocated_mb: Optional[float]


def write_csv(path: str, rows: List[ResultRow]) -> None:
    if not rows:
        return
    dict_rows = [asdict(r) for r in rows]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        w.writeheader()
        w.writerows(dict_rows)


def save_checkpoint(path: str, model: nn.Module, meta: dict) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("state_dict", payload)
    # Be tolerant to extra keys introduced by profilers (e.g. thop's total_ops).
    model.load_state_dict(state, strict=False)
    return payload.get("meta", {})
