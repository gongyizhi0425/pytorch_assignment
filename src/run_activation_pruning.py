import argparse
import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pruning_common import (
    build_resnet18,
    cleanup,
    compute_macs_flops_per_image,
    count_nonzero_prunable_weights,
    device_from_str,
    eval_accuracy,
    load_checkpoint,
    make_cifar10_loaders,
    mb,
    measure_latency_throughput,
    measure_peak_gpu_allocated,
    param_bytes_assuming_dtype,
    parse_float_list,
    peak_rss_during,
    require_cuda,
    ResultRow,
    seed_all,
    train_one_epoch,
    write_csv,
    effective_nonzero_weight_bytes,
)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def _make_grad_scaler_if_needed(amp: bool, device: torch.device):
    if not amp:
        return None
    if device.type != "cuda":
        raise RuntimeError("--amp is only supported on CUDA in this script")
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def collect_activation_sparsity_resnet18(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    max_batches: int,
) -> Dict[Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]], torch.Tensor]:
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
        call_idx = {"i": 0}

        def hook_fn(_m, _inp, out, *, blk=block, state=call_idx):
            if not torch.is_tensor(out):
                return
            state["i"] += 1
            if state["i"] == 1:
                add_stats((blk.conv1, blk.bn1), out)
            elif state["i"] == 2:
                add_stats((blk.conv2, blk.bn2), out)

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


def _slim_basicblock_intermediate(block, keep_idx: torch.Tensor) -> None:
    keep_idx = keep_idx.to(device=block.conv1.weight.device)
    k = int(keep_idx.numel())
    if k <= 0:
        return

    old_conv1: nn.Conv2d = block.conv1
    old_bn1: nn.BatchNorm2d = block.bn1
    old_conv2: nn.Conv2d = block.conv2

    new_conv1 = nn.Conv2d(
        in_channels=old_conv1.in_channels,
        out_channels=k,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        dilation=old_conv1.dilation,
        groups=old_conv1.groups,
        bias=(old_conv1.bias is not None),
        padding_mode=old_conv1.padding_mode,
    ).to(old_conv1.weight.device)

    with torch.no_grad():
        new_conv1.weight.copy_(old_conv1.weight.index_select(0, keep_idx))
        if old_conv1.bias is not None and new_conv1.bias is not None:
            new_conv1.bias.copy_(old_conv1.bias.index_select(0, keep_idx))

    new_bn1 = nn.BatchNorm2d(
        num_features=k,
        eps=old_bn1.eps,
        momentum=old_bn1.momentum,
        affine=old_bn1.affine,
        track_running_stats=old_bn1.track_running_stats,
    ).to(old_conv1.weight.device)

    with torch.no_grad():
        if old_bn1.affine:
            new_bn1.weight.copy_(old_bn1.weight.index_select(0, keep_idx))
            new_bn1.bias.copy_(old_bn1.bias.index_select(0, keep_idx))
        if old_bn1.track_running_stats:
            new_bn1.running_mean.copy_(old_bn1.running_mean.index_select(0, keep_idx))
            new_bn1.running_var.copy_(old_bn1.running_var.index_select(0, keep_idx))
            new_bn1.num_batches_tracked.copy_(old_bn1.num_batches_tracked)

    new_conv2 = nn.Conv2d(
        in_channels=k,
        out_channels=old_conv2.out_channels,
        kernel_size=old_conv2.kernel_size,
        stride=old_conv2.stride,
        padding=old_conv2.padding,
        dilation=old_conv2.dilation,
        groups=old_conv2.groups,
        bias=(old_conv2.bias is not None),
        padding_mode=old_conv2.padding_mode,
    ).to(old_conv2.weight.device)

    with torch.no_grad():
        new_conv2.weight.copy_(old_conv2.weight.index_select(1, keep_idx))
        if old_conv2.bias is not None and new_conv2.bias is not None:
            new_conv2.bias.copy_(old_conv2.bias)

    block.conv1 = new_conv1
    block.bn1 = new_bn1
    block.conv2 = new_conv2


def apply_activation_channel_slimming(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    ratio: float,
    max_calib_batches: int = 10,
) -> nn.Module:
    """True slimming based on ReLU-output zero fraction.

    Uses the conv1/bn1 ReLU stats inside each BasicBlock to decide which
    intermediate channels to keep.
    """

    if ratio <= 0:
        return model

    try:
        from torchvision.models.resnet import BasicBlock  # type: ignore
    except Exception:
        BasicBlock = None  # type: ignore
    if BasicBlock is None:
        raise RuntimeError("Unable to import torchvision.models.resnet.BasicBlock")

    stats = collect_activation_sparsity_resnet18(
        model,
        calib_loader,
        device=device,
        max_batches=max_calib_batches,
    )

    # Decide keep indices per block first, then apply slimming.
    keep_plan = []
    for block in model.modules():
        if not isinstance(block, BasicBlock):
            continue
        key = (block.conv1, block.bn1)
        if key not in stats:
            continue
        s = stats[key]  # per-channel zero fraction
        C = int(s.numel())
        k = max(1, int(round((1.0 - ratio) * C)))
        keep = torch.topk(s, k=k, largest=False).indices  # keep most-active channels
        keep, _ = torch.sort(keep)
        keep_plan.append((block, keep))

    for block, keep in keep_plan:
        _slim_basicblock_intermediate(block, keep)

    return model


def apply_activation_channel_pruning(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    ratio: float,
    max_calib_batches: int = 10,
) -> tuple[nn.Module, list[tuple[nn.Conv2d, Optional[nn.BatchNorm2d], torch.Tensor]]]:
    if ratio <= 0:
        return model, []

    sparsity = collect_activation_sparsity_resnet18(
        model,
        calib_loader,
        device=device,
        max_batches=max_calib_batches,
    )

    pruned: list[tuple[nn.Conv2d, Optional[nn.BatchNorm2d], torch.Tensor]] = []

    with torch.no_grad():
        for (conv, bn), s in sparsity.items():
            C = int(s.numel())
            k = int(math.floor(ratio * C))
            if k <= 0:
                continue
            idx = torch.topk(s, k=k, largest=True).indices
            pruned.append((conv, bn, idx))

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

    return model, pruned


def main() -> None:
    ap = argparse.ArgumentParser(description="Activation-based channel pruning for ResNet18 CIFAR-10 (GPU only)")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--device", default="cuda", choices=["cuda", "auto"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--calib-limit", type=int, default=1000)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=32)

    ap.add_argument("--finetune-epochs", type=int, default=0, help="If >0, finetune after pruning to recover accuracy")
    ap.add_argument("--finetune-lr", type=float, default=0.01)
    ap.add_argument("--finetune-bs", type=int, default=128)
    ap.add_argument("--finetune-train-limit", type=int, default=5000, help="0=full train set; default keeps finetune fast")
    ap.add_argument("--finetune-momentum", type=float, default=0.9)
    ap.add_argument("--finetune-weight-decay", type=float, default=5e-4)
    ap.add_argument("--amp", action="store_true", help="Use AMP during finetune (CUDA only)")

    ap.add_argument("--ratios", default="0.3,0.5,0.8")
    ap.add_argument("--max-calib-batches", type=int, default=10)

    ap.add_argument(
        "--impl",
        default="mask",
        choices=["mask", "slim"],
        help="actchannel implementation: mask (zero channels) or slim (physically remove intermediate channels)",
    )

    ap.add_argument("--infer-bs", type=int, default=16)
    ap.add_argument("--infer-warmup", type=int, default=10)
    ap.add_argument("--infer-iters", type=int, default=50)

    ap.add_argument("--out-prefix", default="resnet18_cifar10")
    ap.add_argument("--ckpt", default="", help="Baseline checkpoint path (from run_baseline.py)")

    args = ap.parse_args()

    # Make peak GPU measurements more comparable across variants/runs.
    # cuDNN benchmark may select different algorithms/workspaces run-to-run.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    device = device_from_str(args.device)
    require_cuda(device)
    seed_all(args.seed, device=device)

    if args.impl == "mask":
        print("[note] Using mask-style pruning (shapes unchanged): MACs/peak GPU may not decrease without model slimming.")
    else:
        print("[note] Using true channel slimming for actchannel: MACs/latency/peak GPU can decrease.")

    train_loader, calib_loader, test_loader = make_cifar10_loaders(
        data_dir=args.data_dir,
        train_bs=(args.finetune_bs if args.finetune_epochs > 0 else 1),
        eval_bs=args.eval_bs,
        num_workers=args.num_workers,
        train_limit=(args.finetune_train_limit if args.finetune_epochs > 0 else 1),
        calib_limit=args.calib_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        pin_memory=True,
        resolution=args.resolution,
    )

    if not args.ckpt:
        raise RuntimeError("Missing --ckpt. Run run_baseline.py first to produce a baseline checkpoint.")

    base = build_resnet18(pretrained=False).to(device)
    load_checkpoint(args.ckpt, base, map_location="cuda")

    base_gmacs, base_gflops = compute_macs_flops_per_image(base, resolution=args.resolution)
    ratios = parse_float_list(args.ratios)

    rows: list[ResultRow] = []

    def finetune_after_prune(model: nn.Module, post_step_fn=None) -> None:
        if args.finetune_epochs <= 0:
            return
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.finetune_lr,
            momentum=args.finetune_momentum,
            weight_decay=args.finetune_weight_decay,
        )
        scaler = _make_grad_scaler_if_needed(args.amp, device)
        for ep in range(args.finetune_epochs):
            acc_tr = train_one_epoch(
                model,
                train_loader,
                device,
                opt,
                amp=args.amp,
                scaler=scaler,
                post_step_fn=post_step_fn,
            )
            print(f"  finetune epoch {ep+1}/{args.finetune_epochs} train_acc={acc_tr:.3f}")

    def measure_variant(name: str, ratio: float, model: nn.Module, *, gmacs, gflops) -> None:
        cleanup(device)
        model = model.to(device)

        nonzero, total, sparsity = count_nonzero_prunable_weights(model)
        dense_mb = mb(param_bytes_assuming_dtype(model, dtype=torch.float32))
        eff_mb = mb(effective_nonzero_weight_bytes(model, dtype=torch.float32))

        acc = eval_accuracy(model, test_loader, device)
        ms_img, ips = measure_latency_throughput(
            model,
            device,
            args.infer_bs,
            args.resolution,
            args.infer_warmup,
            args.infer_iters,
        )

        eff_gflops_s = None
        if gflops is not None:
            eff_gflops_s = gflops * ips

        def _infer_loop():
            _ = measure_latency_throughput(model, device, args.infer_bs, args.resolution, warmup=0, iters=10)

        peak_rss = peak_rss_during(_infer_loop)
        peak_gpu = measure_peak_gpu_allocated(_infer_loop, device)

        rows.append(
            ResultRow(
                variant=name,
                method="actchannel",
                ratio=ratio,
                nm_pattern="",
                accuracy=float(acc),
                nonzero_prunable=int(nonzero),
                total_prunable=int(total),
                weight_sparsity=float(sparsity),
                dense_model_mb=float(dense_mb),
                effective_nonzero_weight_mb=float(eff_mb),
                gmacs_per_image=gmacs,
                gflops_per_image=gflops,
                infer_ms_per_image=float(ms_img),
                infer_images_per_s=float(ips),
                effective_gflops_per_s=eff_gflops_s,
                peak_cpu_rss_mb=peak_rss,
                peak_gpu_allocated_mb=peak_gpu,
            )
        )

        print(
            f"{name}: acc={acc:.3f} sparsity={sparsity:.2%} ms/img={ms_img:.3f} imgs/s={ips:.1f} peakGPU={peak_gpu}"
        )

    for r in ratios:
        if args.impl == "mask":
            m, pruned = apply_activation_channel_pruning(
                clone_model(base),
                calib_loader,
                device=device,
                ratio=r,
                max_calib_batches=args.max_calib_batches,
            )

            def _enforce_pruned_channels(pruned=pruned):
                with torch.no_grad():
                    for conv, bn, idx in pruned:
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

            finetune_after_prune(m, post_step_fn=_enforce_pruned_channels)
            _enforce_pruned_channels()
            measure_variant(f"actchannel@{r}", r, m, gmacs=base_gmacs, gflops=base_gflops)
        else:
            m = apply_activation_channel_slimming(
                clone_model(base),
                calib_loader,
                device=device,
                ratio=r,
                max_calib_batches=args.max_calib_batches,
            )
            finetune_after_prune(m)
            gmacs, gflops = compute_macs_flops_per_image(m, resolution=args.resolution)
            measure_variant(f"actchannel_slim@{r}", r, m, gmacs=gmacs, gflops=gflops)

    out_csv = f"{args.out_prefix}_activation.csv"
    write_csv(out_csv, rows)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
