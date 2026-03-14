import argparse
import copy

import torch
import torch.nn as nn

try:
    import torch.nn.utils.prune as prune
except Exception:
    prune = None

from pruning_common import (
    build_resnet18,
    cleanup,
    compute_macs_flops_per_image,
    count_nonzero_prunable_weights,
    device_from_str,
    eval_accuracy,
    make_cifar10_loaders,
    mb,
    measure_latency_throughput,
    measure_peak_gpu_allocated,
    param_bytes_assuming_dtype,
    parse_float_list,
    parse_nm_list,
    peak_rss_during,
    require_cuda,
    ResultRow,
    seed_all,
    train_one_epoch,
    write_csv,
    effective_nonzero_weight_bytes,
    load_checkpoint,
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


def apply_fine_grained_magnitude_pruning(model: nn.Module, amount: float, finalize: bool = True) -> nn.Module:
    if prune is None:
        raise RuntimeError("torch.nn.utils.prune not available")

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, name="weight", amount=amount)

    if finalize:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_orig"):
                prune.remove(m, "weight")

    return model


def apply_channel_pruning_ln_structured(model: nn.Module, amount: float, finalize: bool = True) -> nn.Module:
    if prune is None:
        raise RuntimeError("torch.nn.utils.prune not available")

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
        elif isinstance(m, nn.Linear):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)

    if finalize:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_orig"):
                prune.remove(m, "weight")

    return model


def finalize_pruning_reparam(model: nn.Module) -> None:
    if prune is None:
        return
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")


def _slim_basicblock_intermediate(block, keep_idx: torch.Tensor) -> None:
    """Physically remove intermediate channels in a torchvision ResNet BasicBlock.

    We only slim the "hidden" channels between conv1->bn1->relu and conv2 input.
    conv2 output channels (and the block output shape) stay unchanged, so residual
    connections remain valid without touching downsample.
    """

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

    # conv2: reduce input channels, keep output channels unchanged
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


def apply_channel_slimming_weight_norm(model: nn.Module, ratio: float) -> nn.Module:
    """True channel slimming based on conv1 output-channel weight norms.

    This slims each BasicBlock's intermediate channels (conv1/bn1/conv2 input).
    """

    if ratio <= 0:
        return model

    try:
        from torchvision.models.resnet import BasicBlock  # type: ignore
    except Exception:
        BasicBlock = None  # type: ignore

    if BasicBlock is None:
        raise RuntimeError("Unable to import torchvision.models.resnet.BasicBlock")

    for block in model.modules():
        if not isinstance(block, BasicBlock):
            continue
        w = block.conv1.weight.detach()
        C = int(w.size(0))
        k = max(1, int(round((1.0 - ratio) * C)))
        # L2 norm per output channel
        imp = torch.linalg.vector_norm(w.flatten(1), ord=2, dim=1)
        keep = torch.topk(imp, k=k, largest=True).indices
        keep, _ = torch.sort(keep)
        _slim_basicblock_intermediate(block, keep)

    return model


def apply_nm_pruning(model: nn.Module, n: int, m: int) -> tuple[nn.Module, list[tuple[torch.Tensor, torch.Tensor]]]:
    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern {n}:{m}")

    masks: list[tuple[torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for mod in model.modules():
            if not isinstance(mod, (nn.Conv2d, nn.Linear)):
                continue
            w = mod.weight
            w2 = w.view(w.size(0), -1)
            pad = (m - (w2.size(1) % m)) % m
            if pad:
                w2p = torch.cat(
                    [w2, torch.zeros(w2.size(0), pad, device=w2.device, dtype=w2.dtype)],
                    dim=1,
                )
            else:
                w2p = w2
            w3 = w2p.view(w2.size(0), -1, m)
            absw = w3.abs()
            idx = torch.topk(absw, k=n, dim=2, largest=False).indices
            mask = torch.ones_like(w3)
            mask.scatter_(2, idx, 0.0)
            w3.mul_(mask)
            w2_new = w3.view(w2p.size(0), -1)
            if pad:
                w2_new = w2_new[:, : w2.size(1)]
            mod.weight.copy_(w2_new.view_as(w))

            # Save an enforcement mask to keep sparsity fixed during finetune
            w2_mask = mask.view(w2p.size(0), -1)
            if pad:
                w2_mask = w2_mask[:, : w2.size(1)]
            masks.append((mod.weight, w2_mask.view_as(w)))

    return model, masks


def main() -> None:
    ap = argparse.ArgumentParser(description="Weight pruning (fine/channel/N:M) for ResNet18 CIFAR-10 (GPU only)")
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
    ap.add_argument("--nm-patterns", default="3:10,5:10,8:10")

    ap.add_argument(
        "--channel-impl",
        default="mask",
        choices=["mask", "slim"],
        help="channel pruning implementation: mask (zero channels) or slim (physically remove intermediate channels)",
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

    if args.channel_impl == "mask":
        print("[note] Using mask-style pruning (shapes unchanged): MACs/peak GPU may not decrease without model slimming.")
    else:
        print("[note] Using true channel slimming for channel pruning: MACs/latency/peak GPU can decrease.")

    train_loader, _calib_loader, test_loader = make_cifar10_loaders(
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

    # Compute-only metrics
    # - mask pruning: same shapes => same MACs, compute once
    # - slimming: shapes change => compute per-variant
    base_gmacs, base_gflops = compute_macs_flops_per_image(base, resolution=args.resolution)

    ratios = parse_float_list(args.ratios)
    nm_patterns = parse_nm_list(args.nm_patterns)

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

    def measure_variant(
        name: str,
        method: str,
        ratio: float,
        nm_pattern: str,
        model: nn.Module,
        *,
        gmacs: float | None,
        gflops: float | None,
    ) -> None:
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
                method=method,
                ratio=ratio,
                nm_pattern=nm_pattern,
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

    if prune is None:
        print("[note] torch.nn.utils.prune not available; fine/channel pruning disabled")
    else:
        for r in ratios:
            m = apply_fine_grained_magnitude_pruning(clone_model(base), amount=r, finalize=(args.finetune_epochs <= 0))
            finetune_after_prune(m)
            if args.finetune_epochs > 0:
                finalize_pruning_reparam(m)
            measure_variant(f"fine@{r}", "fine", r, "", m, gmacs=base_gmacs, gflops=base_gflops)

        for r in ratios:
            if args.channel_impl == "mask":
                m = apply_channel_pruning_ln_structured(clone_model(base), amount=r, finalize=(args.finetune_epochs <= 0))
                finetune_after_prune(m)
                if args.finetune_epochs > 0:
                    finalize_pruning_reparam(m)
                measure_variant(f"channel@{r}", "channel", r, "", m, gmacs=base_gmacs, gflops=base_gflops)
            else:
                m = apply_channel_slimming_weight_norm(clone_model(base), ratio=r)
                finetune_after_prune(m)
                gmacs, gflops = compute_macs_flops_per_image(m, resolution=args.resolution)
                measure_variant(f"channel_slim@{r}", "channel", r, "", m, gmacs=gmacs, gflops=gflops)

    for (n, mval) in nm_patterns:
        r = n / mval
        m, masks = apply_nm_pruning(clone_model(base), n=n, m=mval)

        def _enforce_nm_masks(masks=masks):
            with torch.no_grad():
                for w, mask in masks:
                    w.mul_(mask)

        finetune_after_prune(m, post_step_fn=_enforce_nm_masks)
        # Ensure final weights respect the mask
        _enforce_nm_masks()

        measure_variant(f"nm@{n}:{mval}", "nm", r, f"{n}:{mval}", m, gmacs=base_gmacs, gflops=base_gflops)

    out_csv = f"{args.out_prefix}_weight.csv"
    write_csv(out_csv, rows)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
