import argparse

import torch

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
    peak_rss_during,
    require_cuda,
    ResultRow,
    save_checkpoint,
    seed_all,
    train_one_epoch,
    write_csv,
    effective_nonzero_weight_bytes,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline ResNet18 on CIFAR-10 (GPU only)")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--device", default="cuda", choices=["cuda", "auto"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pretrained", action="store_true", help="Start from ImageNet weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--train-bs", type=int, default=128)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--train-limit", type=int, default=0, help="0=full train set")
    ap.add_argument("--calib-limit", type=int, default=5000)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=32, help="CIFAR-10 default is 32")
    ap.add_argument("--amp", action="store_true", help="Use AMP mixed precision training")

    ap.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Evaluate test_acc every N epochs during training (0=disable)",
    )
    ap.add_argument(
        "--early-stop-acc",
        type=float,
        default=0.0,
        help="If >0, stop training early when test_acc >= this value",
    )

    ap.add_argument("--infer-bs", type=int, default=16)
    ap.add_argument("--infer-warmup", type=int, default=10)
    ap.add_argument("--infer-iters", type=int, default=50)

    ap.add_argument("--out-prefix", default="resnet18_cifar10")
    ap.add_argument("--ckpt", default="", help="If set, load baseline weights from this checkpoint instead of training")

    args = ap.parse_args()

    device = device_from_str(args.device)
    require_cuda(device)
    seed_all(args.seed, device=device)

    torch.backends.cudnn.benchmark = True

    train_loader, calib_loader, test_loader = make_cifar10_loaders(
        data_dir=args.data_dir,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        calib_limit=args.calib_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        pin_memory=True,
        resolution=args.resolution,
    )

    model = build_resnet18(pretrained=args.pretrained, cifar10_stem=(args.resolution == 32)).to(device)

    if args.ckpt:
        load_checkpoint(args.ckpt, model, map_location="cuda")
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 60, 70], gamma=0.2)

        if args.amp:
            # torch.cuda.amp.GradScaler is deprecated in newer PyTorch.
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            scaler = None

        best_test_acc = -1.0
        best_epoch = -1
        best_ckpt = f"{args.out_prefix}_baseline_best.pt"

        for epoch in range(args.epochs):
            acc_train = train_one_epoch(
                model,
                train_loader,
                device,
                opt,
                amp=bool(args.amp),
                scaler=scaler,
            )
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            # Optional periodic evaluation
            test_acc = None
            if args.eval_every and args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs):
                test_acc = eval_accuracy(model, test_loader, device)
                if test_acc > best_test_acc:
                    best_test_acc = float(test_acc)
                    best_epoch = int(epoch + 1)
                    save_checkpoint(
                        best_ckpt,
                        model,
                        meta={
                            "seed": args.seed,
                            "pretrained": bool(args.pretrained),
                            "epoch": best_epoch,
                            "best_test_acc": best_test_acc,
                            "train_limit": int(args.train_limit),
                            "test_limit": int(args.test_limit),
                            "resolution": int(args.resolution),
                        },
                    )

            if test_acc is None:
                print(f"epoch {epoch+1}/{args.epochs} lr={lr:.5f} train_acc={acc_train:.3f}")
            else:
                print(
                    f"epoch {epoch+1}/{args.epochs} lr={lr:.5f} "
                    f"train_acc={acc_train:.3f} test_acc={float(test_acc):.3f} best={best_test_acc:.3f}@{best_epoch}"
                )

            if args.early_stop_acc and args.early_stop_acc > 0 and test_acc is not None:
                if float(test_acc) >= float(args.early_stop_acc):
                    print(f"[early-stop] reached test_acc={float(test_acc):.3f} >= {args.early_stop_acc:.3f}")
                    break

    cleanup(device)

    test_acc = eval_accuracy(model, test_loader, device)
    print(f"baseline test_acc={test_acc:.3f}")

    gmacs, gflops = compute_macs_flops_per_image(model, resolution=args.resolution)

    nonzero, total, sparsity = count_nonzero_prunable_weights(model)
    dense_mb = mb(param_bytes_assuming_dtype(model, dtype=torch.float32))
    eff_mb = mb(effective_nonzero_weight_bytes(model, dtype=torch.float32))

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

    row = ResultRow(
        variant="baseline",
        method="baseline",
        ratio=0.0,
        nm_pattern="",
        accuracy=float(test_acc),
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

    out_csv = f"{args.out_prefix}_baseline.csv"
    write_csv(out_csv, [row])
    print(f"Saved: {out_csv}")

    # Save checkpoint for pruning scripts
    out_ckpt = f"{args.out_prefix}_baseline.pt"
    save_checkpoint(
        out_ckpt,
        model,
        meta={
            "seed": args.seed,
            "pretrained": bool(args.pretrained),
            "epochs": int(args.epochs),
            "train_limit": int(args.train_limit),
            "test_limit": int(args.test_limit),
        },
    )
    print(f"Saved: {out_ckpt}")


if __name__ == "__main__":
    main()
