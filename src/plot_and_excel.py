import argparse
from pathlib import Path

import pandas as pd


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge pruning CSVs -> Excel (multi-sheet) + plots")
    ap.add_argument("--out-prefix", default="resnet18_cifar10", help="Output prefix for xlsx/png (default: resnet18_cifar10)")
    ap.add_argument("--excel", default="", help="Output xlsx path (default: <out-prefix>_results.xlsx)")
    ap.add_argument("--no-plot", action="store_true")

    # Input control
    ap.add_argument(
        "--baseline-prefix",
        default="",
        help="Prefix for baseline CSV (reads <prefix>_baseline.csv). Default: use --out-prefix",
    )
    ap.add_argument(
        "--weight-prefix",
        default="",
        help="Prefix for weight CSV (reads <prefix>_weight.csv). Default: use --out-prefix",
    )
    ap.add_argument(
        "--activation-prefix",
        default="",
        help="Prefix for activation CSV (reads <prefix>_activation.csv). Default: use --out-prefix",
    )
    ap.add_argument("--baseline-csv", default="", help="Explicit path to baseline CSV (overrides --baseline-prefix)")
    ap.add_argument("--weight-csv", default="", help="Explicit path to weight CSV (overrides --weight-prefix)")
    ap.add_argument("--activation-csv", default="", help="Explicit path to activation CSV (overrides --activation-prefix)")
    args = ap.parse_args()

    out_prefix = args.out_prefix

    baseline_prefix = args.baseline_prefix or out_prefix
    weight_prefix = args.weight_prefix or out_prefix
    activation_prefix = args.activation_prefix or out_prefix

    baseline_csv = Path(args.baseline_csv) if args.baseline_csv else Path(f"{baseline_prefix}_baseline.csv")
    weight_csv = Path(args.weight_csv) if args.weight_csv else Path(f"{weight_prefix}_weight.csv")
    act_csv = Path(args.activation_csv) if args.activation_csv else Path(f"{activation_prefix}_activation.csv")

    df_base = _read_csv_if_exists(baseline_csv)
    df_weight = _read_csv_if_exists(weight_csv)
    df_act = _read_csv_if_exists(act_csv)

    if df_base.empty and df_weight.empty and df_act.empty:
        raise RuntimeError("No input CSVs found. Run baseline/weight/activation scripts first.")

    # Normalize method name for activation
    if not df_act.empty and "method" in df_act.columns:
        df_act["method"] = "actchannel"

    df_all = pd.concat([df_base, df_weight, df_act], ignore_index=True)

    # Sheet mapping required by the assignment
    sheets = {
        "baseline": df_all[df_all["method"] == "baseline"],
        "fine": df_all[df_all["method"] == "fine"],
        "channel": df_all[df_all["method"] == "channel"],
        "nm": df_all[df_all["method"] == "nm"],
        "actchannel": df_all[df_all["method"] == "actchannel"],
    }

    excel_path = Path(args.excel) if args.excel else Path(f"{out_prefix}_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df.empty:
                # Still create an empty sheet for consistency
                pd.DataFrame({"note": ["no rows"]}).to_excel(writer, sheet_name=name, index=False)
            else:
                df.to_excel(writer, sheet_name=name, index=False)

    print(f"Saved: {excel_path}")

    if args.no_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[note] matplotlib not available; skip plots")
        return

    baseline_row = None
    if not sheets["baseline"].empty:
        baseline_row = sheets["baseline"].iloc[0]

    def plot_metric(title: str, y_label: str, col: str, filename: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
        ax.set_title(title)
        ax.set_xlabel("pruning ratio")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        if baseline_row is not None and col in baseline_row:
            ax.scatter([0.0], [baseline_row[col]], marker="*", s=120, color="black", label="baseline")

        for method in ["fine", "channel", "nm", "actchannel"]:
            dfm = sheets.get(method, pd.DataFrame())
            if dfm is None or dfm.empty or col not in dfm.columns:
                continue
            dfm = dfm.sort_values("ratio")
            ax.plot(dfm["ratio"], dfm[col], marker="o", linewidth=1.5, label=method)

        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = Path(f"{out_prefix}_{filename}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    plot_metric("CIFAR-10 accuracy vs pruning ratio", "accuracy", "accuracy", "acc")
    plot_metric("Weight sparsity vs pruning ratio", "sparsity", "weight_sparsity", "sparsity")
    plot_metric("Inference latency vs pruning ratio", "ms/image", "infer_ms_per_image", "latency")
    plot_metric("Throughput vs pruning ratio", "images/s", "infer_images_per_s", "throughput")

    # Static/structural metrics (useful for both mask pruning and true slimming)
    if "dense_model_mb" in df_all.columns and df_all["dense_model_mb"].notna().any():
        plot_metric("Model size vs pruning ratio", "MB", "dense_model_mb", "model_mb")

    if "effective_nonzero_weight_mb" in df_all.columns and df_all["effective_nonzero_weight_mb"].notna().any():
        plot_metric("Effective nonzero weight size vs pruning ratio", "MB", "effective_nonzero_weight_mb", "nonzero_weight_mb")

    if "gmacs_per_image" in df_all.columns and df_all["gmacs_per_image"].notna().any():
        plot_metric("Compute (GMACs/image) vs pruning ratio", "GMACs/image", "gmacs_per_image", "gmacs")

    # GPU memory is the focus here
    if "peak_gpu_allocated_mb" in df_all.columns and df_all["peak_gpu_allocated_mb"].notna().any():
        plot_metric("Peak GPU allocated vs pruning ratio", "MB", "peak_gpu_allocated_mb", "peak_gpu")


if __name__ == "__main__":
    main()
