#!/usr/bin/env python3
"""Re-run only the error analysis step (Step 5) using saved AWQ models.
No re-quantisation needed — loads from results_awq/awq_g{best}/.
Uses safetensors to directly read weight tensors (no model instantiation needed
for the AWQ checkpoint, which avoids AutoAWQ / quantization-config pitfalls).
"""
import sys, os, yaml, torch
import pandas as pd
import numpy as np
from safetensors.torch import load_file as st_load_file

# add src to path so we can import from run_awq
sys.path.insert(0, os.path.dirname(__file__))
from run_awq import (
    _naive_int4_group_quant_dequant,
    compute_per_channel_mae,
    compute_awq_equiv_error,
    plot_quant_error,
    plot_quant_error_hist,
    write_analysis,
)
from transformers import AutoModelForCausalLM


def main():
    with open("configs/awq.yaml") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output"]["dir"]
    model_name = cfg["model"]["name"]
    target_name = cfg["error_analysis"]["target_layer"]
    naive_gs = cfg["error_analysis"]["naive_group_size"]

    # Load results CSV to find best AWQ group size
    df = pd.read_csv(os.path.join(out_dir, "awq_results.csv"))
    awq_rows = df[df["method"] == "awq"]
    best_row = awq_rows.loc[awq_rows["ppl"].idxmin()]
    best_gs = int(best_row["group_size"])
    print(f"Best AWQ group_size = {best_gs}  (PPL = {best_row['ppl']:.4f})")

    # 1) Load original FP16 model → get original weight
    print("\n[1/3] Loading FP16 model to get original weight …")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    target_mod = dict(model_fp16.named_modules())[target_name]
    w_orig = target_mod.weight.data.clone()
    print(f"  w_orig shape: {w_orig.shape}")
    del model_fp16

    # 2) Load AWQ weight directly from safetensors (no model instantiation)
    print("[2/3] Loading AWQ weight from safetensors …")
    awq_path = os.path.join(out_dir, f"awq_g{best_gs}")
    st_path = os.path.join(awq_path, "model.safetensors")
    weight_key = target_name + ".weight"
    state = st_load_file(st_path)
    w_awq = state[weight_key].clone()
    print(f"  w_awq shape: {w_awq.shape}")
    del state

    # 3) Compute equiv error
    print("[3/3] Computing per-channel MAE …")
    mae_naive, mae_awq = compute_awq_equiv_error(w_orig, w_awq, group_size=best_gs)

    print(f"\n  Before AWQ (Naïve W4 g={naive_gs}):")
    print(f"    mean MAE = {mae_naive.mean():.6f}")
    print(f"    max  MAE = {mae_naive.max():.6f}")
    print(f"    std  MAE = {mae_naive.std():.6f}")
    print(f"\n  After AWQ (equiv. error, g={best_gs}):")
    print(f"    mean MAE = {mae_awq.mean():.6f}")
    print(f"    max  MAE = {mae_awq.max():.6f}")
    print(f"    std  MAE = {mae_awq.std():.6f}")
    reduction = (1 - mae_awq.mean() / mae_naive.mean()) * 100
    print(f"\n  Error reduction = {reduction:.1f}%")

    # 4) Plot
    print("\nGenerating plots …")
    plot_quant_error(mae_naive, mae_awq, target_name, out_dir)
    plot_quant_error_hist(mae_naive, mae_awq, target_name, out_dir)
    write_analysis(df, mae_naive, mae_awq, out_dir, target_name=target_name)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
