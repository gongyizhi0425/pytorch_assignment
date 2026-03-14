from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_ppl_vs_sparsity(csv_path: str, out_png: Optional[str] = None, title: str = "PPL vs Sparsity") -> str:
    df = pd.read_csv(csv_path)
    if "achieved_sparsity" not in df.columns:
        raise ValueError("CSV must contain achieved_sparsity")

    if out_png is None:
        out_png = os.path.splitext(csv_path)[0] + "_ppl_vs_sparsity.png"

    plt.figure(figsize=(7, 5))
    for method, g in df.groupby("method"):
        g = g.sort_values("achieved_sparsity")
        plt.plot(g["achieved_sparsity"], g["ppl"], marker="o", label=method)

    plt.xlabel("Achieved sparsity")
    plt.ylabel("Perplexity (lower is better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
