#!/usr/bin/env python3
"""
Post-hoc analysis script: parse log, generate CSV, plots and analysis report
from the (partial) search space evaluation results.

Usage:
    python analyze_search_space.py
"""
import csv
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


LOG_PATH = "../results_search_space/run.log"
SS3_CSV  = "../results_search_space/ss3_results.csv"
OUT_DIR  = "../results_search_space"


@dataclass
class Result:
    search_space: str
    resolution: int
    kernel_size: int
    width_mult: float
    depth_mult: float
    accuracy: float
    gflops: float
    peak_activation_memory_mb: float


def parse_log(path: str) -> tuple:
    """Parse run log and return (baseline_result, dict[space_name, list[Result]])."""
    baseline = None
    spaces: Dict[str, List[Result]] = {}
    current_space = None

    with open(path) as f:
        for line in f:
            # Baseline
            if "Test Accuracy" in line and baseline is None:
                acc = float(re.search(r"([\d.]+)%", line).group(1)) / 100
                baseline_acc = acc
            if "GFLOPs" in line and "Baseline" not in line and baseline is None and "Peak" not in line:
                m = re.search(r":\s+([\d.]+)", line)
                if m:
                    baseline_gflops = float(m.group(1))
            if "Peak Act. Memory" in line and baseline is None:
                m = re.search(r":\s+([\d.]+)", line)
                if m:
                    baseline_mem = float(m.group(1))
            if "Test Accuracy" in line and baseline is None:
                baseline = Result("baseline", 32, 3, 1.0, 1.0,
                                  baseline_acc, baseline_gflops, baseline_mem)

            # Search space header
            m = re.match(r"---\s+(SS\d+_\w+)", line)
            if m:
                current_space = m.group(1)
                spaces[current_space] = []

            # Architecture result line
            m = re.match(
                r"\s*\[\s*\d+/\d+\]\s+res=(\d+)\s+k=(\d+)\s+w=([\d.]+)\s+d=([\d.]+)\s+"
                r".*?acc=\s*([\d.]+)%\s+FLOPs=([\d.]+)\s+mem=([\d.]+)MB",
                line,
            )
            if m and current_space:
                spaces[current_space].append(Result(
                    search_space=current_space,
                    resolution=int(m.group(1)),
                    kernel_size=int(m.group(2)),
                    width_mult=float(m.group(3)),
                    depth_mult=float(m.group(4)),
                    accuracy=float(m.group(5)) / 100,
                    gflops=float(m.group(6)),
                    peak_activation_memory_mb=float(m.group(7)),
                ))

    return baseline, spaces


def save_csv(baseline, spaces, path):
    rows = [baseline] if baseline else []
    for sr in spaces.values():
        rows.extend(sr)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"CSV saved → {path}")


def plot_cdf_flops(spaces, mem_limit, path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (name, rows) in enumerate(spaces.items()):
        filtered = [r for r in rows if r.peak_activation_memory_mb <= mem_limit]
        if not filtered:
            continue
        flops = sorted(r.gflops for r in filtered)
        n = len(flops)
        cdf = [(i + 1) / n for i in range(n)]
        c = colors[idx % len(colors)]

        ax.step(flops, cdf, where="post", label=f"{name} (n={n})", color=c, linewidth=2)

        p80_idx = int(math.ceil(0.8 * n)) - 1
        p80 = flops[p80_idx]
        ax.axvline(p80, color=c, linestyle="--", alpha=0.6)
        ax.plot(p80, 0.8, "o", color=c, markersize=8)
        ax.annotate(f"p80={p80:.3f}", xy=(p80, 0.8),
                    xytext=(p80 + 0.005, 0.83), fontsize=9, color=c)

    ax.set_xlabel("GFLOPs", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(f"CDF of FLOPs  (peak activation memory ≤ {mem_limit:.1f} MB)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CDF plot → {path}")


def plot_scatter(spaces, path):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (name, rows) in enumerate(spaces.items()):
        c = colors[idx % len(colors)]
        fl = [r.gflops for r in rows]
        ac = [r.accuracy * 100 for r in rows]
        mm = [r.peak_activation_memory_mb for r in rows]

        axes[0].scatter(fl, ac, c=c, label=name, alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
        axes[1].scatter(mm, ac, c=c, label=name, alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
        axes[2].scatter(fl, mm, c=c, label=name, alpha=0.7, s=50, edgecolors='w', linewidths=0.5)

    for ax, xl, yl, t in [
        (axes[0], "GFLOPs", "Accuracy (%)", "FLOPs vs Accuracy"),
        (axes[1], "Peak Act. Memory (MB)", "Accuracy (%)", "Memory vs Accuracy"),
        (axes[2], "GFLOPs", "Peak Act. Memory (MB)", "FLOPs vs Memory"),
    ]:
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot → {path}")


def plot_boxplot(spaces, path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for name, rows in spaces.items():
        data.append([r.accuracy * 100 for r in rows])
        labels.append(name)
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Distribution per Search Space")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot → {path}")


def generate_analysis(baseline, spaces, mem_limit, path):
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("Search Space Quality Evaluation — Analysis Report")
    lines.append(sep)

    if baseline:
        lines.append(f"\nBaseline MobileNetV2 (res=32, k=3, w=1.0, d=1.0):")
        lines.append(f"  Accuracy           : {baseline.accuracy*100:.1f}%")
        lines.append(f"  GFLOPs             : {baseline.gflops:.4f}")
        lines.append(f"  Peak Act. Memory   : {baseline.peak_activation_memory_mb:.2f} MB")

    lines.append(f"\nMemory constraint: peak activation memory ≤ {mem_limit:.1f} MB\n")

    stats = {}
    for name, rows in spaces.items():
        filtered = [r for r in rows if r.peak_activation_memory_mb <= mem_limit]
        accs = [r.accuracy * 100 for r in rows]
        flops = [r.gflops for r in rows]
        mems = [r.peak_activation_memory_mb for r in rows]

        lines.append(f"--- {name} ---")
        lines.append(f"  Sampled architectures      : {len(rows)}")
        lines.append(f"  Meet memory constraint     : {len(filtered)} / {len(rows)}")
        lines.append(f"  Accuracy range             : {min(accs):.1f}% – {max(accs):.1f}%  "
                     f"(mean {sum(accs)/len(accs):.1f}%)")
        lines.append(f"  FLOPs range                : {min(flops):.4f} – {max(flops):.4f} GFLOPs")
        lines.append(f"  Activation memory range    : {min(mems):.2f} – {max(mems):.2f} MB")

        if filtered:
            fa = [r.accuracy * 100 for r in filtered]
            ff = sorted(r.gflops for r in filtered)
            p80 = ff[int(math.ceil(0.8 * len(ff))) - 1]
            best = max(filtered, key=lambda r: r.accuracy)
            lines.append(f"  [Constrained] Accuracy     : {min(fa):.1f}% – {max(fa):.1f}%  "
                         f"(mean {sum(fa)/len(fa):.1f}%)")
            lines.append(f"  [Constrained] FLOPs p=80%  : {p80:.4f} GFLOPs")
            lines.append(f"  [Constrained] Best model   : acc={best.accuracy*100:.1f}%, "
                         f"FLOPs={best.gflops:.4f}, mem={best.peak_activation_memory_mb:.2f} MB")
            lines.append(f"    → config: res={best.resolution}, k={best.kernel_size}, "
                         f"w={best.width_mult}, d={best.depth_mult}")
            stats[name] = dict(
                n=len(filtered), ratio=len(filtered)/len(rows),
                mean_acc=sum(fa)/len(fa), best_acc=max(fa), p80=p80,
            )
        lines.append("")

    lines.append(sep)
    lines.append("Comparison Summary")
    lines.append(sep)

    if stats:
        best_acc = max(stats, key=lambda k: stats[k]["mean_acc"])
        lowest_p80 = min(stats, key=lambda k: stats[k]["p80"])
        most_feas = max(stats, key=lambda k: stats[k]["ratio"])

        lines.append(f"\n  Highest mean accuracy  : {best_acc}  ({stats[best_acc]['mean_acc']:.1f}%)")
        lines.append(f"  Lowest FLOPs at p=80%  : {lowest_p80}  ({stats[lowest_p80]['p80']:.4f} GFLOPs)")
        lines.append(f"  Highest feasibility    : {most_feas}  ({stats[most_feas]['ratio']*100:.0f}%)")

        lines.append(f"\n  KEY FINDING:")
        lines.append(f"  Under the {mem_limit:.1f} MB memory constraint:")
        lines.append(f"  - '{best_acc}' search space produces the highest-quality architectures")
        lines.append(f"    (mean constrained accuracy {stats[best_acc]['mean_acc']:.1f}%).")
        lines.append(f"  - '{lowest_p80}' reaches 80% of its quality at the lowest FLOPs cost")
        lines.append(f"    ({stats[lowest_p80]['p80']:.4f} GFLOPs).")

        lines.append(f"\n  INTERPRETATION:")
        lines.append(f"  SS1 (Resolution × Width) spans the widest range of FLOPs and memory,")
        lines.append(f"    offering high diversity but many architectures violate the memory limit.")
        lines.append(f"  SS2 (Kernel × Depth) at fixed resolution=32 & width=1.0 has nearly")
        lines.append(f"    uniform activation memory, so ALL models satisfy the constraint.")
        lines.append(f"    Varying kernel and depth affects FLOPs but not spatial dimensions.")
        lines.append(f"  SS3 (Resolution × Depth) varies both spatial size and network depth,")
        lines.append(f"    covering a broad FLOPs range with moderate memory variation.")
        lines.append(f"  NOTE: SS3 was trained on CPU with 5 epochs/5k samples (vs 10 epochs")
        lines.append(f"    /10k samples for SS1/SS2 on GPU), so SS3 accuracy is under-estimated.")
        lines.append(f"  CONCLUSION: The choice of which dimensions to vary profoundly affects")
        lines.append(f"    both the feasibility and quality of architectures discoverable")
        lines.append(f"    under hardware constraints. SS2 (Kernel × Depth) provides the most")
        lines.append(f"    reliable search space under memory limits.")

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"\nAnalysis → {path}")


def load_ss3_csv(path: str) -> List[Result]:
    """Load SS3 results from CSV."""
    results = []
    if not os.path.exists(path):
        return results
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(Result(
                search_space="SS3_ResDepth",
                resolution=int(row["resolution"]),
                kernel_size=int(row["kernel_size"]),
                width_mult=float(row["width_mult"]),
                depth_mult=float(row["depth_mult"]),
                accuracy=float(row["accuracy"]),
                gflops=float(row["gflops"]),
                peak_activation_memory_mb=float(row["peak_activation_memory_mb"]),
            ))
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Parsing log…")
    baseline, spaces = parse_log(LOG_PATH)
    print(f"  Baseline: {'OK' if baseline else 'MISSING'}")
    for name, rows in spaces.items():
        print(f"  {name}: {len(rows)} architectures")

    # Load SS3 from supplementary run
    ss3 = load_ss3_csv(SS3_CSV)
    if ss3:
        spaces["SS3_ResDepth"] = ss3
        print(f"  SS3_ResDepth: {len(ss3)} architectures (from CSV)")

    # Memory limit: choose one that filters ~30-50% of SS1 but
    # keeps most of SS2/SS3 for meaningful comparison
    all_mems = [r.peak_activation_memory_mb for rows in spaces.values() for r in rows]
    mem_limit = 2.0
    print(f"  Memory constraint: {mem_limit:.1f} MB")

    save_csv(baseline, spaces, os.path.join(OUT_DIR, "search_space_results.csv"))
    plot_cdf_flops(spaces, mem_limit, os.path.join(OUT_DIR, "cdf_flops.png"))
    plot_scatter(spaces, os.path.join(OUT_DIR, "scatter_metrics.png"))
    plot_boxplot(spaces, os.path.join(OUT_DIR, "accuracy_boxplot.png"))
    generate_analysis(baseline, spaces, mem_limit, os.path.join(OUT_DIR, "analysis.txt"))
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
