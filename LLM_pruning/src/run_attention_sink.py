from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_sink.attn_curve import _choose_layer_indices, attention_to_position_curve_llama, find_sink_positions
from attention_sink.pg19 import Pg19Config, iter_pg19_excerpts, load_pg19_dataset
from attention_sink.ppl import ppl_for_sequence


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isabs(path) and not os.path.exists(path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(repo_root, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _prepare(model_name: str, device: torch.device, dtype: torch.dtype, cache_dir: str | None, local_only: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir, local_files_only=local_only)
    # Avoid warnings when encoding long PG-19 books (we will slice to target lengths ourselves).
    tok.model_max_length = 10**9
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_only,
    )
    model.to(device)
    model.eval()
    return model, tok


def _plot_curve(curve: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(curve)), curve)
    plt.xlabel("Key position")
    plt.ylabel("Avg attention mass")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_prefix_ablation(df_sum: pd.DataFrame, out_png: str, title: str):
    """Plot ΔPPL vs prefix_len for each prefix type, one subplot per sequence length."""
    prefix_rows = df_sum[df_sum["intervention"].str.startswith("prefix:")].copy()
    if prefix_rows.empty:
        return
    prefix_rows["prefix_name"] = prefix_rows["intervention"].str.extract(r"prefix:(.+):plen\d+")[0]
    prefix_rows["prefix_len"] = prefix_rows["intervention"].str.extract(r":plen(\d+)$")[0].astype(int)

    lengths = sorted(prefix_rows["length"].unique())
    n_cols = len(lengths)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, seq_L in zip(axes, lengths):
        sub = prefix_rows[prefix_rows["length"] == seq_L]
        for pname in sorted(sub["prefix_name"].unique()):
            psub = sub[sub["prefix_name"] == pname].sort_values("prefix_len")
            ax.plot(psub["prefix_len"], psub["delta_vs_baseline"], marker="o", label=pname)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("Prefix length (tokens)")
        ax.set_title(f"L={seq_L}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("ΔPPL vs baseline")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_delta_ppl(df: pd.DataFrame, out_png: str, title: str):
    if df.empty:
        return
    # Expect columns: length, intervention, mean_ppl, delta_vs_baseline
    lengths = sorted(df["length"].unique().tolist())
    interventions = [x for x in df["intervention"].unique().tolist() if x != "baseline"]
    if not interventions:
        return

    plt.figure(figsize=(10, 4))
    x = np.arange(len(lengths))
    width = 0.85 / max(1, len(interventions))
    for i, name in enumerate(interventions):
        sub = df[df["intervention"] == name].set_index("length").reindex(lengths)
        y = sub["delta_vs_baseline"].to_numpy(dtype=np.float64)
        plt.bar(x + i * width, y, width=width, label=name)

    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x + (len(interventions) - 1) * width / 2.0, [str(L) for L in lengths])
    plt.xlabel("Sequence length")
    plt.ylabel("ΔPPL (intervention - baseline)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _fixed_len_prefix_ids(prefix_ids: list[int], prefix_len: int) -> list[int]:
    """Return a fixed-length prefix token-id list.

    If prefix_ids is shorter than prefix_len, repeat it cyclically.
    If longer, truncate.
    """
    if prefix_len <= 0:
        return []
    if not prefix_ids:
        return []
    if len(prefix_ids) >= prefix_len:
        return list(prefix_ids[:prefix_len])
    out: list[int] = []
    i = 0
    while len(out) < prefix_len:
        out.append(int(prefix_ids[i % len(prefix_ids)]))
        i += 1
    return out


def _prepend_prefix(excerpt_ids: list[int], prefix_ids: list[int]) -> list[int]:
    if not prefix_ids:
        return list(excerpt_ids)
    return list(prefix_ids) + list(excerpt_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_cfg = cfg["model"]
    model_name = str(model_cfg["name"])
    device = _pick_device(str(model_cfg.get("device", "auto")))
    dtype = _pick_dtype(str(model_cfg.get("dtype", "auto")), device)

    cache_dir = args.cache_dir or model_cfg.get("cache_dir")
    local_only = bool(args.local_files_only or model_cfg.get("local_files_only", False))

    out_cfg = cfg.get("output", {})
    out_dir = str(out_cfg.get("out_dir", "results_attention"))
    if not os.path.isabs(out_dir):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(repo_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_name = str(out_cfg.get("run_name", "attn_sink"))

    model, tok = _prepare(model_name, device, dtype, cache_dir, local_only)

    pg19_cfg = Pg19Config(**cfg.get("pg19", {}))

    # Key code: download/stream PG-19 dataset
    # ds = load_dataset("pg19", split="test", streaming=True)
    ds = load_pg19_dataset(split=pg19_cfg.split, streaming=bool(pg19_cfg.streaming))

    analysis_cfg = cfg.get("analysis", {})
    lengths: List[int] = list(analysis_cfg.get("lengths", [256, 512, 1024]))
    num_layers = int(analysis_cfg.get("num_layers", 6))
    num_heads = int(analysis_cfg.get("num_heads", 8))
    windows_per_book = int(analysis_cfg.get("windows_per_book", 1))
    eval_last_n = int(analysis_cfg.get("eval_last_n", 128))

    # choose layers/heads
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0) or getattr(model.config, "n_layer", 0) or 0)
    layer_indices = _choose_layer_indices(num_hidden_layers, num_layers)
    head_indices = list(range(num_heads))

    # interventions
    interventions = cfg.get("interventions", {})
    enabled_interventions = bool(interventions.get("enabled", True))
    raw_plen = interventions.get("prefix_len_tokens", [10])
    prefix_len_list: list[int] = [int(x) for x in raw_plen] if isinstance(raw_plen, list) else [int(raw_plen)]
    prefixes = interventions.get("prefixes", []) if enabled_interventions else [{"name": "baseline", "text": ""}]
    sink_replacements = interventions.get("sink_replacements", []) if enabled_interventions else []

    rows_curve = []
    rows_ppl = []
    rows_ppl_summary = []

    for L in lengths:
        excerpt_iter = iter_pg19_excerpts(
            tokenizer=tok,
            ds=ds,
            num_excerpts=int(pg19_cfg.num_excerpts),
            min_book_tokens=int(pg19_cfg.min_book_tokens),
            excerpt_len=int(L),
            windows_per_book=windows_per_book,
            seed=seed + L,
        )

        excerpts: list[list[int]] = []

        acc_curve = np.zeros((L,), dtype=np.float64)
        n_ok = 0

        for excerpt_ids in excerpt_iter:
            excerpts.append(excerpt_ids)
            input_ids = torch.tensor(excerpt_ids, dtype=torch.long).unsqueeze(0).to(device)
            attn_mask = torch.ones_like(input_ids)
            try:
                curve_t = attention_to_position_curve_llama(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    layer_indices=layer_indices,
                    head_indices=head_indices,
                )
                curve = curve_t.cpu().numpy()
                if curve.shape[0] == L:
                    acc_curve += curve
                    n_ok += 1
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # skip this excerpt
                continue

        if n_ok == 0 or not excerpts:
            continue

        mean_curve = (acc_curve / float(n_ok)).astype(np.float32)
        sinks = find_sink_positions(mean_curve, topk=5)

        # PPL interventions: compute loss on a fixed suffix (last eval_last_n tokens of the ORIGINAL excerpt)
        # so changes are measurable/comparable.
        loss_tokens = min(int(eval_last_n), int(L) - 1)
        base_loss_start = int(L - loss_tokens)

        # Pre-tokenize prefixes once
        prefix_name_to_ids: dict[str, list[int]] = {}
        for p in prefixes:
            name = str(p.get("name", ""))
            prefix_text = str(p.get("text", ""))
            if prefix_text:
                prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
            else:
                prefix_ids = []
            prefix_name_to_ids[name] = prefix_ids

        # Token ids used for sink replacements
        sink_repl_specs: list[dict[str, Any]] = []
        for r in sink_replacements:
            r_name = str(r.get("name", ""))
            r_text = str(r.get("text", ""))
            r_token_id = r.get("token_id", None)
            if r_token_id is None:
                if r_text:
                    enc = tok(r_text, add_special_tokens=False)["input_ids"]
                    if not enc:
                        continue
                    r_token_id = int(enc[0])
                else:
                    continue
            sink_repl_specs.append({"name": r_name or f"sink_repl_{r_token_id}", "token_id": int(r_token_id)})

        # Evaluate each excerpt
        per_intervention_ppl: dict[str, list[float]] = {}
        for excerpt_ids in excerpts:
            # baseline ids
            base_ids = list(excerpt_ids)
            inp = torch.tensor(base_ids, dtype=torch.long).unsqueeze(0).to(device)
            attn_mask = torch.ones_like(inp)
            base = ppl_for_sequence(model, inp, attn_mask, loss_start=base_loss_start)
            per_intervention_ppl.setdefault("baseline", []).append(base.ppl)
            rows_ppl.append(
                {
                    "length": L,
                    "intervention": "baseline",
                    "ppl": base.ppl,
                    "tokens": base.tokens,
                    "loss_start": base_loss_start,
                    "loss_tokens": loss_tokens,
                }
            )

            # "prefix" intervention: sweep over each prefix length in prefix_len_list.
            # Build a fixed-length prefix and PREPEND it; shift loss_start accordingly.
            for plen in prefix_len_list:
                for p in prefixes:
                    name = str(p.get("name", ""))
                    if name == "baseline":
                        continue
                    pref_ids_raw = prefix_name_to_ids.get(name, [])
                    pref_ids = _fixed_len_prefix_ids(pref_ids_raw, plen)
                    new_ids = _prepend_prefix(base_ids, pref_ids)
                    inp2 = torch.tensor(new_ids, dtype=torch.long).unsqueeze(0).to(device)
                    attn_mask2 = torch.ones_like(inp2)
                    out = ppl_for_sequence(model, inp2, attn_mask2, loss_start=int(base_loss_start + len(pref_ids)))
                    ikey = f"prefix:{name}:plen{plen}"
                    per_intervention_ppl.setdefault(ikey, []).append(out.ppl)
                    rows_ppl.append(
                        {
                            "length": L,
                            "intervention": ikey,
                            "ppl": out.ppl,
                            "tokens": out.tokens,
                            "loss_start": int(base_loss_start + len(pref_ids)),
                            "loss_tokens": loss_tokens,
                            "prefix_len": plen,
                        }
                    )

            # sink token replacement at identified sink positions (usually includes pos 0)
            for spec in sink_repl_specs:
                repl_name = str(spec["name"])
                repl_id = int(spec["token_id"])
                for pos in sinks[:2]:
                    if pos < 0 or pos >= len(base_ids):
                        continue
                    new_ids = list(base_ids)
                    new_ids[int(pos)] = repl_id
                    inp3 = torch.tensor(new_ids, dtype=torch.long).unsqueeze(0).to(device)
                    attn_mask3 = torch.ones_like(inp3)
                    out = ppl_for_sequence(model, inp3, attn_mask3, loss_start=base_loss_start)
                    key = f"sinkpos{int(pos)}:{repl_name}"
                    per_intervention_ppl.setdefault(key, []).append(out.ppl)
                    rows_ppl.append(
                        {
                            "length": L,
                            "intervention": key,
                            "ppl": out.ppl,
                            "tokens": out.tokens,
                            "loss_start": base_loss_start,
                            "loss_tokens": loss_tokens,
                        }
                    )

        # Build a compact summary with ΔPPL vs baseline
        baseline_mean = float(np.nanmean(per_intervention_ppl.get("baseline", [np.nan])))
        for k, v in sorted(per_intervention_ppl.items()):
            mean_p = float(np.nanmean(np.asarray(v, dtype=np.float64)))
            std_p = float(np.nanstd(np.asarray(v, dtype=np.float64)))
            rows_ppl_summary.append(
                {
                    "length": L,
                    "intervention": k,
                    "mean_ppl": mean_p,
                    "std_ppl": std_p,
                    "delta_vs_baseline": mean_p - baseline_mean,
                    "n": int(len(v)),
                    "loss_tokens": loss_tokens,
                }
            )

        # save curve csv
        curve_csv = os.path.join(out_dir, f"{run_name}_L{L}_curve.csv")
        pd.DataFrame({"pos": np.arange(L), "attn_mass": mean_curve}).to_csv(curve_csv, index=False)

        # plot
        curve_png = os.path.join(out_dir, f"{run_name}_L{L}_curve.png")
        _plot_curve(mean_curve, curve_png, title=f"{model_name} PG-19 Attention-to-Position (L={L})")

        rows_curve.append({"length": L, "n_excerpts": n_ok, "sink_top5": json.dumps(sinks)})

    # summary tables
    pd.DataFrame(rows_curve).to_csv(os.path.join(out_dir, f"{run_name}_sink_summary.csv"), index=False)
    pd.DataFrame(rows_ppl).to_csv(os.path.join(out_dir, f"{run_name}_ppl_interventions.csv"), index=False)

    df_sum = pd.DataFrame(rows_ppl_summary)
    df_sum.to_csv(os.path.join(out_dir, f"{run_name}_ppl_summary.csv"), index=False)

    _plot_delta_ppl(
        df_sum,
        os.path.join(out_dir, f"{run_name}_delta_ppl.png"),
        title=f"{model_name} PG-19 ΔPPL vs baseline (loss last {int(analysis_cfg.get('eval_last_n', 128))} tokens)",
    )

    _plot_prefix_ablation(
        df_sum,
        os.path.join(out_dir, f"{run_name}_prefix_ablation.png"),
        title=f"{model_name} PG-19 Prefix Length Ablation (eval last {int(analysis_cfg.get('eval_last_n', 128))} tokens)",
    )


if __name__ == "__main__":
    main()
