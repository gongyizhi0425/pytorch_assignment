from __future__ import annotations

import argparse
import json
import os
import gc
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_prune.activation_stats import activation_outlier_summary, collect_activation_rms
from llm_prune.data import DatasetSpec, build_lm_eval_tokens, iter_calibration_batches, load_hf_text_dataset
from llm_prune.metrics import count_parameters, eval_perplexity, measure_prefill_decode_speed
from llm_prune.plot import plot_ppl_vs_sparsity
from llm_prune.prune import prune_magnitude, prune_wanda, wanda_score_summary
from llm_prune.utils import ensure_dir, pick_device, pick_dtype, set_seed


def _load_yaml(path: str) -> Dict[str, Any]:
    # Allow running from either repo root or src/ by resolving relative paths.
    if not os.path.isabs(path) and not os.path.exists(path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(repo_root, path)
        if os.path.exists(candidate):
            path = candidate

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _maybe_to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x)


def _make_repro_snapshot(cfg: Dict[str, Any], calibration_section: str) -> Dict[str, Any]:
    """Create a compact, per-run config snapshot.

    The original YAML contains both `calibration` and `calibration_shifted` blocks.
    For reproducibility and clarity, we store which section was actually used for
    activation statistics in this run.
    """

    snap: Dict[str, Any] = {
        "seed": cfg.get("seed", 0),
        "model": cfg.get("model", {}),
        "eval": cfg.get("eval", {}),
        "pruning": cfg.get("pruning", {}),
        "speed": cfg.get("speed", {}),
        "output": cfg.get("output", {}),
        "run_section": calibration_section,
        "active_calibration": cfg.get(calibration_section, {}),
    }

    # Keep the base calibration for context when running shifted.
    if calibration_section != "calibration":
        snap["base_calibration"] = cfg.get("calibration", {})
    return snap


def _prepare_model_and_tokenizer(model_name: str, device: torch.device, dtype: torch.dtype):
    # Optional cache dir can be passed via cfg['model']['cache_dir'] using the global var below.
    cache_dir = getattr(_prepare_model_and_tokenizer, "cache_dir", None)
    token = getattr(_prepare_model_and_tokenizer, "token", None)
    local_files_only = bool(getattr(_prepare_model_and_tokenizer, "local_files_only", False))

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model, tok


def _build_calibration_iter(*, cfg: Dict[str, Any], tok: AutoTokenizer, calibration_section: str, device: torch.device, include_regex: Optional[str], exclude_regex: Optional[str]):
    calib_cfg = cfg[calibration_section]
    calib_name = str(calib_cfg["dataset"]).strip()

    # Calibration data can be either a HF dataset or synthetic random tokens
    if calib_name == "random_tokens":
        num_samples = int(calib_cfg.get("num_samples", 128))
        seq_len = int(calib_cfg.get("seq_len", 256))

        def calib_iter():
            vocab = int(tok.vocab_size)
            for _ in range(num_samples):
                input_ids = torch.randint(0, vocab, (1, seq_len), dtype=torch.long)
                attn = torch.ones((1, seq_len), dtype=torch.long)
                yield input_ids, attn

        return calib_iter()

    calib_spec = DatasetSpec(
        name=calib_name,
        config=_maybe_to_str(calib_cfg.get("config")),
        split=str(calib_cfg.get("split", "train")),
    )
    calib_ds = load_hf_text_dataset(calib_spec)
    return iter_calibration_batches(
        tok,
        calib_ds,
        num_samples=int(calib_cfg.get("num_samples", 128)),
        seq_len=int(calib_cfg.get("seq_len", 256)),
        batch_size=1,
    )


def _run_wanda_score_summary_only(*, cfg: Dict[str, Any], out_dir: str, run_tag: str, calibration_section: str) -> str:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    model_cfg = cfg["model"]
    device = pick_device(str(model_cfg.get("device", "auto")))
    dtype = pick_dtype(str(model_cfg.get("dtype", "auto")), device)
    model_name = str(model_cfg["name"])

    pruning_cfg = cfg["pruning"]
    include_regex = _maybe_to_str(pruning_cfg.get("include_regex"))
    exclude_regex = _maybe_to_str(pruning_cfg.get("exclude_regex"))

    model, tok = _prepare_model_and_tokenizer(model_name, device, dtype)
    calib_iter = _build_calibration_iter(
        cfg=cfg,
        tok=tok,
        calibration_section=calibration_section,
        device=device,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )

    act = collect_activation_rms(
        model,
        device=device,
        calibration_iter=calib_iter,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )

    summary = wanda_score_summary(
        model,
        act,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        sample_size=int(cfg.get("wanda_score_summary", {}).get("sample_size", 200000)),
        seed=seed,
    )
    summary["run_tag"] = run_tag
    summary["calibration"] = calibration_section
    summary["model"] = model_name
    summary["dtype"] = str(dtype).replace("torch.", "")
    summary["device"] = str(device)
    summary["include_regex"] = include_regex
    summary["exclude_regex"] = exclude_regex

    out_path = os.path.join(out_dir, f"{run_tag}_{calibration_section}_wanda_score_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_path


def _run_one_setting(
    *,
    cfg: Dict[str, Any],
    out_dir: str,
    run_tag: str,
    calibration_section: str,
) -> str:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    model_cfg = cfg["model"]
    device = pick_device(str(model_cfg.get("device", "auto")))
    dtype = pick_dtype(str(model_cfg.get("dtype", "auto")), device)

    model_name = str(model_cfg["name"])

    model, tok = _prepare_model_and_tokenizer(model_name, device, dtype)

    eval_cfg = cfg["eval"]
    eval_spec = DatasetSpec(
        name=str(eval_cfg["dataset"]),
        config=_maybe_to_str(eval_cfg.get("config")),
        split=str(eval_cfg.get("split", "test")),
    )
    eval_ds = load_hf_text_dataset(eval_spec)
    token_stream = build_lm_eval_tokens(tok, eval_ds, max_tokens=int(eval_cfg.get("max_eval_tokens", 200000)))

    pruning_cfg = cfg["pruning"]
    sparsity_list = list(pruning_cfg.get("sparsity_ratios", [0.3, 0.5, 0.8]))
    include_regex = _maybe_to_str(pruning_cfg.get("include_regex"))
    exclude_regex = _maybe_to_str(pruning_cfg.get("exclude_regex"))

    speed_cfg = cfg.get("speed", {"enabled": True})

    rows = []

    def eval_and_log(method: str, target_sparsity: float, report_extra: Dict[str, Any]):
        ppl_res = eval_perplexity(
            model,
            token_stream=token_stream,
            device=device,
            block_size=int(eval_cfg.get("block_size", 1024)),
        )

        speed = {}
        if bool(speed_cfg.get("enabled", True)):
            speed = measure_prefill_decode_speed(
                model,
                device=device,
                vocab_size=int(tok.vocab_size),
                prefill_seq_len=int(speed_cfg.get("prefill_seq_len", 256)),
                decode_new_tokens=int(speed_cfg.get("decode_new_tokens", 64)),
                repeats=int(speed_cfg.get("repeats", 10)),
            )

        param_count = count_parameters(model)

        row = {
            "run_tag": run_tag,
            "calibration": calibration_section,
            "model": model_name,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "method": method,
            "target_sparsity": float(target_sparsity),
            "achieved_sparsity": float(report_extra.get("achieved_sparsity", 0.0)),
            "ppl": float(ppl_res.ppl),
            "eval_tokens": int(ppl_res.tokens),
            "param_count": int(param_count),
            **speed,
            **{k: v for k, v in report_extra.items() if k not in {"achieved_sparsity"}},
        }
        rows.append(row)

    # Baseline
    eval_and_log("baseline", 0.0, {"achieved_sparsity": 0.0})

    # Free baseline model before loading more copies (avoid OOM / exit 137)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Magnitude pruning sweeps (load fresh model each time instead of deepcopy)
    for ratio in sparsity_list:
        model, tok = _prepare_model_and_tokenizer(model_name, device, dtype)
        rep = prune_magnitude(model, float(ratio), include_regex=include_regex, exclude_regex=exclude_regex)
        eval_and_log(
            "magnitude",
            float(ratio),
            {
                "achieved_sparsity": float(rep.achieved_sparsity),
                "total_weights_pruned_scope": int(rep.total_weights),
            },
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Reload fresh for Wanda (for activation stats)
    model, tok = _prepare_model_and_tokenizer(model_name, device, dtype)

    calib_cfg = cfg[calibration_section]
    calib_name = str(calib_cfg["dataset"]).strip()

    # Calibration data can be either a HF dataset or synthetic random tokens
    # (useful for a domain-shift setting without relying on external dataset scripts).
    if calib_name == "random_tokens":
        num_samples = int(calib_cfg.get("num_samples", 128))
        seq_len = int(calib_cfg.get("seq_len", 256))

        def calib_iter():
            vocab = int(tok.vocab_size)
            for _ in range(num_samples):
                input_ids = torch.randint(0, vocab, (1, seq_len), dtype=torch.long)
                attn = torch.ones((1, seq_len), dtype=torch.long)
                yield input_ids, attn

        calib_iter = calib_iter()
    else:
        calib_spec = DatasetSpec(
            name=calib_name,
            config=_maybe_to_str(calib_cfg.get("config")),
            split=str(calib_cfg.get("split", "train")),
        )
        calib_ds = load_hf_text_dataset(calib_spec)

        calib_iter = iter_calibration_batches(
            tok,
            calib_ds,
            num_samples=int(calib_cfg.get("num_samples", 128)),
            seq_len=int(calib_cfg.get("seq_len", 256)),
            batch_size=1,
        )

    act = collect_activation_rms(
        model,
        device=device,
        calibration_iter=calib_iter,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )

    act_summary = activation_outlier_summary(act)

    # Free activation-stat model before running pruning sweeps
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Wanda pruning sweeps (load fresh model each time instead of deepcopy)
    for ratio in sparsity_list:
        model, tok = _prepare_model_and_tokenizer(model_name, device, dtype)
        rep = prune_wanda(model, act, float(ratio), include_regex=include_regex, exclude_regex=exclude_regex)
        eval_and_log(
            "wanda",
            float(ratio),
            {
                "achieved_sparsity": float(rep.achieved_sparsity),
                "total_weights_pruned_scope": int(rep.total_weights),
                **{f"act_{k}": v for k, v in act_summary.items()},
            },
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, f"{run_tag}_{calibration_section}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Also dump a tiny JSON with config for reproducibility
    with open(os.path.join(out_dir, f"{run_tag}_{calibration_section}_config.json"), "w", encoding="utf-8") as f:
        json.dump(_make_repro_snapshot(cfg, calibration_section), f, indent=2, ensure_ascii=False)

    plot_ppl_vs_sparsity(csv_path, title=f"{model_name} PPL vs Sparsity ({calibration_section})")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/cache the model+tokenizer then exit.",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (e.g., mistralai/Mistral-7B-v0.1).",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store HuggingFace cache (weights/tokenizer).",
    )
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token for gated models (or set HF_TOKEN env var).",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt any network calls; load only from local HF cache_dir/default cache.",
    )
    ap.add_argument(
        "--wanda-score-summary-only",
        action="store_true",
        help="Compute and dump a whole-model WANDA score summary then exit (no pruning/eval sweeps).",
    )
    ap.add_argument(
        "--calibration-section",
        choices=["calibration", "calibration_shifted"],
        default="calibration",
        help="Which calibration section to use with --wanda-score-summary-only.",
    )
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    if args.model_name:
        cfg.setdefault("model", {})
        cfg["model"]["name"] = args.model_name

    if args.cache_dir:
        cfg.setdefault("model", {})
        cfg["model"]["cache_dir"] = args.cache_dir

    if args.token:
        cfg.setdefault("model", {})
        cfg["model"]["token"] = args.token

    if args.local_files_only:
        cfg.setdefault("model", {})
        cfg["model"]["local_files_only"] = True

    # Stash cache_dir into the loader helper.
    _prepare_model_and_tokenizer.cache_dir = cfg.get("model", {}).get("cache_dir")
    _prepare_model_and_tokenizer.token = cfg.get("model", {}).get("token")
    _prepare_model_and_tokenizer.local_files_only = bool(cfg.get("model", {}).get("local_files_only", False))

    if args.download_only:
        model_cfg = cfg.get("model", {})
        device = pick_device(str(model_cfg.get("device", "auto")))
        dtype = pick_dtype(str(model_cfg.get("dtype", "auto")), device)
        model_name = str(model_cfg.get("name"))
        _prepare_model_and_tokenizer(model_name, device, dtype)
        print(f"Downloaded/cached: {model_name}")
        return

    out_cfg = cfg.get("output", {})
    out_dir = str(out_cfg.get("out_dir", "results_llm"))
    # Make output path stable regardless of CWD.
    if not os.path.isabs(out_dir):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(repo_root, out_dir)
    run_name = str(out_cfg.get("run_name", "llm_wanda"))
    ensure_dir(out_dir)

    if bool(args.wanda_score_summary_only):
        _run_wanda_score_summary_only(
            cfg=cfg,
            out_dir=out_dir,
            run_tag=run_name,
            calibration_section=str(args.calibration_section),
        )
        return

    # Normal calibration
    _run_one_setting(cfg=cfg, out_dir=out_dir, run_tag=run_name, calibration_section="calibration")

    # Shifted calibration (intentionally degrade)
    shifted = cfg.get("calibration_shifted", {})
    if bool(shifted.get("enabled", False)):
        _run_one_setting(cfg=cfg, out_dir=out_dir, run_tag=run_name, calibration_section="calibration_shifted")


if __name__ == "__main__":
    main()
