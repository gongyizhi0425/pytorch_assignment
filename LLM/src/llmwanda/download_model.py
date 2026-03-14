from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model id (e.g. mistralai/Mistral-7B-v0.1)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Where to put the model. For 7B on 8GB GPU, prefer cpu or auto+device_map.",
    )
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
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
        help="Do not attempt any network calls; load only from local cache.",
    )
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/cache model+tokenizer; do not move model to device.",
    )
    ap.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Pass through to from_pretrained(device_map=...). Common: auto (requires accelerate).",
    )
    ap.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Use low_cpu_mem_usage=True when loading weights (recommended).",
    )
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    )
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    model_id = args.model
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Downloading tokenizer: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=args.cache_dir,
            token=token,
            local_files_only=bool(args.local_files_only),
        )
    except OSError as e:
        msg = str(e)
        if "gated repo" in msg.lower() or "restricted" in msg.lower() or "401" in msg:
            raise SystemExit(
                "Cannot access gated model repo.\n"
                "1) Open the model page on HuggingFace and request/accept access (license).\n"
                "2) Login: `huggingface-cli login` (or set env var HF_TOKEN).\n"
                f"3) Re-run with: --token <YOUR_TOKEN> (optional).\n\nOriginal error: {e}"
            )
        raise

    print(f"Downloading model weights: {model_id}")
    load_kwargs = {
        "torch_dtype": dtype,
    }
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir
    if token is not None:
        load_kwargs["token"] = token
    if args.local_files_only:
        load_kwargs["local_files_only"] = True
    if args.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if not args.download_only and args.device_map is None:
        # If device_map is used, transformers already placed modules.
        model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    placement = f"device={device}" if args.device_map is None else f"device_map={args.device_map}"
    print(f"Done. total_params={total_params:,} {placement} dtype={dtype} download_only={args.download_only}")


if __name__ == "__main__":
    main()
