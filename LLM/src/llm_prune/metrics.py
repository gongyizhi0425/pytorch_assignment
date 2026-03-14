from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .utils import cuda_sync_if_needed, get_cuda_peak_mb, maybe_reset_cuda_peak


@dataclass
class PplResult:
    ppl: float
    nll: float
    tokens: int


@torch.inference_mode()
def eval_perplexity(
    model: torch.nn.Module,
    token_stream: torch.Tensor,
    device: torch.device,
    block_size: int = 1024,
    stride: Optional[int] = None,
) -> PplResult:
    """Compute perplexity over a concatenated token stream.

    Uses sliding-window evaluation to handle long streams.
    """
    model.eval()
    if stride is None:
        stride = block_size

    # Keep the full token stream on CPU to reduce peak GPU/CPU memory.
    # Move only the current window to the device.
    token_stream = token_stream.to("cpu")

    nll_sum = 0.0
    token_count = 0

    for begin in range(0, token_stream.numel(), stride):
        end = min(begin + block_size, token_stream.numel())
        input_ids = token_stream[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        # mask context tokens so we only score newly introduced tokens
        if begin > 0:
            target_ids[:, : block_size - stride] = -100

        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = float(outputs.loss)

        # Estimate number of scored tokens
        scored = int((target_ids != -100).sum().item())
        nll_sum += loss * scored
        token_count += scored

        if end == token_stream.numel():
            break

    ppl = float(math.exp(nll_sum / max(1, token_count)))
    return PplResult(ppl=ppl, nll=nll_sum, tokens=token_count)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def count_nonzero_2d_weights(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (nonzero, total) for all 2D weights."""
    nonzero = 0
    total = 0
    for _, module in model.named_modules():
        w = getattr(module, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.dim() == 2:
            t = int(w.numel())
            nz = int((w != 0).sum().item())
            nonzero += nz
            total += t
    return nonzero, total


@torch.inference_mode()
def measure_prefill_decode_speed(
    model: torch.nn.Module,
    device: torch.device,
    vocab_size: int,
    prefill_seq_len: int,
    decode_new_tokens: int,
    repeats: int = 10,
) -> Dict[str, Optional[float]]:
    """Measure prefill and decode throughput (tokens/s)."""

    model.eval()

    # random tokens are fine for measuring speed
    input_ids = torch.randint(0, vocab_size, (1, prefill_seq_len), device=device)

    # Prefill
    maybe_reset_cuda_peak(device)
    cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(input_ids=input_ids, use_cache=True)
    cuda_sync_if_needed(device)
    t1 = time.perf_counter()
    prefill_tok_s = float((repeats * prefill_seq_len) / max(1e-9, (t1 - t0)))
    peak_mb = get_cuda_peak_mb(device)

    # Decode (token-by-token with cache)
    maybe_reset_cuda_peak(device)
    cuda_sync_if_needed(device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    t0 = time.perf_counter()
    for _ in range(decode_new_tokens):
        out = model(input_ids=next_token, use_cache=True, past_key_values=past)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    cuda_sync_if_needed(device)
    t1 = time.perf_counter()
    decode_tok_s = float(decode_new_tokens / max(1e-9, (t1 - t0)))
    peak_mb_decode = get_cuda_peak_mb(device)

    return {
        "prefill_tokens_per_s": prefill_tok_s,
        "decode_tokens_per_s": decode_tok_s,
        "cuda_peak_mb_prefill": peak_mb,
        "cuda_peak_mb_decode": peak_mb_decode,
    }
