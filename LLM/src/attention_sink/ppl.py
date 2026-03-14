from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Ppl:
    ppl: float
    nll: float
    tokens: int


@torch.inference_mode()
def ppl_for_sequence(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    loss_start: int | None = None,
    loss_end: int | None = None,
) -> Ppl:
    """Compute perplexity for a single sequence (B=1) using next-token prediction.

    If loss_start/loss_end are provided, only tokens in [loss_start, loss_end)
    contribute to the loss (implemented via -100 label masking).
    """
    model.eval()

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape [1, T], got {tuple(input_ids.shape)}")

    labels = input_ids.clone()
    seq_len = int(labels.shape[1])
    start = 0 if loss_start is None else int(max(0, min(seq_len, loss_start)))
    end = seq_len if loss_end is None else int(max(0, min(seq_len, loss_end)))
    if start > 0:
        labels[:, :start] = -100
    if end < seq_len:
        labels[:, end:] = -100

    # HF causal LM loss uses shifted labels: shift_labels = labels[..., 1:]
    shift_labels = labels[:, 1:]
    tokens = int((shift_labels != -100).sum().item())
    if tokens <= 0:
        return Ppl(ppl=float("nan"), nll=float("nan"), tokens=0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = float(outputs.loss)
    nll = loss * tokens
    return Ppl(ppl=float(math.exp(loss)), nll=nll, tokens=tokens)
