from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DatasetSpec:
    name: str
    config: Optional[str]
    split: str


def load_hf_text_dataset(spec: DatasetSpec):
    if spec.config:
        return load_dataset(spec.name, spec.config, split=spec.split)
    return load_dataset(spec.name, split=spec.split)


def tokenize_texts(tokenizer: AutoTokenizer, texts: List[str], max_length: int):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def iter_calibration_batches(
    tokenizer: AutoTokenizer,
    dataset,
    num_samples: int,
    seq_len: int,
    batch_size: int = 1,
) -> Iterable[torch.Tensor]:
    # Pick a text field
    text_key = "text" if "text" in dataset.column_names else dataset.column_names[0]
    n = min(num_samples, len(dataset))

    buf: List[str] = []
    for i in range(n):
        item = dataset[i]
        buf.append(str(item[text_key]))
        if len(buf) == batch_size:
            toks = tokenize_texts(tokenizer, buf, max_length=seq_len)
            yield toks["input_ids"], toks.get("attention_mask")
            buf = []

    if buf:
        toks = tokenize_texts(tokenizer, buf, max_length=seq_len)
        yield toks["input_ids"], toks.get("attention_mask")


def build_lm_eval_tokens(tokenizer: AutoTokenizer, dataset, max_tokens: int):
    """Concatenate texts into a single token stream for perplexity evaluation."""
    text_key = "text" if "text" in dataset.column_names else dataset.column_names[0]

    ids: List[int] = []
    for i in range(len(dataset)):
        text = str(dataset[i][text_key])
        enc = tokenizer(text, add_special_tokens=False)
        ids.extend(enc["input_ids"])
        # add EOS between docs to avoid boundary artifacts
        if tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
        if len(ids) >= max_tokens:
            break

    ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long)
