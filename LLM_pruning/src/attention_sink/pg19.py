from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class Pg19Config:
    split: str = "test"
    streaming: bool = True
    num_excerpts: int = 50
    min_book_tokens: int = 4000


def load_pg19_dataset(split: str, streaming: bool):
    # Key code: streaming PG-19 from HuggingFace datasets
    # ds = load_dataset("pg19", split="test", streaming=True)
    return load_dataset("pg19", split=split, streaming=streaming, trust_remote_code=True)


def iter_pg19_excerpts(
    *,
    tokenizer: AutoTokenizer,
    ds,
    num_excerpts: int,
    min_book_tokens: int,
    excerpt_len: int,
    windows_per_book: int = 1,
    seed: int = 0,
) -> Iterator[list[int]]:
    """Yield token-id excerpts of fixed length from PG-19 streaming dataset."""

    rng = random.Random(seed)

    produced = 0
    for item in ds:
        if produced >= num_excerpts:
            break

        text = item.get("text")
        if not text:
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < max(min_book_tokens, excerpt_len + 1):
            continue

        # sample 1+ random windows from this book
        for _ in range(windows_per_book):
            if produced >= num_excerpts:
                break
            start = rng.randint(0, len(ids) - excerpt_len - 1)
            excerpt = ids[start : start + excerpt_len]
            if len(excerpt) == excerpt_len:
                yield excerpt
                produced += 1
