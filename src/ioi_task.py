from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformer_lens import HookedTransformer
except Exception:  # pragma: no cover
    HookedTransformer = None  # type: ignore


@dataclass
class IOIExample:
    clean: str
    corrupted: str


class IOIDataset(Dataset):
    def __init__(self, examples: List[IOIExample], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        clean_ids = self.tokenizer(ex.clean, return_tensors="pt")
        corr_ids = self.tokenizer(ex.corrupted, return_tensors="pt")
        return clean_ids.input_ids[0], corr_ids.input_ids[0]


def load_ioi_examples(path: str | Path) -> List[IOIExample]:
    path = Path(path)
    data = json.loads(path.read_text())
    return [IOIExample(clean=item["clean"], corrupted=item["corrupted"]) for item in data]


def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_list, corr_list = zip(*batch)
    clean_padded = torch.nn.utils.rnn.pad_sequence(clean_list, batch_first=True, padding_value=pad_token_id)
    corr_padded = torch.nn.utils.rnn.pad_sequence(corr_list, batch_first=True, padding_value=pad_token_id)
    return clean_padded, corr_padded


def make_dataloader(examples: List[IOIExample], model: HookedTransformer, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    tokenizer = model.tokenizer
    ds = IOIDataset(examples, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: collate_pad(b, pad_id))
