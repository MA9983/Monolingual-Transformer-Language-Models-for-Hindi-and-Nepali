"""PyTorch Dataset over a flat, pre-tokenized uint16 binary file (see
prepare_data.py), yielding fixed-length (context_length) input/target
blocks for causal language-model training.

Uses numpy memmap rather than loading the whole token array into RAM: a
~500M-token split is ~1GB as uint16, which is fine to hold in memory on a
Kaggle GPU instance, but memmap costs nothing extra and keeps this
correct even on more constrained machines (e.g. local correctness
testing on this project's own machine).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    def __init__(self, bin_path, context_length):
        self.context_length = context_length
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if len(self.data) <= context_length:
            raise ValueError(
                f"{bin_path} has only {len(self.data)} tokens, "
                f"need more than context_length={context_length}"
            )

    def __len__(self):
        # Every valid start index i in [0, len(data) - context_length - 1)
        # gives one (x, y) sample: x = data[i : i+context_length],
        # y = data[i+1 : i+context_length+1] -- next-token targets.
        return len(self.data) - self.context_length - 1

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
