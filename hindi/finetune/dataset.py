"""Dataset for reasoning finetuning (Phase 3, section 3.1): each example is a
(prompt, answer) pair from hindi/reasoning/data/*.jsonl, tokenized as
prompt_ids + answer_ids + [eos], with loss masked out over the prompt portion
(and over right-padding) -- the model is only trained to produce the answer
given the prompt, not to reproduce the prompt itself.
"""

import json

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class ReasoningDataset(Dataset):
    def __init__(self, jsonl_path, sp, max_length=256):
        self.sp = sp
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = self.sp.encode(ex["prompt"], out_type=int)
        answer_ids = self.sp.encode(ex["answer"], out_type=int)
        eos_id = self.sp.eos_id()

        full_ids = prompt_ids + answer_ids + [eos_id]
        full_ids = full_ids[: self.max_length + 1]  # +1 since input/target are full_ids shifted by one
        len_prompt = min(len(prompt_ids), len(full_ids) - 1)

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]
        # target_ids[k] predicts full_ids[k+1]; only keep loss once full_ids[k+1] is part of
        # the answer (+eos), i.e. k >= len_prompt - 1.
        mask_len = max(0, len_prompt - 1)
        target_ids = [IGNORE_INDEX] * mask_len + target_ids[mask_len:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def collate_fn(batch, pad_id=0):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target_batch = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_batch[i, :n] = ex["input_ids"]
        target_batch[i, :n] = ex["target_ids"]
    return input_batch, target_batch
