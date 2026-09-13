"""Driver script: finetune Nepali's Phase 2 pretrained checkpoint on the synthetic reasoning
dataset (Phase 3, section 3.1 "Protocol" -- starts from the pretrained checkpoint, keeps the
tokenizer/vocab fixed, documents hyperparameters, saves resume-capable checkpoints).
"""

import json
import os
import sys

import sentencepiece as spm
import torch

LANGUAGE = "nepali"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpt import GPTLanguageModel  # noqa: E402
from dataset import ReasoningDataset  # noqa: E402
from finetune_trainer import FinetuneTrainer  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(ROOT, "configs", "model_config.json")) as f:
    model_cfg = json.load(f)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_config.json")) as f:
    finetune_cfg = json.load(f)

torch.manual_seed(finetune_cfg["seed"])

sp = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "tokenizer", f"{LANGUAGE}_tokenizer.model"))

model = GPTLanguageModel(
    vocab_size=model_cfg["vocab_size"], d_model=model_cfg["d_model"], n_layers=model_cfg["n_layers"],
    n_heads=model_cfg["n_heads"], d_ff=model_cfg["d_ff"], context_length=model_cfg["context_length"],
    embed_dropout=model_cfg["embed_dropout"], attn_dropout=model_cfg["attn_dropout"],
    resid_dropout=model_cfg["resid_dropout"], tie_embeddings=model_cfg["tie_embeddings"],
).to(device)

pretrained_path = os.path.join(ROOT, "model", "checkpoints", "checkpoint_step30000.pt")
pretrained_ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
model.load_state_dict(pretrained_ckpt["model_state_dict"])
print(f"[{LANGUAGE}] loaded pretrained checkpoint from step {pretrained_ckpt['step']}")

reasoning_dir = os.path.join(ROOT, "reasoning", "data")
train_ds = ReasoningDataset(os.path.join(reasoning_dir, "train.jsonl"), sp, max_length=model_cfg["context_length"])
val_ds = ReasoningDataset(os.path.join(reasoning_dir, "val.jsonl"), sp, max_length=model_cfg["context_length"])
print(f"[{LANGUAGE}] train examples: {len(train_ds)}, val examples: {len(val_ds)}")

checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
log_dir = os.path.join(ROOT, "statistics", "phase3_finetune_outputs")
os.makedirs(log_dir, exist_ok=True)

trainer = FinetuneTrainer(model, train_ds, val_ds, finetune_cfg, checkpoint_dir, device)
result = trainer.train(resume=True, log_csv_path=os.path.join(log_dir, "finetune_log.csv"))

print(f"[{LANGUAGE}] finetuning complete. Final train_loss={result['loss_history'][-1][1]:.4f}, "
      f"final val_loss={result['val_loss_history'][-1][1]:.4f}")
