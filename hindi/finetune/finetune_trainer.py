"""Finetuning loop for reasoning tasks (Phase 3, section 3.1), starting from a Phase 2
pretrained checkpoint. Mirrors hindi/train/trainer.py's Trainer (AdamW, checkpointing with
model/optimizer/scheduler/step/config, resume support) so finetuned checkpoints are saved in
the same resume-capable format the assignment requires -- but adapted for the small,
variable-length ReasoningDataset (padded batches, masked prompt loss) instead of the
fixed-length memmap blocks pretraining used.
"""

import csv
import math
import os

import torch
from torch.utils.data import DataLoader

from dataset import collate_fn


class FinetuneTrainer:
    def __init__(self, model, train_dataset, val_dataset, finetune_config, checkpoint_dir, device):
        self.model = model.to(device)
        self.cfg = finetune_config
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            betas=(self.cfg["adam_beta1"], self.cfg["adam_beta2"]),
            eps=self.cfg["adam_eps"],
            weight_decay=self.cfg["weight_decay"],
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.cfg["batch_size"], shuffle=True,
            collate_fn=collate_fn, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.cfg["batch_size"], shuffle=False,
            collate_fn=collate_fn, drop_last=False,
        )

        steps_per_epoch = max(1, len(self.train_loader))
        max_steps = self.cfg["epochs"] * steps_per_epoch
        self.cfg["max_steps"] = max_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)

        self.step = 0
        self.loss_history = []
        self.val_loss_history = []

    def _lr_lambda(self, step):
        warmup = self.cfg["warmup_steps"]
        max_steps = self.cfg["max_steps"]
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def save_checkpoint(self, path=None):
        path = path or os.path.join(self.checkpoint_dir, f"finetune_step{self.step}.pt")
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "finetune_config": self.cfg,
            "loss_history": self.loss_history,
            "val_loss_history": self.val_loss_history,
        }, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt["step"]
        self.loss_history = ckpt.get("loss_history", [])
        self.val_loss_history = ckpt.get("val_loss_history", [])
        return ckpt

    def find_latest_checkpoint(self):
        if not os.path.isdir(self.checkpoint_dir):
            return None
        candidates = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("finetune_step")]
        if not candidates:
            return None
        candidates.sort(key=lambda f: int(f.split("finetune_step")[1].split(".pt")[0]))
        return os.path.join(self.checkpoint_dir, candidates[-1])

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()
            n += 1
        self.model.train()
        return total_loss / max(1, n)

    def train(self, resume=True, log_csv_path=None):
        if resume:
            latest = self.find_latest_checkpoint()
            if latest is not None:
                self.load_checkpoint(latest)
                print(f"Resumed from {latest} at step {self.step}", flush=True)

        self.model.train()
        max_steps = self.cfg["max_steps"]
        grad_clip = self.cfg["grad_clip_norm"]
        eval_every = self.cfg["eval_interval_steps"]
        ckpt_every = self.cfg["checkpoint_interval_steps"]

        data_iter = iter(self.train_loader)
        while self.step < max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            self.loss_history.append((self.step, loss.item()))

            if self.step % eval_every == 0 or self.step == max_steps:
                val_loss = self.evaluate()
                self.val_loss_history.append((self.step, val_loss))
                print(f"step {self.step}/{max_steps} | train_loss {loss.item():.4f} | "
                      f"val_loss {val_loss:.4f} | lr {self.scheduler.get_last_lr()[0]:.2e}", flush=True)

            if self.step % ckpt_every == 0 or self.step == max_steps:
                path = self.save_checkpoint()
                print(f"  saved checkpoint: {path}", flush=True)
                if log_csv_path:
                    self._write_log_csv(log_csv_path)

        return {"loss_history": self.loss_history, "val_loss_history": self.val_loss_history}

    def _write_log_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss"])
            writer.writerows(self.loss_history)
        val_path = path.replace(".csv", "_val.csv")
        with open(val_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "val_loss"])
            writer.writerows(self.val_loss_history)
