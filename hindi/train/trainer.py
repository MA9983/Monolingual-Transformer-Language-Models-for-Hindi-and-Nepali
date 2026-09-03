"""Training loop for one language's GPTLanguageModel (Phase 2, section
2.2 "Pretraining"): AdamW optimizer, linear-warmup-then-cosine-decay
learning-rate schedule, periodic validation loss, and checkpointing that
saves (and can resume from) model weights, optimizer state, scheduler
state, the current training step, and the run's config -- required
because Kaggle/Colab GPU sessions can terminate unexpectedly mid-run.

No visualization logic here (kept separate per the assignment's
guideline to not mix computation and plotting) -- `train()` only returns
the recorded loss history; report/plot/generate_plots.py-style code
elsewhere turns that into the required loss-curve figures.
"""

import csv
import math
import os
import time

import torch


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, train_config, checkpoint_dir, device):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = train_config
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
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)

        self.step = 0
        self.loss_history = []  # list of (step, train_loss) tuples
        self.val_loss_history = []  # list of (step, val_loss) tuples

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg["batch_size"], shuffle=True,
            num_workers=2, pin_memory=(device != "cpu"), drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.cfg["batch_size"], shuffle=True,
            num_workers=2, pin_memory=(device != "cpu"), drop_last=True,
        )

    def _lr_lambda(self, step):
        """Linear warmup over `warmup_steps`, then cosine decay from the
        peak learning rate down to `min_learning_rate` over the remaining
        `max_steps - warmup_steps` steps. Returned as a *multiplier* on
        the optimizer's base lr (LambdaLR's contract), so this function's
        output is relative, not an absolute learning rate.
        """
        warmup = self.cfg["warmup_steps"]
        max_steps = self.cfg["max_steps"]
        peak_lr = self.cfg["learning_rate"]
        min_lr = self.cfg["min_learning_rate"]

        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (peak_lr - min_lr) * cosine
        return target_lr / peak_lr

    def save_checkpoint(self, path=None):
        path = path or os.path.join(self.checkpoint_dir, f"checkpoint_step{self.step}.pt")
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_config": self.cfg,
            "loss_history": self.loss_history,
            "val_loss_history": self.val_loss_history,
        }, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
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
        candidates = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_step")]
        if not candidates:
            return None
        candidates.sort(key=lambda f: int(f.split("checkpoint_step")[1].split(".pt")[0]))
        return os.path.join(self.checkpoint_dir, candidates[-1])

    @torch.no_grad()
    def evaluate(self, num_batches=None):
        self.model.eval()
        num_batches = num_batches or self.cfg["eval_batches"]
        total_loss = 0.0
        n = 0
        for i, (x, y) in enumerate(self.val_loader):
            if i >= num_batches:
                break
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
        start_time = time.time()
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
                elapsed = time.time() - start_time
                print(f"step {self.step}/{max_steps} | train_loss {loss.item():.4f} | "
                      f"val_loss {val_loss:.4f} | lr {self.scheduler.get_last_lr()[0]:.2e} | "
                      f"elapsed {elapsed/60:.1f}min", flush=True)

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
