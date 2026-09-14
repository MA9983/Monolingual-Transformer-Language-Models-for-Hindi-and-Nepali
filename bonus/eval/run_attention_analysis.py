"""Attention analysis for the bonus no-positional-embeddings ablation (this is the
step that crashed on Kaggle due to a missing .cpu() call, now fixed -- re-run
locally against the already-downloaded checkpoint rather than re-training).
Produces the same heatmaps/entropy/distance outputs as Phase 2's real Hindi run,
for direct comparison.
"""

import json
import os
import sys

import sentencepiece as spm
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

from gpt import GPTLanguageModel  # noqa: E402
from attention_analysis import get_attention_weights, attention_entropy, mean_attention_distance  # noqa: E402
from plotting import plot_attention_heatmap, plot_entropy_grid, plot_distance_grid  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(ROOT, "configs", "model_config.json")) as f:
    cfg = json.load(f)

sp = spm.SentencePieceProcessor(model_file=os.path.join(
    os.path.dirname(ROOT), "hindi", "tokenizer", "hindi_tokenizer.model"))

model = GPTLanguageModel(
    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], context_length=cfg["context_length"],
    embed_dropout=cfg["embed_dropout"], attn_dropout=cfg["attn_dropout"],
    resid_dropout=cfg["resid_dropout"], tie_embeddings=cfg["tie_embeddings"],
).to(device)
ckpt = torch.load(os.path.join(ROOT, "checkpoints", "checkpoint_step30000.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"[bonus-nopos] loaded checkpoint at step {ckpt['step']}")

example_sentence = "भारत एक विशाल देश है"  # same placeholder sentence used for the standard Hindi model
pieces, attn_per_layer = get_attention_weights(model, sp, example_sentence, device=device)

entropy_grid = attention_entropy(attn_per_layer)
distance_grid = mean_attention_distance(attn_per_layer)

out_dir = os.path.join(ROOT, "statistics")
os.makedirs(out_dir, exist_ok=True)

plot_entropy_grid(entropy_grid, "hindi_nopos_ablation", os.path.join(out_dir, "attention_entropy.png"))
plot_distance_grid(distance_grid, "hindi_nopos_ablation", os.path.join(out_dir, "attention_distance.png"))

n_layers = len(attn_per_layer)
n_heads = attn_per_layer[0].shape[0]
for layer_idx in (0, n_layers - 1):
    for head_idx in range(min(2, n_heads)):
        plot_attention_heatmap(
            attn_per_layer[layer_idx][head_idx].cpu().numpy(), pieces, layer_idx, head_idx, "hindi_nopos_ablation",
            os.path.join(out_dir, f"attn_heatmap_layer{layer_idx}_head{head_idx}.png"),
        )

summary = {
    "example_sentence": example_sentence,
    "mean_entropy_per_layer": entropy_grid.mean(axis=1).tolist(),
    "mean_distance_per_layer": distance_grid.mean(axis=1).tolist(),
    "overall_mean_entropy": float(entropy_grid.mean()),
    "overall_mean_distance": float(distance_grid.mean()),
}
with open(os.path.join(out_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[bonus-nopos] overall mean entropy: {summary['overall_mean_entropy']:.4f}")
print(f"[bonus-nopos] overall mean distance: {summary['overall_mean_distance']:.4f}")
print(f"[bonus-nopos] wrote outputs to {out_dir}/")
