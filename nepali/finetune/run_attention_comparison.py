"""Pretrained-vs-finetuned attention comparison (Phase 3, section 3.2): reuses the Phase 2
attention toolkit (hindi/eval/attention_analysis.py, plotting.py) completely unchanged, just
run once against the pretrained checkpoint and once against the finetuned checkpoint, on a
comparative-reasoning example sentence (as required: "especially on comparative-reasoning
prompts"). Produces heatmaps for one early (layer 0) and one late (layer 5) layer per model,
plus full entropy/distance grids for both, so the shift can be read numerically, not just
visually.
"""

import json
import os
import sys

import numpy as np
import sentencepiece as spm
import torch

LANGUAGE = "nepali"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from gpt import GPTLanguageModel  # noqa: E402
from attention_analysis import get_attention_weights, attention_entropy, mean_attention_distance  # noqa: E402
from plotting import plot_attention_heatmap, plot_entropy_grid, plot_distance_grid  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(ROOT, "configs", "model_config.json")) as f:
    cfg = json.load(f)
sp = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "tokenizer", f"{LANGUAGE}_tokenizer.model"))


def load_model(checkpoint_path):
    model = GPTLanguageModel(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], context_length=cfg["context_length"],
        embed_dropout=cfg["embed_dropout"], attn_dropout=cfg["attn_dropout"],
        resid_dropout=cfg["resid_dropout"], tie_embeddings=cfg["tie_embeddings"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def analyze(model, label, out_dir, example_sentence):
    pieces, attn_per_layer = get_attention_weights(model, sp, example_sentence, device=device)
    entropy_grid = attention_entropy(attn_per_layer)
    distance_grid = mean_attention_distance(attn_per_layer)

    plot_entropy_grid(entropy_grid, f"{LANGUAGE}_{label}", os.path.join(out_dir, f"{label}_attention_entropy.png"))
    plot_distance_grid(distance_grid, f"{LANGUAGE}_{label}", os.path.join(out_dir, f"{label}_attention_distance.png"))

    n_layers = len(attn_per_layer)
    for layer_idx in (0, n_layers - 1):
        plot_attention_heatmap(
            attn_per_layer[layer_idx][0].cpu().numpy(), pieces, layer_idx, 0, f"{LANGUAGE}_{label}",
            os.path.join(out_dir, f"{label}_attn_heatmap_layer{layer_idx}_head0.png"),
        )
    return entropy_grid, distance_grid


def main():
    reasoning_dir = os.path.join(ROOT, "reasoning", "data")
    with open(os.path.join(reasoning_dir, "test.jsonl"), encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
    # a representative comparative-reasoning prompt, as required
    example = next(ex for ex in test_examples if ex["task_type"] == "pairwise")
    example_sentence = example["prompt"]
    print(f"[{LANGUAGE}] example sentence: {example_sentence}")

    out_dir = os.path.join(ROOT, "statistics", "phase3_finetune_outputs")
    os.makedirs(out_dir, exist_ok=True)

    pretrained_model = load_model(os.path.join(ROOT, "model", "checkpoints", "checkpoint_step30000.pt"))
    finetuned_model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "checkpoints", "finetune_step1000.pt"))

    pre_entropy, pre_distance = analyze(pretrained_model, "pretrained", out_dir, example_sentence)
    ft_entropy, ft_distance = analyze(finetuned_model, "finetuned", out_dir, example_sentence)

    summary = {
        "language": LANGUAGE,
        "example_sentence": example_sentence,
        "pretrained_mean_entropy_per_layer": pre_entropy.mean(axis=1).tolist(),
        "finetuned_mean_entropy_per_layer": ft_entropy.mean(axis=1).tolist(),
        "pretrained_mean_distance_per_layer": pre_distance.mean(axis=1).tolist(),
        "finetuned_mean_distance_per_layer": ft_distance.mean(axis=1).tolist(),
        "pretrained_overall_mean_entropy": float(pre_entropy.mean()),
        "finetuned_overall_mean_entropy": float(ft_entropy.mean()),
        "pretrained_overall_mean_distance": float(pre_distance.mean()),
        "finetuned_overall_mean_distance": float(ft_distance.mean()),
    }
    with open(os.path.join(out_dir, "attention_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{LANGUAGE}] pretrained mean entropy: {summary['pretrained_overall_mean_entropy']:.4f}, "
          f"finetuned: {summary['finetuned_overall_mean_entropy']:.4f}")
    print(f"[{LANGUAGE}] pretrained mean distance: {summary['pretrained_overall_mean_distance']:.4f}, "
          f"finetuned: {summary['finetuned_overall_mean_distance']:.4f}")
    print(f"[{LANGUAGE}] wrote {out_dir}/attention_comparison.json")


if __name__ == "__main__":
    main()
