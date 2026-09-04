"""Real held-out generation-quality evaluation for the Nepali model
(Phase 2, section 2.3 "Generation quality").

Replaces the placeholder prefix/reference pairs used to exercise the
pipeline during development with real examples drawn from
data/splits/test.txt: for each sampled held-out line, the first
PREFIX_TOKENS tokens become the prompt and the following REF_TOKENS
tokens (decoded together with the prefix, so hypothesis and reference
are in the same "prefix+continuation" format) become the reference.
Generates at greedy decoding and temperatures 0.5, 1.0, and 1.5 as
required, scores each setting against the references, and writes both
the aggregate metrics and a sample of the raw generations for the
report.
"""

import json
import os
import random
import sys

import sentencepiece as spm
import torch

LANGUAGE = "nepali"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

from generation_metrics import compute_bleu_chrf_rouge, diversity_diagnostics, generate_continuations
from gpt import GPTLanguageModel
PREFIX_TOKENS = 20
REF_TOKENS = 30
N_SAMPLES = 30
SEED = 1337

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "tokenizer", f"{LANGUAGE}_tokenizer.model"))

with open(os.path.join(ROOT, "configs", "model_config.json")) as f:
    cfg = json.load(f)

model = GPTLanguageModel(
    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], context_length=cfg["context_length"],
    embed_dropout=cfg["embed_dropout"], attn_dropout=cfg["attn_dropout"],
    resid_dropout=cfg["resid_dropout"], tie_embeddings=cfg["tie_embeddings"],
).to(device)

ckpt = torch.load(os.path.join(ROOT, "model", "checkpoints", "checkpoint_step30000.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"[{LANGUAGE}] loaded checkpoint at step {ckpt['step']}")

with open(os.path.join(ROOT, "data", "splits", "test.txt"), encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

random.seed(SEED)
random.shuffle(lines)

prefixes, references = [], []
for line in lines:
    ids = sp.encode(line, out_type=int)
    if len(ids) < PREFIX_TOKENS + REF_TOKENS:
        continue
    prefix_ids = ids[:PREFIX_TOKENS]
    full_ids = ids[:PREFIX_TOKENS + REF_TOKENS]
    prefixes.append(sp.decode(prefix_ids))
    references.append(sp.decode(full_ids))
    if len(prefixes) == N_SAMPLES:
        break

print(f"[{LANGUAGE}] sampled {len(prefixes)} held-out prefix/reference pairs from test.txt")

all_generations = {}
metrics_by_setting = {}
for label, kwargs in [("greedy", dict(greedy=True)),
                       ("temp_0.5", dict(temperature=0.5)),
                       ("temp_1.0", dict(temperature=1.0)),
                       ("temp_1.5", dict(temperature=1.5))]:
    gens = generate_continuations(model, sp, prefixes, max_new_tokens=REF_TOKENS, device=device, **kwargs)
    all_generations[label] = gens
    metrics_by_setting[label] = compute_bleu_chrf_rouge(gens, references)
    print(f"[{LANGUAGE}] {label}: {metrics_by_setting[label]}")

diversity = diversity_diagnostics([g for gens in all_generations.values() for g in gens])
print(f"[{LANGUAGE}] diversity diagnostics (pooled across all settings): {diversity}")

out_dir = os.path.join(ROOT, "statistics", "phase2_eval_outputs")
os.makedirs(out_dir, exist_ok=True)

samples = [
    {"prefix": prefixes[i], "reference": references[i],
     **{label: all_generations[label][i] for label in all_generations}}
    for i in range(len(prefixes))
]

with open(os.path.join(out_dir, "generation_eval.json"), "w", encoding="utf-8") as f:
    json.dump({
        "language": LANGUAGE,
        "n_samples": len(prefixes),
        "prefix_tokens": PREFIX_TOKENS,
        "ref_tokens": REF_TOKENS,
        "metrics_by_setting": metrics_by_setting,
        "diversity": diversity,
        "samples": samples,
    }, f, ensure_ascii=False, indent=2)

results_summary_path = os.path.join(out_dir, "results_summary.json")
with open(results_summary_path) as f:
    results = json.load(f)
results["bleu_chrf_rouge"] = metrics_by_setting["greedy"]
results["bleu_chrf_rouge_by_setting"] = metrics_by_setting
results["diversity"] = diversity
with open(results_summary_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"[{LANGUAGE}] wrote {out_dir}/generation_eval.json and updated results_summary.json")
