"""Pretrained-vs-finetuned reasoning evaluation (Phase 3, section 3.1: "Report pretrained
vs. finetuned accuracy (or exact match) on the synthetic test set, qualitative
successes/failures"). Evaluated on reasoning/data/test.jsonl -- held-out entity names never
seen during finetuning (see generate_reasoning_data.py's leakage-avoidance design).

For each test example, both models greedily generate a short continuation from the prompt;
the continuation is decoded up to the first EOS token and compared via exact string match
(after stripping whitespace) against the ground-truth answer.
"""

import json
import os
import sys

import sentencepiece as spm
import torch

LANGUAGE = "hindi"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

from gpt import GPTLanguageModel  # noqa: E402

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


@torch.no_grad()
def generate_answer(model, prompt, max_new_tokens=12):
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, greedy=True)
    new_ids = out[0, len(ids):].tolist()
    eos_id = sp.eos_id()
    if eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]
    return sp.decode(new_ids).strip()


def evaluate(model, test_examples):
    correct = 0
    results = []
    for ex in test_examples:
        pred = generate_answer(model, ex["prompt"])
        is_correct = pred == ex["answer"]
        correct += int(is_correct)
        results.append({"prompt": ex["prompt"], "answer": ex["answer"], "prediction": pred,
                         "correct": is_correct, "task_type": ex["task_type"]})
    accuracy = correct / len(test_examples)
    return accuracy, results


def main():
    reasoning_dir = os.path.join(ROOT, "reasoning", "data")
    with open(os.path.join(reasoning_dir, "test.jsonl"), encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
    print(f"[{LANGUAGE}] test examples: {len(test_examples)}")

    pretrained_model = load_model(os.path.join(ROOT, "model", "checkpoints", "checkpoint_step30000.pt"))
    finetuned_model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "checkpoints", "finetune_step1000.pt"))

    pretrained_acc, pretrained_results = evaluate(pretrained_model, test_examples)
    print(f"[{LANGUAGE}] pretrained-only accuracy: {pretrained_acc:.4f}")

    finetuned_acc, finetuned_results = evaluate(finetuned_model, test_examples)
    print(f"[{LANGUAGE}] finetuned accuracy: {finetuned_acc:.4f}")

    # per-task-type breakdown for the finetuned model
    by_task = {}
    for r in finetuned_results:
        t = r["task_type"]
        by_task.setdefault(t, {"correct": 0, "total": 0})
        by_task[t]["total"] += 1
        by_task[t]["correct"] += int(r["correct"])
    by_task_acc = {t: v["correct"] / v["total"] for t, v in by_task.items()}
    print(f"[{LANGUAGE}] finetuned accuracy by task type: {by_task_acc}")

    out_dir = os.path.join(ROOT, "statistics", "phase3_finetune_outputs")
    os.makedirs(out_dir, exist_ok=True)

    successes = [r for r in finetuned_results if r["correct"]][:5]
    failures = [r for r in finetuned_results if not r["correct"]][:5]

    summary = {
        "language": LANGUAGE,
        "n_test": len(test_examples),
        "pretrained_accuracy": pretrained_acc,
        "finetuned_accuracy": finetuned_acc,
        "finetuned_accuracy_by_task_type": by_task_acc,
        "qualitative_successes": successes,
        "qualitative_failures": failures,
    }
    with open(os.path.join(out_dir, "reasoning_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "reasoning_eval_pretrained_full.json"), "w", encoding="utf-8") as f:
        json.dump(pretrained_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "reasoning_eval_finetuned_full.json"), "w", encoding="utf-8") as f:
        json.dump(finetuned_results, f, ensure_ascii=False, indent=2)

    print(f"[{LANGUAGE}] wrote {out_dir}/reasoning_eval.json")


if __name__ == "__main__":
    main()
