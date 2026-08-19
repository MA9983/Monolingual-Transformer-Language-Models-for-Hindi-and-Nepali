"""Retrain and save the final tokenizer for one language at an explicitly
chosen vocab size, overwriting whatever `sweep_and_train()` last selected.

Used when the human-reviewed decision (fertility vs. parameter-budget
trade-off, TA guidance) differs from `sweep_and_train()`'s automatic
"lowest fertility among candidates meeting the UNK-rate bar" rule -- which,
with byte_fallback=True giving 0% UNK at every candidate, always
mechanically picks the largest vocab size tested rather than reflecting an
actual trade-off. See report/phase1_report.md ("Tokenizer training") for
the reasoning behind the chosen size.

Re-trains only the single final model (not the whole candidate sweep,
which already ran and is preserved in the existing sweep_candidates on
disk) and updates the same tokenizer_stats.json in place.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_tokenizer import evaluate, sample_tokenizations, token_frequency_stats, train_one  # noqa: E402


def finalize(language, train_path, val_path, out_dir, stats_path, vocab_size,
             model_type="bpe", sample_texts=None):
    final_prefix = os.path.join(out_dir, f"{language}_tokenizer")
    print(f"[{language}] retraining final model at vocab_size={vocab_size} ...", flush=True)
    train_one(train_path, final_prefix, vocab_size, model_type=model_type)
    final_eval = evaluate(final_prefix + ".model", val_path, max_lines=200_000)
    freq_stats = token_frequency_stats(final_prefix + ".model", train_path)
    examples = sample_tokenizations(final_prefix + ".model", sample_texts or [])

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    stats["selected_vocab_size"] = vocab_size
    stats["selection_criterion"] = (
        "human-reviewed: TA-recommended 5k-10k band, weighed against fertility "
        "diminishing returns and embedding-table cost vs. the ~25M-parameter "
        "budget (overrides the automatic 'lowest fertility' rule, which always "
        "selects the largest candidate once UNK rate is 0% at every candidate)"
    )
    stats["final_eval_on_full_val"] = final_eval
    stats["token_frequency_stats"] = freq_stats
    stats["tokenization_examples"] = examples
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[{language}] done. fertility={final_eval['fertility_tokens_per_word']:.4f} "
          f"avg_chars_per_token={final_eval['avg_chars_per_token']:.4f} "
          f"unk_rate={final_eval['unk_rate']:.5f}", flush=True)
    return stats
