"""Train a from-scratch SentencePiece (BPE) tokenizer for one language and
choose its vocabulary size empirically via a fertility/UNK-rate sweep on
held-out (validation) text, per the Phase 1 requirement to pick vocab size
this way rather than guessing a single number.

Each language's tokenizer is trained completely independently -- a fresh
SentencePieceTrainer call per language on that language's own train split
only, so no vocabulary, merges, or model state is ever shared between
Hindi and Nepali.
"""

import os
from collections import Counter

import sentencepiece as spm


def train_one(input_path, model_prefix, vocab_size, model_type="bpe",
               character_coverage=0.9995, byte_fallback=True, input_sentence_size=1_000_000):
    """Train one SentencePiece model. byte_fallback=True guarantees zero
    true UNKs (any byte sequence can be represented), which is why the
    sweep's selection criterion below focuses on fertility rather than
    UNK rate alone -- UNK rate is still reported since it is a required
    deliverable statistic and remains meaningful for model_type="unigram".

    input_sentence_size=1,000,000 (randomly sampled via
    shuffle_input_sentence=True) rather than the full multi-million-line
    train split: SentencePiece's BPE trainer memory use is a large
    multiple of the sampled text size, and this machine's available RAM
    is limited by other running processes, so training on the full ~6-8M
    line split risked OOM. 1M sentences is still a large, representative
    sample for subword statistics at this corpus scale.
    """
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )


def evaluate(model_path, eval_path, max_lines=50_000):
    """Compute the statistics the Phase 1 report needs: average
    characters per token, fertility (tokens per whitespace-delimited
    word), and UNK rate, all measured on held-out text the tokenizer was
    never trained on.
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    unk_id = sp.unk_id()
    total_tokens = total_chars = total_words = unk_count = n_lines = 0
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            total_tokens += len(ids)
            total_chars += len(line)
            total_words += len(line.split())
            unk_count += sum(1 for i in ids if i == unk_id)
            n_lines += 1
            if n_lines >= max_lines:
                break
    return {
        "eval_lines": n_lines,
        "vocab_size_actual": sp.vocab_size(),
        "avg_chars_per_token": (total_chars / total_tokens) if total_tokens else None,
        "fertility_tokens_per_word": (total_tokens / total_words) if total_words else None,
        "unk_rate": (unk_count / total_tokens) if total_tokens else None,
        "total_eval_tokens": total_tokens,
    }


def token_frequency_stats(model_path, corpus_path, max_lines=200_000, top_n=20):
    """Encode a sample of `corpus_path` with the final trained model and
    report the piece-frequency distribution -- a required Phase 1
    deliverable ("token-frequency statistics") that plain fertility/UNK-rate
    evaluation doesn't cover. Measured on the training-side sample (not
    val) since this describes the vocabulary's own usage profile, not a
    held-out generalization metric.
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    counts = Counter()
    n_lines = total_tokens = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            counts.update(ids)
            total_tokens += len(ids)
            n_lines += 1
            if n_lines >= max_lines:
                break

    vocab_size = sp.vocab_size()
    used_ids = len(counts)
    freqs = sorted(counts.values(), reverse=True)
    top_pieces = [
        {"piece": sp.id_to_piece(tid), "id": tid, "count": c, "frequency": c / total_tokens}
        for tid, c in counts.most_common(top_n)
    ]
    top_100_coverage = sum(freqs[:100]) / total_tokens if total_tokens else None

    return {
        "sample_lines": n_lines,
        "sample_total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "unique_tokens_used_in_sample": used_ids,
        "unused_tokens_in_sample": vocab_size - used_ids,
        "pct_vocab_used_in_sample": used_ids / vocab_size if vocab_size else None,
        "min_count": freqs[-1] if freqs else None,
        "max_count": freqs[0] if freqs else None,
        "mean_count": (sum(freqs) / len(freqs)) if freqs else None,
        "top_100_tokens_share_of_all_occurrences": top_100_coverage,
        "top_tokens": top_pieces,
    }


def sample_tokenizations(model_path, examples):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    out = []
    for text in examples:
        pieces = sp.encode(text, out_type=str)
        out.append({"text": text, "pieces": pieces, "n_pieces": len(pieces)})
    return out


def sweep_and_train(language, train_path, val_path, out_dir, candidate_vocab_sizes,
                     model_type="bpe", max_unk_rate=0.001, sample_texts=None):
    """Train one candidate model per vocab size in candidate_vocab_sizes,
    evaluate each on val_path, then select the smallest-fertility
    candidate among those meeting the UNK-rate bar (falling back to
    lowest fertility overall if none do) and keep only that final model's
    files on disk.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for vs in candidate_vocab_sizes:
        prefix = os.path.join(out_dir, f"_sweep_{vs}")
        print(f"[{language}] training candidate vocab_size={vs} ...", flush=True)
        train_one(train_path, prefix, vs, model_type=model_type)
        stats = evaluate(prefix + ".model", val_path)
        stats["vocab_size_requested"] = vs
        results.append(stats)
        print(f"[{language}] vocab_size={vs} -> "
              f"fertility={stats['fertility_tokens_per_word']:.3f} tok/word, "
              f"avg_chars_per_token={stats['avg_chars_per_token']:.3f}, "
              f"unk_rate={stats['unk_rate']:.5f}", flush=True)

    meeting_bar = [r for r in results if (r["unk_rate"] or 0) <= max_unk_rate]
    pool = meeting_bar if meeting_bar else results
    best = min(pool, key=lambda r: r["fertility_tokens_per_word"])
    chosen_vocab_size = best["vocab_size_requested"]
    print(f"[{language}] selected vocab_size={chosen_vocab_size} "
          f"(lowest fertility among candidates with unk_rate <= {max_unk_rate})", flush=True)

    final_prefix = os.path.join(out_dir, f"{language}_tokenizer")
    train_one(train_path, final_prefix, chosen_vocab_size, model_type=model_type)
    final_eval = evaluate(final_prefix + ".model", val_path, max_lines=200_000)
    freq_stats = token_frequency_stats(final_prefix + ".model", train_path)
    examples = sample_tokenizations(final_prefix + ".model", sample_texts or [])

    for vs in candidate_vocab_sizes:
        for ext in (".model", ".vocab"):
            p = os.path.join(out_dir, f"_sweep_{vs}{ext}")
            if os.path.exists(p):
                os.remove(p)

    return {
        "language": language,
        "model_type": model_type,
        "sweep_candidates": results,
        "selection_criterion": f"lowest fertility (tokens/word) among candidates with unk_rate <= {max_unk_rate}",
        "selected_vocab_size": chosen_vocab_size,
        "final_model_path": final_prefix + ".model",
        "final_vocab_path": final_prefix + ".vocab",
        "final_eval_on_full_val": final_eval,
        "token_frequency_stats": freq_stats,
        "tokenization_examples": examples,
    }
