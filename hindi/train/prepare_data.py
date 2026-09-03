"""Tokenize Hindi's Phase 1 text splits into flat binary token-id arrays
for training -- the standard "pre-tokenize once, memory-map many times"
recipe, so the training loop never re-runs the tokenizer per epoch/step
and can sample fixed-length context blocks directly via numpy memmap
without loading the whole split into RAM at once.

Self-contained within hindi/: reads only hindi/data/splits/ and hindi's
own SentencePiece model from Phase 1 (never a pretrained tokenizer, and
never shared with Nepali's copy of this same script).
"""

import os

import numpy as np
import sentencepiece as spm


def tokenize_split(sp_model_path, text_path, out_bin_path):
    """Encode every line of `text_path` (one cleaned record per line, per
    Phase 1's preprocess_corpus.py output) with the trained SentencePiece
    model, inserting the model's own EOS id between records so the
    training data reflects genuine document boundaries rather than
    silently concatenating unrelated records, then write the full token
    stream as a flat little-endian uint16 array (vocab_size=10,000 per
    language fits comfortably in 16 bits) to `out_bin_path`.

    Returns the total token count written, for logging/config bookkeeping.
    """
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    eos_id = sp.eos_id()
    assert sp.vocab_size() < 2**16, "vocab_size must fit in uint16 for this on-disk format"

    total_tokens = 0
    with open(text_path, encoding="utf-8") as f_in, open(out_bin_path, "wb") as f_out:
        buf = []
        flush_every = 100_000
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            ids.append(eos_id)
            buf.extend(ids)
            total_tokens += len(ids)
            if len(buf) >= flush_every:
                np.array(buf, dtype=np.uint16).tofile(f_out)
                buf = []
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f_out)
    return total_tokens


def prepare():
    here = os.path.dirname(__file__)
    language_dir = os.path.join(here, "..")
    sp_model_path = os.path.join(language_dir, "tokenizer", "hindi_tokenizer.model")
    out_dir = os.path.join(language_dir, "data", "tokenized")
    os.makedirs(out_dir, exist_ok=True)

    counts = {}
    for split in ("train", "val", "test"):
        text_path = os.path.join(language_dir, "data", "splits", f"{split}.txt")
        out_bin_path = os.path.join(out_dir, f"{split}.bin")
        n = tokenize_split(sp_model_path, text_path, out_bin_path)
        counts[split] = n
        print(f"[{split}] {n:,} tokens -> {out_bin_path}", flush=True)
    return counts


if __name__ == "__main__":
    counts = prepare()
    print(f"\n[hindi] done: {counts}")
