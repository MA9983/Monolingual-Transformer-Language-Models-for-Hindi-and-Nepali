"""Intrinsic language-modeling evaluation (Phase 2, section 2.3):
validation cross-entropy, perplexity, and bits-per-byte (BPB).

BPB normalizes by the UTF-8 *byte* length of the underlying text rather
than by token count, which is exactly why the assignment asks for it:
Hindi and Nepali use separately-trained tokenizers with different
fertility (tokens/word), so a raw per-token loss or perplexity is not
directly comparable between the two languages -- a "token" means a
different amount of text in each. Bytes are a tokenizer-independent unit,
so BPB puts both languages' language-modeling quality on the same scale.
"""

import math
import os

import numpy as np
import torch


@torch.no_grad()
def compute_ppl_bpb(model, bin_path, text_path, context_length, device, batch_size=32):
    """Evaluate `model` on every non-overlapping context_length block of
    the pre-tokenized `bin_path` (see prepare_data.py), computing:
    - cross-entropy loss (nats/token, averaged)
    - perplexity = exp(cross-entropy)
    - bits-per-byte = (total NLL in bits) / (total UTF-8 bytes of `text_path`)

    Unlike training's random-offset sampling, evaluation walks the token
    stream in non-overlapping blocks so every token is scored exactly
    once -- necessary for an unbiased loss/PPL/BPB estimate over the
    whole split, not a random subsample of it.
    """
    model.eval()
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total_bytes = os.path.getsize(text_path)

    n_blocks = (len(data) - 1) // context_length
    total_nll_nats = 0.0
    total_tokens = 0

    batch = []
    for i in range(n_blocks):
        start = i * context_length
        chunk = data[start: start + context_length + 1].astype(np.int64)
        if len(chunk) < context_length + 1:
            break
        batch.append(chunk)
        if len(batch) == batch_size or i == n_blocks - 1:
            arr = torch.from_numpy(np.stack(batch)).to(device)
            x, y = arr[:, :-1], arr[:, 1:]
            logits, _ = model(x)
            # Sum (not mean) cross-entropy over every token in the batch,
            # so it can be correctly averaged/converted at the end over
            # the TRUE total token count rather than per-batch means.
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            total_nll_nats += nll.item()
            total_tokens += y.numel()
            batch = []

    avg_nll_nats = total_nll_nats / total_tokens
    perplexity = math.exp(avg_nll_nats)
    total_nll_bits = total_nll_nats * math.log2(math.e)
    bpb = total_nll_bits / total_bytes

    model.train()
    return {
        "tokens_evaluated": total_tokens,
        "bytes_evaluated": total_bytes,
        "cross_entropy_nats": avg_nll_nats,
        "perplexity": perplexity,
        "bits_per_byte": bpb,
    }
