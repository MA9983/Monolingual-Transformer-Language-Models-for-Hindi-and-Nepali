"""Input representation for the decoder-only Transformer: token
embeddings followed by embedding dropout -- Phase 2, section 2.1 "Input
representation".

Unlike a learned-absolute or sinusoidal scheme, position is *not*
injected here. This model uses Rotary Position Embedding (RoPE, Su et
al. 2021 -- see rope.py), which is applied directly inside attention (to
the query/key vectors, at every layer) rather than added once to the
token embedding. RoPE was chosen over sinusoidal because it re-injects
relative-position information fresh into every layer's attention scores
rather than only once at the input (where, by the time attention happens
in a later layer, that signal has been mixed through several
transformations) -- it is also the field's current default (Llama,
Mistral, Qwen, and essentially every major open LLM since ~2022 use it).
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Maps token ids to dense vectors via a learned lookup table of shape
    (vocab_size, d_model). Scales by sqrt(d_model) before returning, the
    convention from the original Transformer paper.
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        # token_ids: (B, T) int64 -> (B, T, d_model)
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class InputEmbedding(nn.Module):
    """Token embedding + embedding dropout. No positional information is
    added here -- see rope.py / attention.py for where position actually
    enters the model.
    """

    def __init__(self, vocab_size, d_model, dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        # token_ids: (B, T) -> (B, T, d_model)
        return self.dropout(self.token_embedding(token_ids))
