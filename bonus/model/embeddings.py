"""Input representation for the decoder-only Transformer: token
embeddings followed by embedding dropout.

BONUS ABLATION (positional embeddings removed): this is the standard
model's real input-representation module (identical to hindi/model/), and
it never added position here in the first place -- the real model injects
position via RoPE inside attention.py instead. What makes this an ablation
is that bonus/model/attention.py (unlike hindi/model/attention.py) has had
that RoPE injection deleted entirely and nothing put in its place. So in
this ablated model, position is not encoded *anywhere* -- not here, not in
attention -- and self-attention is a permutation-invariant operation over
the input tokens: shuffling the token order would produce the exact same
per-token output (up to which token sits at which now-shuffled position),
since nothing in the architecture can tell two positions apart.
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
