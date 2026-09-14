"""A single Transformer decoder block (Phase 2, section 2.1 "Transformer
block"): causal self-attention sublayer + position-wise feed-forward
sublayer, each wrapped in a residual connection with pre-norm LayerNorm,
plus dropout.

Pre-norm (LayerNorm applied *before* each sublayer, not after) was chosen
over post-norm because it keeps the residual stream itself unnormalized,
so gradients can flow through the residual path largely unimpeded back
to earlier layers -- this is why pre-norm is the standard choice for
deep Transformer stacks (post-norm's residual-path normalization tends
to cause vanishing/unstable gradients as depth grows, often requiring a
learning-rate warmup workaround that pre-norm needs less of).
"""

import torch.nn as nn

from attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: two linear layers with a GELU
    non-linearity between them, and an inner dimension (`d_ff`) larger
    than `d_model` (2208 vs. 512, ~4.3x here -- close to but not exactly
    the common 4x ratio, deliberately tuned so the model's total parameter
    count lands almost exactly on the assignment's ~25M target rather than
    the 24.04M an exact 4x/2048 ratio gives) so the block can represent a
    richer, higher-dimensional intermediate transformation before
    projecting back down.
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, context_length, attn_dropout, resid_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, attn_dropout, resid_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, resid_dropout)

    def forward(self, x, return_attn=False):
        # Pre-norm: normalize, then apply the sublayer, then add the
        # *original* (un-normalized) x back in as the residual connection.
        if return_attn:
            attn_out, attn_weights = self.attn(self.ln1(x), return_attn=True)
        else:
            attn_out = self.attn(self.ln1(x))
            attn_weights = None
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if return_attn:
            return x, attn_weights
        return x
