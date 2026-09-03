"""Multi-head causal self-attention implemented from first principles
(Phase 2, section 2.1 "Multi-head causal self-attention") -- no
nn.Transformer*, no nn.MultiheadAttention, no pre-built attention block.
Only nn.Linear and nn.Dropout are used as primitives (RoPE's rotation,
applied below, has no learnable parameters of its own).
"""

import math

import torch
import torch.nn as nn

from rope import RotaryPositionalEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """For an input X in R^{T x d_model}: projects to Q, K, V; splits each
    into `n_heads` heads of size d_head = d_model / n_heads; applies RoPE
    to Q and K (position enters here, not at the input embedding -- see
    rope.py); computes scaled dot-product attention per head under a
    causal mask; concatenates the heads back together; and applies the
    output projection W_O.

    d_model must be divisible by n_heads so every head gets an equal share
    of the model dimension (d_head = d_model // n_heads).
    """

    def __init__(self, d_model, n_heads, context_length, attn_dropout, resid_dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V projections (W_Q, W_K, W_V), each (d_model -> d_model);
        # kept as one fused linear layer for efficiency, then split by head
        # below -- mathematically identical to three separate d_model x
        # d_model projections.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        # Output projection W_O: (d_model -> d_model), applied after
        # concatenating all heads back together.
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        self.rope = RotaryPositionalEmbedding(self.d_head, context_length)

        # Causal mask: upper-triangular (excluding the diagonal) boolean
        # matrix marking "future" (query, key) pairs to be masked out.
        # Registered as a non-trainable buffer, precomputed once up to
        # context_length and sliced down to the actual sequence length
        # used in each forward pass.
        causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x, return_attn=False):
        """x: (B, T, d_model) -> (B, T, d_model).

        If return_attn=True, also returns the post-softmax attention
        weights (B, n_heads, T, T) for attention analysis (heatmaps,
        entropy, mean attention distance) in Phase 2's evaluation --
        never used during training, only for the analysis pass.
        """
        B, T, _ = x.shape

        # Project to Q, K, V together then split: (B, T, 3*d_model) -> 3 x (B, T, d_model).
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape each of (B, T, d_model) into per-head form (B, n_heads, T, d_head):
        # first view d_model as (n_heads, d_head), then move the head axis
        # before the sequence axis so attention is computed independently
        # per head and per batch element.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)

        # Apply RoPE to Q and K only (never V, which carries content, not
        # position, into the weighted sum). cos/sin are (T, d_head) and
        # broadcast over the (B, h) dimensions.
        cos, sin = self.rope(T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Scaled dot-product attention scores: (B, h, T, T).
        # Scaling by 1/sqrt(d_head) keeps the dot-product magnitudes
        # roughly unit-variance regardless of d_head -- without it, larger
        # head dimensions would push scores to extreme values, driving the
        # softmax into near-one-hot outputs and starving gradients.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask: additive -inf on future (query, key) positions so
        # softmax assigns them exactly zero probability -- position t can
        # only attend to positions <= t.
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values: (B, h, T, T) @ (B, h, T, d_head) -> (B, h, T, d_head).
        out = attn @ v

        # Concatenate heads back together: (B, h, T, d_head) -> (B, T, h, d_head) -> (B, T, d_model).
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        if return_attn:
            return out, attn
        return out
