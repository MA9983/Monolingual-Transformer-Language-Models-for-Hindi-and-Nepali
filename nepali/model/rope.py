"""Rotary Position Embedding (RoPE; Su et al., 2021 -- "RoFormer"),
Phase 2's chosen positional scheme, applied to query and key vectors
inside every attention layer (see attention.py) rather than added to the
token embedding.

Core idea: rotate each (query or key) vector, in consecutive pairs of
its dimensions, by an angle proportional to its sequence position. Two
vectors rotated this way have a dot product that depends only on the
*difference* between their positions, not their absolute positions --
i.e. relative position falls directly out of the attention score
computation itself, at every layer, rather than being a hint added once
at the input and left for the network to rediscover indirectly in deeper
layers.

Implemented with the standard real-valued "rotate half" trick (as in
GPT-NeoX / Llama) rather than complex-number arithmetic -- mathematically
equivalent, but avoids needing complex tensor support.
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_head, max_len, base=10000):
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires an even head dimension (pairs of dimensions are rotated together)"
        self.d_head = d_head
        self.max_len = max_len

        # One rotation frequency per pair of dimensions: (d_head/2,).
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))

        positions = torch.arange(max_len, dtype=torch.float32)  # (max_len,)
        freqs = torch.outer(positions, inv_freq)  # (max_len, d_head/2)
        # Duplicate along the last axis so cos/sin align with the full
        # d_head width under the "rotate half" pairing (dim i pairs with
        # dim i + d_head/2), rather than adjacent-pair interleaving.
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, d_head)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # (max_len, d_head)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # (max_len, d_head)

    def forward(self, seq_len):
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds RoPE's max_len={self.max_len} "
                f"(== the model's trained context_length)"
            )
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]  # each (seq_len, d_head)


def _rotate_half(x):
    """x: (..., d_head). Splits into two halves [x1, x2] along the last
    dimension and returns [-x2, x1] -- the real-valued equivalent of
    multiplying by i (the imaginary unit) in the complex-number
    formulation of RoPE.
    """
    d_head = x.shape[-1]
    x1, x2 = x[..., :d_head // 2], x[..., d_head // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """x: (B, n_heads, T, d_head). cos/sin: (T, d_head), broadcast over
    the batch and head dimensions. Returns x rotated by the position-
    dependent angle for each of its T positions:
        rotated = x * cos + rotate_half(x) * sin
    which is the real-valued form of multiplying x (viewed as d_head/2
    complex numbers) by e^{i * position * theta}.
    """
    return x * cos + _rotate_half(x) * sin
