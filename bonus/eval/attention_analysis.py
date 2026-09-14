"""Attention analysis (Phase 2, section 2.3 "Attention analysis"):
extracting per-layer, per-head attention weights, and computing entropy
and mean attention distance from them. Pure computation -- no plotting
here, kept in eval/plotting.py per the assignment's guideline to keep
visualization and computation logic in separate functions.
"""

import torch


@torch.no_grad()
def get_attention_weights(model, sp, text, device):
    """Run one example sentence through the model with return_attn=True
    and return (pieces, attn_weights_per_layer): `pieces` is the list of
    SentencePiece string tokens (for labeling heatmap axes), and
    `attn_weights_per_layer` is a list of length n_layers, each entry a
    (n_heads, T, T) tensor of post-softmax attention weights (batch
    dimension squeezed out, since this is always a single example).
    """
    model.eval()
    ids = sp.encode(text, out_type=int)
    pieces = sp.encode(text, out_type=str)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, attn_per_layer = model(x, return_attn=True)
    attn_per_layer = [layer_attn[0] for layer_attn in attn_per_layer]  # drop batch dim -> (n_heads, T, T)
    return pieces, attn_per_layer


def attention_entropy(attn_per_layer):
    """attn_per_layer: list of (n_heads, T, T) tensors. Returns a
    (n_layers, n_heads) array of the average per-query entropy (in bits)
    of each head's attention distribution over keys -- higher entropy
    means a head spreads attention more diffusely across positions;
    lower entropy means it concentrates on a small number of keys.

    Averaged only over VALID queries/keys (query position t only has t+1
    unmasked keys under the causal mask, so entropy is computed over
    just those, not the full T-length row where masked entries are
    exactly zero).
    """
    n_layers = len(attn_per_layer)
    n_heads = attn_per_layer[0].shape[0]
    entropy_grid = torch.zeros(n_layers, n_heads)

    for layer_idx, attn in enumerate(attn_per_layer):
        T = attn.shape[-1]
        for head_idx in range(n_heads):
            per_query_entropies = []
            for t in range(T):
                p = attn[head_idx, t, :t + 1]  # only the t+1 valid (unmasked) positions
                p = p[p > 0]  # guard against log(0); masked/near-zero entries contribute ~0 anyway
                ent = -(p * torch.log2(p)).sum()
                per_query_entropies.append(ent)
            entropy_grid[layer_idx, head_idx] = torch.stack(per_query_entropies).mean()
    return entropy_grid.numpy()


def mean_attention_distance(attn_per_layer):
    """attn_per_layer: list of (n_heads, T, T) tensors. Returns a
    (n_layers, n_heads) array: for each head, the attention-weighted
    average |query_position - key_position| distance, averaged over all
    query positions -- a proxy for whether a head is "local" (small mean
    distance, attends mostly to nearby positions) or "long-range" (large
    mean distance).
    """
    n_layers = len(attn_per_layer)
    n_heads = attn_per_layer[0].shape[0]
    distance_grid = torch.zeros(n_layers, n_heads)

    for layer_idx, attn in enumerate(attn_per_layer):
        T = attn.shape[-1]
        # Must live on the same device as `attn` (cuda during a real GPU
        # run) -- torch.arange defaults to CPU, and combining it with a
        # GPU tensor below via elementwise multiplication raises
        # "Expected all tensors to be on the same device" otherwise.
        positions = torch.arange(T, device=attn.device)
        for head_idx in range(n_heads):
            per_query_distances = []
            for t in range(T):
                p = attn[head_idx, t, :t + 1]
                keys = positions[:t + 1]
                dist = (p * (t - keys).abs().float()).sum()
                per_query_distances.append(dist)
            distance_grid[layer_idx, head_idx] = torch.stack(per_query_distances).mean()
    return distance_grid.numpy()
