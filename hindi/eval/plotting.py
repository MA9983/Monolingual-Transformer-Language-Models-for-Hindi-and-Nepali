"""All Phase 2 visualization code, kept separate from the computation
functions in lm_metrics.py / generation_metrics.py / attention_analysis.py
per the assignment's guideline. Every plot sets a title, axis labels, and
a legend where applicable (plots missing these may score zero for that
component, per the guidelines).
"""

import os
import urllib.request

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# The attention heatmap's axis tick labels are Devanagari token pieces
# (required: "include example sentences in the model's language"), but
# matplotlib's default font (DejaVu Sans) has no Devanagari glyphs, so
# those labels would silently render as empty boxes. Rather than assume
# a Devanagari-capable font happens to be preinstalled wherever this runs
# (this machine has one; Kaggle's base image is not guaranteed to), look
# for one in a list of common locations, and fall back to downloading
# Noto Sans Devanagari once into a local cache if none is found.
_DEVANAGARI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/truetype/mangal/Mangal.ttf",
    os.path.expanduser("~/.fonts/NotoSansDevanagari-Regular.ttf"),
]
_DEVANAGARI_FONT_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "phase2_fonts", "NotoSansDevanagari-Regular.ttf")
_DEVANAGARI_FONT_URL = (
    "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/"
    "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
)


def _find_or_fetch_devanagari_font():
    for path in _DEVANAGARI_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    if os.path.exists(_DEVANAGARI_FONT_CACHE):
        return _DEVANAGARI_FONT_CACHE
    try:
        os.makedirs(os.path.dirname(_DEVANAGARI_FONT_CACHE), exist_ok=True)
        urllib.request.urlretrieve(_DEVANAGARI_FONT_URL, _DEVANAGARI_FONT_CACHE)
        return _DEVANAGARI_FONT_CACHE
    except Exception as e:  # noqa: BLE001 -- no internet, blocked, etc.: degrade, don't crash plotting
        print(f"[plotting] warning: could not obtain a Devanagari font ({e}); "
              f"Devanagari heatmap labels may render as missing-glyph boxes.", flush=True)
        return None


_devanagari_font_path = _find_or_fetch_devanagari_font()
if _devanagari_font_path:
    fm.fontManager.addfont(_devanagari_font_path)
    DEVANAGARI_FONT = fm.FontProperties(fname=_devanagari_font_path)
else:
    DEVANAGARI_FONT = None


def plot_loss_curves(loss_history, val_loss_history, language, out_path):
    """loss_history / val_loss_history: list of (step, loss) tuples."""
    steps, losses = zip(*loss_history)
    val_steps, val_losses = zip(*val_loss_history)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, label="Train loss", alpha=0.6, linewidth=1)
    ax.plot(val_steps, val_losses, label="Validation loss", linewidth=2, marker="o", markersize=3)
    ax.set_title(f"{language.capitalize()} pretraining loss curve")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention_heatmap(attn_weights, pieces, layer_idx, head_idx, language, out_path):
    """attn_weights: (T, T) numpy array for one specific layer/head.
    pieces: list of T SentencePiece string tokens, used as axis tick
    labels so the heatmap is readable against real text.
    """
    # SentencePiece's "▁" word-boundary marker (U+2581) has no glyph in
    # Devanagari fonts and would render as a tofu box; swap it for a
    # plain space, which reads naturally as "this piece starts a new word".
    display_pieces = [p.replace("▁", " ") for p in pieces]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_weights, cmap="viridis", aspect="auto")
    ax.set_title(f"{language.capitalize()}: attention weights, layer {layer_idx}, head {head_idx}")
    ax.set_xlabel("Key position (token attended to)")
    ax.set_ylabel("Query position (attending token)")
    ax.set_xticks(range(len(display_pieces)))
    ax.set_xticklabels(display_pieces, rotation=90, fontsize=9, fontproperties=DEVANAGARI_FONT)
    ax.set_yticks(range(len(display_pieces)))
    ax.set_yticklabels(display_pieces, fontsize=9, fontproperties=DEVANAGARI_FONT)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_entropy_grid(entropy_grid, language, out_path):
    """entropy_grid: (n_layers, n_heads) numpy array."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(entropy_grid, cmap="magma", aspect="auto")
    ax.set_title(f"{language.capitalize()}: attention entropy per layer/head")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(entropy_grid.shape[1]))
    ax.set_yticks(range(entropy_grid.shape[0]))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean entropy (bits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distance_grid(distance_grid, language, out_path):
    """distance_grid: (n_layers, n_heads) numpy array."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(distance_grid, cmap="cividis", aspect="auto")
    ax.set_title(f"{language.capitalize()}: mean attention distance per layer/head")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(distance_grid.shape[1]))
    ax.set_yticks(range(distance_grid.shape[0]))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |query - key| distance (positions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ppl_bpb_comparison(hindi_metrics, nepali_metrics, out_path):
    """hindi_metrics/nepali_metrics: dicts with 'perplexity' and
    'bits_per_byte' keys (see lm_metrics.compute_ppl_bpb).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    languages = ["Hindi", "Nepali"]

    axes[0].bar(languages, [hindi_metrics["perplexity"], nepali_metrics["perplexity"]],
                color=["#4C72B0", "#DD8452"], label="Perplexity")
    axes[0].set_title("Validation perplexity")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Perplexity (lower is better)")
    axes[0].legend()

    axes[1].bar(languages, [hindi_metrics["bits_per_byte"], nepali_metrics["bits_per_byte"]],
                color=["#4C72B0", "#DD8452"], label="Bits-per-byte")
    axes[1].set_title("Validation bits-per-byte (BPB)")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Bits per byte (lower is better)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
