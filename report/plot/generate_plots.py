"""Generate the Phase 1 report figures for Hindi and Nepali from the actual
pipeline statistics on disk (hindi/statistics/*.json, nepali/statistics/*.json)
-- no fabricated numbers, every plot is a direct visualization of a JSON
already produced by preprocess.py / train_tokenizer.py / finalize_tokenizer.py.

Computation (loading/aggregating the stats) and visualization (matplotlib
calls) are kept in separate functions per plot, per the project's guidelines.
Run from the repo root: `python3 report/plot/generate_plots.py`.
"""

import json
import os

import matplotlib.pyplot as plt

# Devanagari script (Hindi/Nepali token labels) needs a font that covers it;
# matplotlib's default DejaVu Sans does not. Noto Sans Devanagari in turn has
# no Latin glyphs, so English titles/labels would break with it alone -- list
# both and let matplotlib (3.6+) fall back per-glyph to whichever covers it.
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Devanagari"]

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.dirname(__file__)
LANGS = ["hindi", "nepali"]
COLORS = {"hindi": "#d95f02", "nepali": "#1b9e77"}


def load_json(language, name):
    path = os.path.join(REPO_ROOT, language, "statistics", f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_corpus_composition():
    """Return {language: {"downloaded": tokens, "manual": tokens}} from
    preprocessing_stats.json."""
    out = {}
    for lang in LANGS:
        stats = load_json(lang, "preprocessing_stats")
        out[lang] = {
            "downloaded": stats["downloaded_chars"] / stats["chars_per_token_assumption"],
            "manual": stats["manual_chars"] / stats["chars_per_token_assumption"],
        }
    return out


def plot_corpus_composition(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(LANGS))
    width = 0.6
    downloaded = [data[lang]["downloaded"] / 1e6 for lang in LANGS]
    manual = [data[lang]["manual"] / 1e6 for lang in LANGS]
    ax.bar(x, downloaded, width, label="Downloaded (IndicCorp)", color="#7570b3")
    ax.bar(x, manual, width, bottom=downloaded, label="Manual (scraped)", color="#e7298a")
    for i, lang in enumerate(LANGS):
        total = downloaded[i] + manual[i]
        pct = manual[i] / total * 100
        ax.text(i, total + 5, f"{total:.1f}M tokens\n({pct:.1f}% manual)",
                 ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lang.capitalize() for lang in LANGS])
    ax.set_ylabel("Tokens (millions)")
    ax.set_title("Phase 1 final corpus composition: downloaded vs. manual tokens")
    ax.legend()
    ax.set_ylim(0, 560)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_fertility_sweep():
    """Return {language: (vocab_sizes, fertilities)} from tokenizer_stats.json."""
    out = {}
    for lang in LANGS:
        stats = load_json(lang, "tokenizer_stats")
        candidates = sorted(stats["sweep_candidates"], key=lambda r: r["vocab_size_requested"])
        vocab_sizes = [c["vocab_size_requested"] for c in candidates]
        fertilities = [c["fertility_tokens_per_word"] for c in candidates]
        out[lang] = (vocab_sizes, fertilities, stats["selected_vocab_size"])
    return out


def plot_fertility_sweep(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        vocab_sizes, fertilities, selected = data[lang]
        ax.plot(vocab_sizes, fertilities, marker="o", label=lang.capitalize(),
                 color=COLORS[lang])
        sel_idx = vocab_sizes.index(selected)
        ax.scatter([vocab_sizes[sel_idx]], [fertilities[sel_idx]], s=140,
                    facecolors="none", edgecolors=COLORS[lang], linewidths=2, zorder=5)
    ax.annotate("selected (10,000)", xy=(10000, data["hindi"][1][data["hindi"][0].index(10000)]),
                 xytext=(12500, 1.55), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Fertility (tokens / word) on held-out val split")
    ax.set_title("Tokenizer vocab-size sweep: fertility vs. vocabulary size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_top_tokens(language, top_n=20):
    stats = load_json(language, "tokenizer_stats")
    tokens = stats["token_frequency_stats"]["top_tokens"][:top_n]
    return [t["piece"] for t in tokens], [t["count"] for t in tokens]


def plot_top_tokens(language, pieces, counts, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    y = range(len(pieces))
    ax.barh(y, counts, color=COLORS[language])
    ax.set_yticks(list(y))
    ax.set_yticklabels(pieces, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Occurrence count (200,000-line sample of train.txt)")
    ax.set_title(f"Top 20 most frequent tokens — {language.capitalize()} (10,000-vocab tokenizer)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_split_sizes():
    """Return {language: {"train": tokens, "val": tokens, "test": tokens}}."""
    out = {}
    for lang in LANGS:
        stats = load_json(lang, "preprocessing_stats")
        cpt = stats["chars_per_token_assumption"]
        out[lang] = {name: s["chars"] / cpt for name, s in stats["splits"].items()}
    return out


def plot_split_sizes(data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    for ax, lang in zip(axes, LANGS):
        splits = data[lang]
        names = ["train", "val", "test"]
        values = [splits[n] / 1e6 for n in names]
        ax.bar(names, values, color=["#66a61e", "#e6ab02", "#a6761d"])
        ax.set_title(lang.capitalize())
        ax.set_ylabel("Tokens (millions)")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}M", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Train / validation / test split sizes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_corpus_composition(compute_corpus_composition(),
                             os.path.join(OUT_DIR, "corpus_composition.png"))
    plot_fertility_sweep(compute_fertility_sweep(),
                          os.path.join(OUT_DIR, "fertility_vs_vocab_size.png"))
    for lang in LANGS:
        pieces, counts = compute_top_tokens(lang)
        plot_top_tokens(lang, pieces, counts, os.path.join(OUT_DIR, f"top20_tokens_{lang}.png"))
    plot_split_sizes(compute_split_sizes(), os.path.join(OUT_DIR, "split_sizes.png"))
    print("Plots written to", OUT_DIR)
