"""Generation-quality evaluation (Phase 2, section 2.3 "Generation
quality"): greedy + temperature-sampled continuations from held-out
prefixes, scored against reference continuations with BLEU-4, chrF++,
and ROUGE-L, plus fluency/diversity diagnostics (repetition rate,
Distinct-1/2).

BLEU uses sacrebleu with tokenize="none" (whitespace-only splitting):
Hindi and Nepali are written with space-delimited words like most
languages sacrebleu's built-in tokenizers target, but those tokenizers
assume Latin-script punctuation conventions, so "none" (treat the
SentencePiece-decoded text as already tokenized on whitespace) avoids
mis-tokenizing Devanagari punctuation. chrF operates on character
n-grams by construction, which is inherently script-agnostic and is why
it is generally the more informative metric for non-Latin-script,
open-ended generation -- BLEU and ROUGE-L are included because the
assignment requires them, but their word-overlap basis is a weaker
signal here than for well-tokenized English text (see the discussion
this module's caller writes into the report).
"""

import sacrebleu
from rouge_score import rouge_scorer


def generate_continuations(model, sp, prefixes, max_new_tokens, device, temperature=1.0, greedy=False):
    """prefixes: list of raw text strings. Returns list of decoded
    continuation strings (prefix included, matching what the model
    actually saw as context).
    """
    import torch
    outputs = []
    for prefix in prefixes:
        ids = sp.encode(prefix, out_type=int)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, greedy=greedy)
        text = sp.decode(out_ids[0].tolist())
        outputs.append(text)
    return outputs


def compute_bleu_chrf_rouge(hypotheses, references):
    """hypotheses, references: parallel lists of strings (corpus-level).
    Returns BLEU-4 (sacrebleu default n-gram order), chrF++, and
    corpus-averaged ROUGE-L F1.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="none")
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)  # chrF++ (word_order=2)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    return {
        "bleu_4": bleu.score,
        "chrf_plus_plus": chrf.score,
        "rouge_l_f1": rouge_l,
    }


def repetition_rate(text, n=4):
    """Fraction of overlapping word n-grams that are repeats of an
    earlier n-gram in the same text (1 - unique/total); higher means
    more degenerate/repetitive generation.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def distinct_n(texts, n):
    """Distinct-n (Li et al., 2016): unique n-grams / total n-grams,
    computed over the WHOLE collection of generated texts together (not
    averaged per-sentence) -- the standard corpus-level diversity metric.
    """
    total = 0
    unique = set()
    for text in texts:
        words = text.split()
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def diversity_diagnostics(generated_texts):
    return {
        "mean_repetition_rate_4gram": sum(repetition_rate(t) for t in generated_texts) / len(generated_texts),
        "distinct_1": distinct_n(generated_texts, 1),
        "distinct_2": distinct_n(generated_texts, 2),
    }
