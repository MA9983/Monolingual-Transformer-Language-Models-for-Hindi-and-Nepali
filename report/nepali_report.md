# Phase 1 — Nepali (Model L) Report

**Role:** Model L — lower-resource language, chosen from the assignment's allowed
lower-resource list. Genuinely resource-constrained online relative to Hindi: manual
collection required substantially more source-hunting effort (more sites, plus a Wikipedia
crawl) to reach a comparable token count. Full cross-language methodology, site-vetting
policy, and comparison with Hindi are in [`phase1_report.md`](phase1_report.md); this document
is the per-language deep dive for Nepali, structured around the assignment's Phase 1 reporting
requirements.

## 1. Dataset summary

| | Chars | Tokens |
|---|---:|---:|
| Downloaded (IndicCorp, capped) | 1,800,000,113 | 400,000,025 |
| Manual — news (used in full) | 407,414,663 | 90,536,592 |
| Manual — Wikipedia (used in full) | 43,670,965 | 9,704,659 |
| **Manual subtotal** | **451,085,628** | **100,241,251** |
| **Total (final, post-cleaning)** | **2,251,085,741** | **500,241,276** |

**Manual fraction: 20.04%** — meets the assignment's ≥20% requirement, and the total lands
almost exactly on the ~500M-token target. This margin has narrowed from an earlier 20.41%
after near-duplicate removal found real duplicate content on the manual side (see below) —
still comfortably compliant, but closer to the 20% floor than before.

Cleaning pipeline (`common/preprocess_corpus.py`, run independently of Hindi — no shared
data, state, or vocabulary): Unicode NFC normalization → Devanagari-script quality filtering
→ repetitive-document removal (Gopher-style top-n-gram-ratio heuristic) → profanity safety
filter (`common/badwords_nepali.txt`, exact-whole-word matching) → content-hash exact
deduplication → MinHash/LSH near-duplicate deduplication (both manual sources). Combined
cleaning/dedup/safety loss was **~2.85%** across all three sources. Wikipedia lost
proportionally the most (**31.99%** total) — still dominated by the length/quality filter
(short stubs/disambiguation/list pages), but now with a meaningful near-duplicate
contribution too: **1,653 records (~245.4k tokens, ~6.2% of candidate Wikipedia records)**
were near-duplicates that exact-hash matching missed. News scraping shows the same pattern
more moderately: **6,760 near-duplicate records (~2.06M tokens, ~2.9% of scanned news
records)**. The safety filter remains negligible (92 records total across all sources). A
from-scratch bigram-LM perplexity filter was also implemented and tested, then deliberately
removed before these final numbers. Full raw-vs-kept tables, per-source breakdowns, and the
manual-source site list are in `phase1_report.md` ("Data cleaning and deduplication —
Nepali").

| Split | Records | Chars | Target ratio | Actual ratio |
|---|---:|---:|---:|---:|
| `train` | 7,876,088 | 2,138,614,328 | 95% | 95.00% |
| `val` | 207,528 | 56,231,353 | 2.5% | 2.50% |
| `test` | 207,340 | 56,240,060 | 2.5% | 2.50% |

Split assignment is a deterministic hash of each record's own cleaned content, so
re-running preprocessing is reproducible without a materialized shuffle. Outputs:
`nepali/data/splits/{train,val,test}.txt` and `nepali/data/processed/cleaned_corpus.txt` (a
combined pre-split audit artifact; line count verified to exactly equal the sum of the three
splits: 8,290,956).

![Final corpus composition and split sizes](plot/corpus_composition.png)

## 2. Tokenizer

**Retrained on the current splits** (§1's numbers above), after repetitive-document removal
and near-duplicate removal were added to the cleaning pipeline — figures below are close to
before but now current.

Trained from scratch with SentencePiece BPE (`common/train_tokenizer.py`,
`nepali/tokenizer/train_tokenizer.py`) on `nepali/data/splits/train.txt` only; vocab size
chosen by sweeping candidates and evaluating each on the held-out `val.txt` split (`test.txt`
stays untouched for later phases). Pretrained tokenizers were not used and this vocabulary is
not shared with Hindi's.

### 2.1 Vocabulary size

Same candidate range as Hindi: 5,000 / 7,000 / 9,000 / 10,000 / 16,000 / 24,000.

| Vocab size | Fertility (tok/word) | Avg chars/token | UNK rate |
|---:|---:|---:|---:|
| 5,000 | 1.7252 | 3.738 | 0.0% |
| 7,000 | 1.6005 | 4.029 | 0.0% |
| 9,000 | 1.5224 | 4.236 | 0.0% |
| **10,000 (selected)** | **1.4934** | **4.318** | **0.0%** |
| 16,000 | 1.3848 | 4.657 | 0.0% |
| 24,000 | 1.3126 | 4.913 | 0.0% |

**Nepali's fertility is higher than Hindi's at every matched vocab size** (e.g. 1.493 vs.
1.323 at 10,000) — more subword pieces are needed per word for the same vocabulary budget,
consistent with Nepali's richer suffix-chaining morphology (case/postposition markers fuse
onto stems rather than staying separate words) and its smaller, more source-diverse corpus.

**Selected vocabulary size: 10,000.** Same reasoning as Hindi: `byte_fallback=True` gives
every candidate 0.0% UNK, so the naive "lowest fertility" rule would mechanically pick the
largest candidate (24,000) rather than reflecting a real trade-off. Fertility-improvement
rate per +1,000 vocab falls from 0.062 (5k→7k) to 0.009 (16k→24k), a ~7x drop, and the
embedding-table cost against the ~25M-parameter budget is identical to Hindi's (10,000 ≈
10–20% of budget vs. ~25–49% at 24,000). 10,000 captures 56% of the total available fertility
improvement using only 26% of the tested vocab-size range and sits at the top of the
TA-recommended band. Full derivation in `phase1_report.md` ("Tokenizer training — Nepali").

![Fertility vs. vocabulary size sweep, with the selected point circled](plot/fertility_vs_vocab_size.png)

The final model (`nepali/tokenizer/nepali_tokenizer.model` / `.vocab`) was retrained at this
vocab size (`common/finalize_tokenizer.py`) so the shipped artifact matches the recommendation.

### 2.2 Token-frequency statistics

Measured on a 200,000-line sample of `train.txt` (10,330,982 tokens): **9,880 of 10,000**
vocabulary entries (98.8%) were used at least once. The top 100 tokens account for only
**27.5%** of all occurrences — notably flatter than Hindi's 39.3%, i.e. Nepali's token usage
is more evenly spread rather than dominated by a small set of function words.

| Rank | Piece | Meaning | Count | Share of all tokens |
|---:|---|---|---:|---:|
| 1 | `▁।` | danda w/ leading space (sentence-end) | 258,230 | 2.50% |
| 2 | `,` | comma | 190,627 | 1.85% |
| 3 | `।` | danda (sentence-end) | 146,936 | 1.42% |
| 4 | `▁छ` | "is/exists" (copula) | 133,130 | 1.29% |
| 5 | `▁र` | "and" | 133,100 | 1.29% |
| 6 | `को` | genitive/postposition suffix | 96,257 | 0.93% |
| 7 | `मा` | locative suffix ("in/at") | 82,001 | 0.79% |
| 8 | `का` | genitive suffix (masc./plural) | 73,238 | 0.71% |
| 9 | `ले` | ergative suffix | 68,563 | 0.66% |
| 10 | `▁पनि` | "also" | 58,745 | 0.57% |
| 11 | `लाई` | dative/accusative suffix | 54,532 | 0.53% |
| 12 | `▁छन्` | "are" (copula, plural) | 45,038 | 0.44% |
| 13 | `▁भएको` | "having become/happened" | 44,383 | 0.43% |
| 14 | `▁लागि` | "for" | 42,571 | 0.41% |
| 15 | `▁गर्न` | "to do" (infinitive) | 39,138 | 0.38% |
| 16 | `▁हो` | "is" (copula) | 37,162 | 0.36% |
| 17 | `▁एक` | "one/a" | 36,233 | 0.35% |
| 18 | `▁गरेको` | "having done" | 35,943 | 0.35% |
| 19 | `▁को` | postposition, standalone form | 34,659 | 0.34% |
| 20 | `बाट` | ablative suffix ("from") | 33,100 | 0.32% |

![Top 20 most frequent tokens, Nepali 10,000-vocab tokenizer](plot/top20_tokens_nepali.png)

Several top-20 pieces are **bound case/postposition suffixes without a leading space marker**
(`को`, `मा`, `का`, `ले`, `लाई`, `बाट`) rather than whole standalone words — a direct signature
of Nepali's suffix-chaining morphology, which Hindi's largely-analytic (mostly separate,
space-delimited) postposition system does not show at the same vocab size.

### 2.3 Average characters per token

**4.315 characters/token**, measured on the full 200,000-line `val.txt` split
(10,382,990 tokens total) — noticeably higher than Hindi's 3.887, consistent with Nepali
needing longer average pieces/more characters absorbed per token despite also having higher
fertility (tokens/word); the two metrics capture different things (fertility normalizes by
whitespace-delimited word, chars/token does not).

### 2.4 Tokenization examples

| Text | Pieces | Piece count |
|---|---|---:|
| नेपाल हिमालयको काखमा रहेको सुन्दर देश हो। | ▁नेपाल / ▁हिमाल / यको / ▁का / ख / मा / ▁रहेको / ▁सुन्दर / ▁देश / ▁हो / । | 11 |
| प्रधानमन्त्रीले आज काठमाडौंमा एक महत्वपूर्ण बैठकको अध्यक्षता गर्नुभयो। | ▁प्रधानमन्त्रीले / ▁आज / ▁काठमाडौंमा / ▁एक / ▁महत्वपूर्ण / ▁बैठकको / ▁अध्यक्ष / ता / ▁गर्नुभयो / । | 10 |
| क्रिकेट विश्वकपको फाइनल खेलमा नेपालले उत्कृष्ट प्रदर्शन गर्यो। | ▁क्रिकेट / ▁विश्वक / पको / ▁फाइनल / ▁खेलमा / ▁नेपालले / ▁उत्कृष्ट / ▁प्रदर्शन / ▁गर्यो / । | 10 |

Several long inflected forms still tokenize as a single whole-word piece, including
case/postposition suffixes fused onto the stem (प्रधानमन्त्रीले, काठमाडौंमा, बैठकको,
नेपालले) — but more stem+suffix splits are visible than in Hindi's examples at the same vocab
size (▁हिमाल+यको+▁का+ख+मा, ▁विश्वक+पको, ▁अध्यक्ष+ता), directly illustrating why Nepali's
fertility (1.49) is meaningfully higher than Hindi's (1.32) at the same 10,000-token budget.

### 2.5 Unknown-token statistics

**UNK rate: 0.0% at every candidate vocab size tested (5k–24k), and 0.0% on the final
10,000-vocab model's full validation-split evaluation** (200,000 lines / 10,382,990 tokens).
This is guaranteed by `byte_fallback=True`: SentencePiece reserves 256 byte-level fallback
pieces (`<0x00>`–`<0xFF>`) independent of vocab size, so any character never seen in training
is spelled out as raw UTF-8 bytes instead of emitting a true `<unk>`. Because this holds at
every candidate, UNK rate provided no discriminating signal in the vocab-size sweep — fertility
did that work instead (see §2.1).
