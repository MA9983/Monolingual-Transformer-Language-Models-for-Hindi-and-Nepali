# Phase 1 — Hindi (Model H) Report

**Role:** Model H — higher-resource language. Chosen for its large public corpora (IndicCorp,
Wikipedia, OSCAR/CC-100) and abundant online news/literature text. Full cross-language
methodology, site-vetting policy, and comparison with Nepali are in
[`phase1_report.md`](phase1_report.md); this document is the per-language deep dive for
Hindi, structured around the assignment's Phase 1 reporting requirements.

## 1. Dataset summary

| | Chars | Tokens |
|---|---:|---:|
| Downloaded (IndicCorp, capped) | 1,800,000,073 | 400,000,016 |
| Manual (news scrape, used in full) | 480,632,418 | 106,807,204 |
| **Total (final, post-cleaning)** | **2,280,632,491** | **506,807,220** |

**Manual fraction: 21.07%** — meets the assignment's ≥20% requirement.

Cleaning pipeline (`common/preprocess_corpus.py`, run independently of Nepali — no shared
data, state, or vocabulary): Unicode NFC normalization → Devanagari-script quality filtering
→ repetitive-document removal (Gopher-style top-n-gram-ratio heuristic) → profanity safety
filter (`common/badwords_hindi.txt`, exact-whole-word matching) → content-hash exact
deduplication → MinHash/LSH near-duplicate deduplication (manual source only). Combined
cleaning/dedup/safety loss across both sources was **~1.50%** of scanned text. Near-dedup was
the biggest new contributor: **1,299 news records (~580.8k tokens, 0.53% of scanned manual
records)** were near-duplicates (e.g. syndicated/republished articles) that exact-hash
matching had missed entirely. The safety filter remains negligible (1,820 records total,
<0.03% of scanned records). A from-scratch bigram-LM perplexity filter was also implemented
and tested, then deliberately removed before these final numbers. Full raw-vs-kept tables,
per-source breakdowns, and the manual-source site list are in `phase1_report.md` ("Data
cleaning and deduplication — Hindi").

| Split | Records | Chars | Target ratio | Actual ratio |
|---|---:|---:|---:|---:|
| `train` | 5,973,084 | 2,166,767,969 | 95% | 95.01% |
| `val` | 156,611 | 56,840,593 | 2.5% | 2.49% |
| `test` | 157,515 | 57,023,929 | 2.5% | 2.50% |

Split assignment is a deterministic hash of each record's own cleaned content, so
re-running preprocessing is reproducible without a materialized shuffle. Outputs:
`hindi/data/splits/{train,val,test}.txt` and `hindi/data/processed/cleaned_corpus.txt` (a
combined pre-split audit artifact; line count verified to exactly equal the sum of the three
splits: 6,287,210).

![Final corpus composition and split sizes](plot/corpus_composition.png)

## 2. Tokenizer

**Retrained on the current splits** (§1's numbers above), after repetitive-document removal
and near-duplicate removal were added to the cleaning pipeline — the corpus shifted well
under 1% from the previous run, so figures below are close to before but now current.

Trained from scratch with SentencePiece BPE (`common/train_tokenizer.py`,
`hindi/tokenizer/train_tokenizer.py`) on `hindi/data/splits/train.txt` only; vocab size chosen
by sweeping candidates and evaluating each on the held-out `val.txt` split (`test.txt` stays
untouched for later phases). Pretrained tokenizers were not used and this vocabulary is not
shared with Nepali's.

### 2.1 Vocabulary size

Candidates swept: 5,000 / 7,000 / 9,000 / 10,000 / 16,000 / 24,000 — the 5k–10k band per TA
guidance, plus two larger points for a wider comparison.

| Vocab size | Fertility (tok/word) | Avg chars/token | UNK rate |
|---:|---:|---:|---:|
| 5,000 | 1.4839 | 3.437 | 0.0% |
| 7,000 | 1.3960 | 3.653 | 0.0% |
| 9,000 | 1.3421 | 3.800 | 0.0% |
| **10,000 (selected)** | **1.3227** | **3.856** | **0.0%** |
| 16,000 | 1.2509 | 4.077 | 0.0% |
| 24,000 | 1.2067 | 4.226 | 0.0% |

**Selected vocabulary size: 10,000.** Because `byte_fallback=True` gives every candidate a
0.0% UNK rate, "lowest fertility subject to the UNK bar" would mechanically pick the largest
candidate tested (24,000) — not a real trade-off. Fertility improvement per +1,000 vocab
falls off sharply (0.044 at 5k→7k down to 0.006 at 16k→24k, an ~8x drop), and the token
embedding table (`vocab_size × d_model`) competes directly against the assignment's
~25M-parameter model budget: at `d_model=256`, 10,000 costs ~10–20% of that budget versus
~25–49% at 24,000. 10,000 captures 58% of the total available fertility improvement using
only 26% of the tested vocab-size range — the best point on that trade-off curve, and it sits
at the top of the TA-recommended band. Full derivation in `phase1_report.md`
("Tokenizer training — Hindi").

![Fertility vs. vocabulary size sweep, with the selected point circled](plot/fertility_vs_vocab_size.png)

The final model (`hindi/tokenizer/hindi_tokenizer.model` / `.vocab`) was retrained at this
vocab size (`common/finalize_tokenizer.py`) so the shipped artifact matches the recommendation.

### 2.2 Token-frequency statistics

Measured on a 200,000-line sample of `train.txt` (15,438,061 tokens): **9,860 of 10,000**
vocabulary entries (98.6%) were used at least once. The top 100 tokens account for **39.3%**
of all token occurrences — a small set of function words/postpositions and punctuation
dominates, as expected for a postposition-heavy language.

| Rank | Piece | Meaning | Count | Share of all tokens |
|---:|---|---|---:|---:|
| 1 | `▁के` | postposition ("of") | 489,670 | 3.17% |
| 2 | `।` | danda (sentence-end) | 468,857 | 3.04% |
| 3 | `▁में` | "in" | 370,314 | 2.40% |
| 4 | `▁है` | "is" | 311,389 | 2.02% |
| 5 | `▁की` | postposition ("of", fem.) | 302,111 | 1.96% |
| 6 | `,` | comma | 247,519 | 1.60% |
| 7 | `▁को` | postposition (dative/accusative) | 231,889 | 1.50% |
| 8 | `▁से` | "from/by/with" | 219,697 | 1.42% |
| 9 | `▁ने` | ergative postposition | 168,541 | 1.09% |
| 10 | `▁का` | postposition ("of", masc.) | 166,112 | 1.08% |
| 11 | `▁और` | "and" | 155,157 | 1.01% |
| 12 | `▁पर` | "on/but" | 142,397 | 0.92% |
| 13 | `▁हैं` | "are" | 115,751 | 0.75% |
| 14 | `▁कि` | "that" (conjunction) | 114,897 | 0.74% |
| 15 | `.` | period | 113,740 | 0.74% |
| 16 | `▁भी` | "also" | 93,401 | 0.61% |
| 17 | `▁लिए` | "for" | 76,980 | 0.50% |
| 18 | `▁एक` | "one/a" | 71,415 | 0.46% |
| 19 | `▁कर` | verb stem ("do") | 71,179 | 0.46% |
| 20 | `▁इस` | "this" (oblique) | 68,564 | 0.44% |

![Top 20 most frequent tokens, Hindi 10,000-vocab tokenizer](plot/top20_tokens_hindi.png)

The entire top 20 is grammatical function words and punctuation rather than content words —
content words appear much further down the frequency distribution, consistent with Zipf's law.

### 2.3 Average characters per token

**3.887 characters/token**, measured on the full 156,611-line `val.txt` split
(14,623,086 tokens total) — i.e. each subword piece the tokenizer emits corresponds to just
under 4 raw Devanagari/punctuation characters on average.

### 2.4 Tokenization examples

| Text | Pieces | Piece count |
|---|---|---:|
| भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं। | ▁भारत / ▁एक / ▁विशाल / ▁देश / ▁है / ▁जहाँ / ▁अनेक / ▁भाषा / एँ / ▁बोली / ▁जाती / ▁हैं / । | 13 |
| प्रधानमंत्री ने आज नई दिल्ली में एक महत्वपूर्ण बैठक की अध्यक्षता की। | ▁प्रधानमंत्री / ▁ने / ▁आज / ▁नई / ▁दिल्ली / ▁में / ▁एक / ▁महत्वपूर्ण / ▁बैठक / ▁की / ▁अध्यक्षता / ▁की / । | 13 |
| क्रिकेट विश्व कप के फाइनल मैच में भारत ने शानदार जीत दर्ज की। | ▁क्रिकेट / ▁विश्व / ▁कप / ▁के / ▁फाइनल / ▁मैच / ▁में / ▁भारत / ▁ने / ▁शानदार / ▁जीत / ▁दर्ज / ▁की / । | 14 |

Most content words tokenize as a single whole-word piece even at 10k vocab (भारत,
प्रधानमंत्री, दिल्ली, क्रिकेट); the one visible split (▁भाषा + एँ) shows the expected BPE
fallback to a stem + suffix decomposition for a less frequent inflected form.

### 2.5 Unknown-token statistics

**UNK rate: 0.0% at every candidate vocab size tested (5k–24k), and 0.0% on the final
10,000-vocab model's full validation-split evaluation** (156,611 lines / 14,623,086 tokens).
This is guaranteed by `byte_fallback=True`: SentencePiece reserves 256 byte-level fallback
pieces (`<0x00>`–`<0xFF>`) independent of vocab size, so any character never seen in training
is spelled out as raw UTF-8 bytes instead of emitting a true `<unk>`. Because this holds at
every candidate, UNK rate provided no discriminating signal in the vocab-size sweep — fertility
did that work instead (see §2.1).
