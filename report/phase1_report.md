# Phase 1 Report — Data Collection and Tokenizer Construction

**Model H (higher-resource):** Hindi
**Model L (lower-resource):** Nepali

This report covers data collection (downloaded + manually scraped) and the completed
cleaning/deduplication/splitting pass for **both languages** (`hindi/data/preprocess.py`,
`nepali/data/preprocess.py`) — see "Data cleaning and deduplication" below for the full
before/after comparison for each. Raw-text character counts are converted to an estimated
token count using a **4.5 characters/token** planning assumption (Devanagari BPE fertility
typically falls in the 4–5 chars/token range; the real figure will be measured once the
tokenizer is trained in a later step).

Per-language deep-dive reports focused on the tokenizer reporting requirement (vocabulary
size, token-frequency statistics, average characters per token, tokenization examples, and
unknown-token statistics) are in [`hindi_report.md`](hindi_report.md) and
[`nepali_report.md`](nepali_report.md).

## Language selection

- **Hindi (Model H):** chosen as the higher-resource language — large public corpora
  (IndicCorp, Wikipedia, OSCAR/CC-100) and abundant online news/literature text.
- **Nepali (Model L):** chosen from the assignment's allowed lower-resource list. Nepali is
  genuinely resource-constrained online relative to Hindi, which is reflected directly in
  the collection numbers below (see "Nepali manual-data ceiling").

## Collection methodology

Two independent sources per language, run through completely separate pipelines (no
shared code state beyond generic utilities, no shared data, no shared vocabulary):

1. **Downloaded**: [`ai4bharat/IndicCorpV2`](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
   (CC-0 licensed), streamed directly from HuggingFace and stopped once the character-based
   token-count target was reached, rather than downloading the full multi-GB source files
   (`common/download_indiccorp.py`).
2. **Manual**: news-site scraping (`common/news_scraper.py`) and, for Nepali only, an
   additional Wikipedia crawl (`common/wikipedia_scraper.py`) — see "Manual collection" below
   for why Wikipedia was added and why it counts as manual collection under the assignment's
   own definition.

Both scrapers are sitemap/RSS-driven, robots.txt-compliant (checked live per request via
`common/robots_utils.py`, not just at site-selection time), rate-limited per site, and
checkpointed so a multi-hour/overnight crawl can be killed and resumed without re-fetching
or double-counting anything.

### Site vetting

Every candidate site's `robots.txt` was checked before use for two things: (a) whether it
disallows AI/anthropic crawlers by name (`anthropic-ai`, `ClaudeBot`, `Claude-Web`), and (b)
whether it states an explicit anti-AI-training policy in plain text or via the
`Content-Signal: ai-train=no` convention. Sites matching either were excluded outright,
regardless of how much archive depth they had. Notable exclusions:

- **bbc.com** — explicit "no AI training / no dataset creation" policy in its robots.txt.
- **bhaskar.com, amarujala.com** — disallow `anthropic-ai` by name.
- **news18.com, zeenews.india.com** — disallow `ClaudeBot`/`Claude-Web`.
- **patrika.com** — disallows both `ClaudeBot` and `anthropic-ai`.
- **haribhoomi.com, hindi.oneindia.com, hindi.webdunia.com, dw.com** — disallow
  `ClaudeBot`/`anthropic-ai`.
- **baahrakhari.com, thahakhabar.com, ratopati.com.np, himalkhabar.com** (Nepali) —
  explicit `Content-Signal: ai-train=no` plus a `ClaudeBot` disallow.
- **karobardaily.com** (Nepali) — disallows `ClaudeBot` despite having a real deep archive.
- A handful of others were dropped for being unreachable or persistently 403ing on
  `robots.txt` even with a browser User-Agent (`gorkhapatraonline.com`, `nepalpress.com`,
  `annapurnapost.com`, `lokaantar.com`, `newsofnepal.com`, `reportersnepal.com`,
  `samaylive.com`, `livehindi.tv`, `jansatta.com`, `khabar.ndtv.com`).

---

## Hindi — raw collection totals (input to preprocessing)

| Source | Estimated tokens | Raw chars |
|---|---:|---:|
| Downloaded (IndicCorp) | 500,078,717 | 2,250,354,226 |
| Manual (news scrape) | 110,304,397 | 496,369,786 |
| **Total (pre-cleaning)** | **610,383,114** | **2,746,724,012** |

**Manual fraction (pre-cleaning): 18.07%** — currently just under the assignment's 20%
floor; see "Manual fraction" note below.

### Hindi manual source breakdown (231,944 articles collected)

| Site | Est. tokens | Notes |
|---|---:|---|
| livehindustan.com | 70,300,656 | Deep dated archive (363 historical sub-sitemaps); main contributor |
| timesnowhindi.com | 25,266,934 | 37 monthly sitemaps back to 2023-08 |
| abplive.com | 7,882,846 | |
| punjabkesari.in | 2,430,740 | 15 declared sitemaps incl. regional editions |
| indiatv.in | 1,388,015 | Removed from the active site list after repeated worker-thread stalls; kept whatever it had already contributed |
| jagran.com | 2,280,880 | Rolling recent-window sitemap, exhausts quickly |
| aajtak.in | 754,326 | |

---

## Data cleaning and deduplication — Hindi (complete)

Cleaning steps applied by `common/preprocess_corpus.py` (shared code, run independently
per language — no shared state, data, or vocabulary between Hindi and Nepali):

1. **Unicode normalization** (NFC) — canonicalizes Devanagari combining-character sequences
   so visually-identical text always has one representation.
2. **Quality / script filtering** (`is_valid_line` in `common/text_cleaning.py`) — drops
   records that are too short, not majority-Devanagari script, or mostly punctuation/digits
   rather than prose.
3. **Repetitive-document removal** (`is_repetitive_text`) — the standard "top n-gram
   repetition ratio" heuristic (Gopher/MassiveText, Rae et al. 2021): among overlapping
   4-word n-grams in a record, if the single most common one accounts for more than 30% of
   all n-gram occurrences, the record is dropped as boilerplate/spam rather than genuine
   prose. Pure frequency counting, no external model.
4. **Safety filter (profanity removal)** — `common/badwords_hindi.txt` (Devanagari, ~40
   terms), matched by exact whole-word lookup after tokenizing each record into
   Devanagari-codepoint runs (`load_badwords_set` / `contains_bad_word` in
   `common/text_cleaning.py`). Any record containing one of these terms as a whole word is
   dropped. Deliberately scoped to explicit sexual/scatological vulgarity only — not a
   general hate-speech/slur filter — because word-list matching against caste, religious, or
   community terms has a very high false-positive rate on legitimate news/encyclopedic
   content (those terms appear constantly in neutral reporting), which would discard good
   data for little safety benefit. The matcher also deliberately avoids naive `\b`-anchored
   regex matching: Python's `\b` treats Devanagari vowel signs (matra) as non-word
   characters, so a `\b`-based match would false-positive on unrelated, entirely benign
   words that merely share a listed term's consonant prefix before their own vowel sign
   (verified directly during development against real examples from the corpus). Exact-
   token-set matching avoids this class of false positive entirely.
5. **Exact deduplication** — content-hash based, using **one shared `Deduplicator` instance
   across the entire language's run**, so duplicates are caught both *within* a source and
   *across* downloaded vs. manual (e.g. a sentence appearing in both IndicCorp and a scraped
   article would only be kept once).
6. **Near-duplicate (MinHash/LSH) deduplication** — `NearDuplicateDetector` in
   `common/text_cleaning.py`: MinHash signatures (num_perm=128) over overlapping 3-word
   shingles, compared via `MinHashLSH` at a 0.85 Jaccard-similarity threshold, catching
   near-identical records (e.g. a wire story republished with a changed byline/date) that
   exact-hash matching cannot, since it requires an identical hash of the whole normalized
   text. Applied only to the manually scraped article source (not the multi-million-line
   downloaded source, where a retained signature per record would be memory-prohibitive at
   that scale and upstream curation already makes near-duplicates rare).

Two separate effects are involved in going from "raw collected" to "final corpus," and the
table below deliberately keeps them apart:

- **Deliberate downsizing**: the downloaded source was capped at ~400M tokens (see "Manual
  fraction" below for why), so most of its raw collected pool (500.08M tokens) was never
  even scanned — this is a size choice, not a quality judgment.
- **Actual cleaning/dedup loss**: within the portion that *was* scanned, how much got
  dropped for being invalid, repetitive, duplicate, near-duplicate, or unsafe.

### Raw (scanned) vs. kept, per source

| Source | Raw scanned | Invalid | Repetitive | Exact dup | Near-dup | Unsafe | **Kept** | **Total reduction** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Downloaded (IndicCorp) | 403,922,909 tok | 2,517,541 tok | 37 tok | 218,715 tok | — | 1,584,552 tok | **400,000,016 tok** | **0.97%** |
| Manual (news scrape) | 110,594,370 tok | 107,452 tok | 0 tok | 44,659 tok | 580,828 tok | 9,696 tok | **106,807,204 tok** | **3.42%** |
| **Combined** | **514,517,279 tok** | **2,624,993 tok** | **37 tok** | **263,374 tok** | **580,828 tok** | **1,594,248 tok** | **506,807,220 tok** | **~1.50%** |

(Character-level source numbers, from `hindi/statistics/preprocessing_stats.json`:
downloaded raw_chars=1,817,653,089 → kept_chars=1,800,000,073; manual raw_chars=497,674,666
→ kept_chars=480,632,418.)

**Reading this table**: actual cleaning/dedup/safety loss is still small — about 1.50%
combined, up slightly from 1.39% before near-dedup was added. Repetitive-document removal
barely triggered on either source (37 tokens total across both) — both were already fairly
clean going in. **Near-duplicate removal is where the real signal is**: it found **1,299
records (~580.8k tokens, 0.53% of scanned manual records)** that exact-hash matching had
missed entirely — genuine near-identical articles (e.g. syndicated wire stories, or the same
piece republished with a trailing byline/date change) that differ just enough in raw text to
produce a different hash but are still substantively duplicate content. The safety filter
itself remains negligible — **1,808 records (~1.58M tokens, 0.03%)** from IndicCorp and
**12 records (~9.7k tokens, 0.005%)** from news — consistent with both sources already being
curated/edited prose.

Separately — not shown as a percentage in the table above because it isn't a cleaning
effect — capping the downloaded source at 400M tokens discarded **~96.16M tokens** that
were never scanned (the file has 7,687,702 lines total; only 6,206,383 were read before the
cap was reached).

### Record-level detail

| Source | Raw records | Invalid | Repetitive | Exact dup | Near-dup | Unsafe | Kept records |
|---|---:|---:|---:|---:|---:|---:|---:|
| Downloaded (IndicCorp) | 6,206,383 | 144,930 | 4 | 4,305 | — | 1,808 | 6,055,336 |
| Manual (news scrape) | 235,120 | 1,536 | 0 | 399 | 1,299 | 12 | 231,874 |

### Final Hindi corpus and splits

| | Chars | Tokens |
|---|---:|---:|
| Downloaded (capped) | 1,800,000,073 | 400,000,016 |
| Manual (used in full) | 480,632,418 | 106,807,204 |
| **Total** | **2,280,632,491** | **506,807,220** |

**Manual fraction (final, post-cleaning, post-near-dedup, post-safety-filter): 21.07%** —
meets the assignment's ≥20% requirement (down slightly from 21.16% before near-dedup, since
near-dedup removed real duplicate content specifically from the manual side).

| Split | Records | Chars | Target ratio | Actual ratio |
|---|---:|---:|---:|---:|
| `train` | 5,973,084 | 2,166,767,969 | 95% | 95.01% |
| `val` | 156,611 | 56,840,593 | 2.5% | 2.49% |
| `test` | 157,515 | 57,023,929 | 2.5% | 2.50% |

Split assignment is a deterministic hash of each record's own cleaned content (not a
materialized shuffle), so re-running preprocessing is reproducible and memory stays flat
regardless of corpus size.

**Outputs**: `hindi/data/splits/{train,val,test}.txt` (the actual training data) and
`hindi/data/processed/cleaned_corpus.txt` — a combined, pre-split dump of every kept record
across all sources, written purely as a transparency/audit artifact; its line count
(6,287,210) exactly equals the sum of the three split files, confirmed directly.

**Note**: a perplexity-filtering stage (a from-scratch bigram language model used to reject
high-perplexity/anomalous records) was implemented and tested, then deliberately removed per
explicit instruction before this run — the numbers above reflect repetitive-document removal
and near-duplicate removal only, not perplexity filtering.

## Nepali — raw collection totals (input to preprocessing)

| Source | Estimated tokens | Raw chars |
|---|---:|---:|
| Downloaded (IndicCorp) | 500,063,365 | 2,250,285,142 |
| Manual — Wikipedia (ne.wikipedia.org) | 14,296,660 | 64,334,968 |
| Manual — news scrape | 95,705,021 | 430,672,593 |
| **Manual subtotal** | **110,001,680** | **495,007,561** |
| **Total (pre-cleaning)** | **610,065,045** | **2,745,292,703** |

**Manual fraction (pre-cleaning): 18.03%** — see "Manual fraction" note below.

### Nepali manual — news source breakdown (218,742 articles collected)

| Site | Est. tokens | Notes |
|---|---:|---|
| bizmandu.com | 47,640,701 | Deep archive back to 2013; largest contributor |
| rajdhanidaily.com | 34,831,036 | Deep archive back to 2020 |
| nepalnews.com | 10,676,934 | |
| deshsanchar.com | 2,184,693 | Sitemap server intermittently rate-limits after the first request; a circuit-breaker was added so discovery moves on instead of stalling |
| onlinekhabar.com | 131,927 | RSS-only (no deep sitemap found) |
| ratopati.com | 63,880 | 20s crawl-delay declared in robots.txt, respected |
| nagariknews.nagariknetwork.com | 49,454 | |
| nepalmag.com.np | 74,032 | |
| others (setopati, nepalkhabar, news24nepal, ujyaaloonline, pahilopost) | ~52,000 combined | RSS-only, small recent-window pools |
| ekantipur.com, nayapatrikadaily.com | 0 | No discoverable sitemap/feed content found |

### Nepali manual — Wikipedia (29,089 articles collected)

Scraped via a robots.txt-compliant BFS link-following crawl of `ne.wikipedia.org`
(`common/wikipedia_scraper.py`), not a download of the pre-packaged Wikipedia XML/HTML
dumps — this keeps it inside the assignment's own definition of manual collection
("scraping + cleaning pages you gather yourself"). The crawl also recurses into category
pages (discovered via their listed member articles and subcategories) once plain
link-following plateaued, without ever saving a category page's own listing as content.

**This source is now genuinely exhausted**: the discovery frontier reached 0 twice (once
after the initial link-following pass, once again after adding a further ~85 seed
topics/categories), confirming ~14.3M tokens is close to the practical ceiling for this
source — Nepali Wikipedia has on the order of 40,000 articles total, many of them short.

---

## Data cleaning and deduplication — Nepali (complete)

Same pipeline as Hindi (`common/preprocess_corpus.py`, run independently — no shared state,
data, or vocabulary between the two languages): Unicode NFC normalization, Devanagari-script
quality filtering, repetitive-document removal (`is_repetitive_text`), the profanity safety
filter (`common/badwords_nepali.txt`, ~24 terms, same exact-whole-word-token matching as
Hindi's), content-hash exact deduplication with one shared `Deduplicator` instance across the
whole run, and MinHash/LSH near-duplicate deduplication on both manual sources (news +
Wikipedia) — see "Data cleaning and deduplication — Hindi" above for the full rationale
behind each stage and the false-positive classes they avoid. The downloaded source is capped
at ~400M tokens (1,800,000,000 chars); both manual sources are used in full (after filtering).

### Raw (scanned) vs. kept, per source

| Source | Raw scanned | Invalid | Repetitive | Exact dup | Near-dup | Unsafe | **Kept** | **Total reduction** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Downloaded (IndicCorp) | 405,117,978 tok | 1,538,799 tok | 5,487 tok | 3,555,783 tok | — | 18,274 tok | **400,000,025 tok** | **1.26%** |
| Manual (news scrape) | 95,518,820 tok | 1,473,206 tok | 0 tok | 188,387 tok | 2,063,357 tok | 4,948 tok | **90,536,592 tok** | **5.22%** |
| Manual (Wikipedia) | 14,268,758 tok | 294,261 tok | 0 tok | 58,012 tok | 245,426 tok | 5,203 tok | **9,704,659 tok** | **31.99%** |
| **Combined** | **514,905,557 tok** | **3,306,266 tok** | **5,487 tok** | **3,802,182 tok** | **2,308,783 tok** | **28,425 tok** | **500,241,276 tok** | **~2.85%** |

(Character-level source numbers, from `nepali/statistics/preprocessing_stats.json`: downloaded
raw_chars=1,823,030,903 → kept_chars=1,800,000,113; news raw_chars=429,834,692 →
kept_chars=407,414,663; Wikipedia raw_chars=64,209,411 → kept_chars=43,670,965.)

**Reading this table**: the downloaded source behaves similarly to before (1.26% loss).
Wikipedia stands out at **31.99%** total loss — still dominated by the length/quality filter
(short stubs/disambiguation/list pages, as before), but now with a further, meaningful
near-duplicate contribution: **1,653 records (~245.4k tokens, ~6.2% of candidate Wikipedia
records)** were near-duplicates that exact-hash matching missed — plausibly redirect-style or
near-identical stub pages. News scraping shows the same pattern more moderately: **6,760
records (~2.06M tokens, ~2.9% of scanned news records)** were near-duplicates — syndicated or
republished articles. Repetitive-document removal barely triggered anywhere (5,487 tokens
total, all in IndicCorp). The safety filter remains negligible across all three sources.

Separately, capping the downloaded source discarded the remainder of its raw collection
pool that was never scanned: the raw IndicCorp file holds ~500.06M tokens
(2,250,285,142 chars) total, but only 1,823,030,903 chars (~405.12M tokens) were read before
the 400M-kept-token cap was reached — leaving ~427.3M chars (~95.0M tokens) unscanned, same
"deliberate downsizing, not a quality judgment" distinction as Hindi.

### Record-level detail

| Source | Raw records | Invalid | Repetitive | Exact dup | Near-dup | Unsafe | Kept records |
|---|---:|---:|---:|---:|---:|---:|---:|
| Downloaded (IndicCorp) | 8,374,452 | 248,015 | 81 | 93,248 | — | 85 | 8,033,023 |
| Manual (news scrape) | 280,345 | 39,131 | 0 | 1,337 | 6,760 | 5 | 233,112 |
| Manual (Wikipedia) | 52,009 | 24,968 | 0 | 565 | 1,653 | 2 | 24,821 |

### Final Nepali corpus and splits

| | Chars | Tokens |
|---|---:|---:|
| Downloaded (capped) | 1,800,000,113 | 400,000,025 |
| Manual — news (used in full) | 407,414,663 | 90,536,592 |
| Manual — Wikipedia (used in full) | 43,670,965 | 9,704,659 |
| **Manual subtotal** | **451,085,628** | **100,241,251** |
| **Total** | **2,251,085,741** | **500,241,276** |

**Manual fraction (final, post-cleaning, post-near-dedup, post-safety-filter): 20.04%** —
still meets the assignment's ≥20% requirement, and the total (500.24M tokens) lands almost
exactly on the ~500M target. **Note the margin has narrowed considerably** (from 20.41%
before near-dedup to 20.04% now) — near-duplicate removal found substantially more real
duplicate content on the manual side (especially Wikipedia's ~6.2% near-dup rate) than on
the downloaded side, so Nepali's manual fraction is now much closer to the 20% floor than
comfortably above it.

| Split | Records | Chars | Target ratio | Actual ratio |
|---|---:|---:|---:|---:|
| `train` | 7,876,088 | 2,138,614,328 | 95% | 95.00% |
| `val` | 207,528 | 56,231,353 | 2.5% | 2.50% |
| `test` | 207,340 | 56,240,060 | 2.5% | 2.50% |

**Outputs**: `nepali/data/splits/{train,val,test}.txt` (the actual training data) and
`nepali/data/processed/cleaned_corpus.txt` — a combined, pre-split dump of every kept record
across all sources, written purely as a transparency/audit artifact; its line count
(8,290,956) exactly equals the sum of the three split files, confirmed directly.

**Note**: as with Hindi, a perplexity-filtering stage (from-scratch bigram LM) was
implemented, tested, then deliberately removed per explicit instruction before this run —
the numbers above reflect repetitive-document removal and near-duplicate removal only.

---

## Nepali manual-data ceiling (why Model L's numbers look different from Model H's)

News-site scraping alone, across all 7 initially-vetted Nepali sites, exhausted at only
~92,500 tokens — the entire discoverable pool via sitemaps/RSS for those sites, not a
"needs more time" problem. This is itself evidence of the underlying resource gap between
the two languages that the assignment's Phase 2/3 discussion asks about: there is simply
far less Nepali news content published and sitemap-indexed online than Hindi content.
Two rounds of adding more Nepali sites (`deshsanchar`, `nepalnews`, `bizmandu`,
`rajdhanidaily`, and others) and adding the Wikipedia crawl closed most of the gap, but the
underlying scarcity — not a tooling limitation — is the reason Nepali's manual pool required
substantially more source-hunting effort than Hindi's to reach a comparable token count.

## Manual fraction

Both languages' *raw, pre-cleaning* collection totals sat at ~18% manual — just under the
assignment's 20% floor (see "Nepali manual-data ceiling" above for why manual collection is
inherently harder for the lower-resource language). Per user instruction, the fix applied
was **not** to force a specific post-cleaning ratio by truncating whichever source cleaning
happened to hit harder; instead, the downloaded source is capped at a fixed ~400M tokens
during preprocessing (`max_kept_chars` in `hindi/data/preprocess.py` /
`nepali/data/preprocess.py`) while the manual source(s) are used in full — landing each
corpus close to ~500M tokens total with manual comfortably above 20%, rather than right at
the boundary.

- **Hindi: 506.81M tokens total, 21.07% manual, final** (measured after cleaning, repetitive-
  document removal, exact + near-duplicate removal, and the safety filter — see the
  comparison table above). Requirement met.
- **Nepali: 500.24M tokens total, 20.04% manual, final** (same full filter stack). Requirement
  met, and the total lands almost exactly on the ~500M target — but the margin above the 20%
  floor is now narrow, since near-duplicate removal found substantially more real duplicate
  content on the manual side than the downloaded side (see "Data cleaning and deduplication —
  Nepali" above).

**Note on perplexity filtering**: a from-scratch bigram-language-model perplexity filter
(trained on each language's own data, not a pretrained model) was implemented and tested as
an additional cleaning stage, then deliberately removed per explicit instruction before the
numbers above were produced. The corpus totals in this report reflect repetitive-document
removal and MinHash near-duplicate removal only.

![Final corpus composition: downloaded vs. manual tokens, both languages](plot/corpus_composition.png)

![Train / validation / test split sizes, both languages](plot/split_sizes.png)

## Tokenizer training — Hindi

**Retrained on the current splits** (post-repetitive-document-removal, post-near-dedup;
506.81M tokens, 21.07% manual — see "Data cleaning and deduplication — Hindi" above). The
corpus shifted by well under 1% from the previous preprocessing run, so the numbers below are
almost unchanged from the prior tokenizer run, but are now current rather than stale.

Trained from scratch with SentencePiece BPE (`common/train_tokenizer.py`,
`hindi/tokenizer/train_tokenizer.py`) on `hindi/data/splits/train.txt` only, with vocab size
chosen by sweeping candidates and evaluating each on the held-out `hindi/data/splits/val.txt`
(never seen in training) — `test.txt` is untouched, kept fully unseen for later phases.
`byte_fallback=True` is enabled, so every candidate achieved a **0.0% UNK rate** (any
otherwise-unrepresentable byte sequence falls back to raw bytes rather than `<unk>`).
Per TA guidance that 5k–10k should be the best range for this corpus scale, the sweep covered
that band plus two larger points (16k, 24k) as a wider comparison:

| Vocab size | Fertility (tok/word) | Avg chars/token | UNK rate | Fertility gain per +1k vocab (vs. previous) |
|---:|---:|---:|---:|---:|
| 5,000 | 1.4839 | 3.437 | 0.0% | — |
| 7,000 | 1.3960 | 3.653 | 0.0% | 0.0440/1k |
| 9,000 | 1.3421 | 3.800 | 0.0% | 0.0270/1k |
| 10,000 | 1.3227 | 3.856 | 0.0% | 0.0194/1k |
| 16,000 | 1.2509 | 4.077 | 0.0% | 0.0120/1k |
| 24,000 | 1.2067 | 4.226 | 0.0% | 0.0055/1k |

**Why "lowest fertility" alone is not a usable selection rule here**: because UNK rate is
0.0% at every candidate (byte-fallback), the sweep's original selection criterion ("lowest
fertility among candidates meeting the UNK bar") mechanically reduces to "pick the largest
vocab size tested" — fertility strictly decreases as vocab size grows, so it would select
24,000 by default regardless of what candidates were even tried. That is not a meaningful
trade-off judgment, just an artifact of testing an upper bound. The table above shows why:
the *rate* of fertility improvement falls off sharply and monotonically — from 0.044 per
+1,000 vocab (5k→7k) down to 0.006 per +1,000 vocab (16k→24k), roughly an 8x drop in "return
per token of vocabulary added." Past ~10k, more vocabulary buys very little additional
compression.

![Tokenizer vocab-size sweep: fertility vs. vocabulary size, Hindi and Nepali, with the selected 10,000 point circled](plot/fertility_vs_vocab_size.png)

**Parameter-budget trade-off**: for the assignment's target of a compact (~25M-parameter)
decoder-only Transformer, the token embedding table costs `vocab_size × d_model` parameters
— pure lookup capacity, not attention/FFN modeling capacity — and (if the output projection
is untied from the input embedding, as is common) that cost is paid twice. Using
`d_model=256` as an illustrative assumption for a ~25M-parameter model (adjust proportionally
for a different hidden size):

| Vocab size | Embedding params (tied) | % of 25M budget (tied) | % of 25M budget (untied) |
|---:|---:|---:|---:|
| 5,000 | 1.28M | 5.1% | 10.2% |
| 7,000 | 1.79M | 7.2% | 14.3% |
| 9,000 | 2.30M | 9.2% | 18.4% |
| **10,000** | **2.56M** | **10.2%** | **20.5%** |
| 16,000 | 4.10M | 16.4% | 32.8% |
| 24,000 | 6.14M | 24.6% | 49.2% |

At 24,000 (what the naive rule would pick), the embedding table alone would consume roughly a
**quarter to a half** of the entire 25M-parameter budget, leaving disproportionately little
for the actual transformer layers that do the language modeling. At 10,000, that cost drops
to ~10–20%, a far more reasonable share for a small model.

**Selected vocab size: 10,000** — not the mechanically-selected 24,000. Justification:
10,000 sits at the top of the TA-recommended 5k–10k band, captures **58%** of the total
fertility improvement available across the entire tested range (1.4838 → 1.2067) while using
only **26%** of the tested vocab-size range (5k→10k out of 5k→24k), and keeps the embedding
table to a modest, defensible fraction of a 25M-parameter budget rather than a quarter-to-half
of it. Beyond 10k, every further fertility gain costs disproportionately more parameters for
steadily diminishing returns.

**Confirmed final choice: 10,000.** The tokenizer files saved on disk
(`hindi/tokenizer/hindi_tokenizer.model` / `.vocab`) have been retrained at vocab_size=10,000
(`common/finalize_tokenizer.py`) so the saved artifact matches this recommendation; the
`final_eval_on_full_val` / `token_frequency_stats` / example-tokenization figures in
`hindi/statistics/tokenizer_stats.json` below are all measured on that 10,000-vocab model,
trained on the current (post-repetitive/near-dedup-filter) splits.

### Token-frequency statistics and tokenization examples (final 10,000 model)

Measured on 200,000 sampled lines from `train.txt` (15,438,061 tokens): 9,860 of 10,000
vocabulary entries (98.6%) were used at least once; the top 100 tokens account for **39.3%**
of all token occurrences (a small set of function words/postpositions and punctuation
dominates, as expected for Hindi).

**Top 20 tokens by frequency (Hindi, 10,000-vocab model):**

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

As expected for a postposition-heavy, high-frequency-function-word language, the entire top
20 is grammatical function words and punctuation rather than content words — content words
appear much further down the frequency distribution, consistent with Zipf's law.

Example tokenizations (final full-val evaluation: fertility 1.3226 tok/word, avg 3.887
chars/token, 0.0% UNK, over 156,611 val lines / 14,623,086 tokens):

| Text | Pieces | Count |
|---|---|---:|
| भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं। | ▁भारत / ▁एक / ▁विशाल / ▁देश / ▁है / ▁जहाँ / ▁अनेक / ▁भाषा / एँ / ▁बोली / ▁जाती / ▁हैं / । | 13 |
| प्रधानमंत्री ने आज नई दिल्ली में एक महत्वपूर्ण बैठक की अध्यक्षता की। | ▁प्रधानमंत्री / ▁ने / ▁आज / ▁नई / ▁दिल्ली / ▁में / ▁एक / ▁महत्वपूर्ण / ▁बैठक / ▁की / ▁अध्यक्षता / ▁की / । | 13 |
| क्रिकेट विश्व कप के फाइनल मैच में भारत ने शानदार जीत दर्ज की। | ▁क्रिकेट / ▁विश्व / ▁कप / ▁के / ▁फाइनल / ▁मैच / ▁में / ▁भारत / ▁ने / ▁शानदार / ▁जीत / ▁दर्ज / ▁की / । | 14 |

Most content words tokenize as a single whole-word piece even at 10k vocab (e.g. भारत,
प्रधानमंत्री, दिल्ली, क्रिकेट); the one visible split (▁भाषा + एँ) shows the tokenizer falling
back to a stem + suffix decomposition for a less frequent inflected form — the expected BPE
behavior at this vocab size, and consistent with fertility ≈ 1.32.

## Tokenizer training — Nepali

**Retrained on the current splits** (post-repetitive-document-removal, post-near-dedup;
500.24M tokens, 20.04% manual — see "Data cleaning and deduplication — Nepali" above). As
with Hindi, the corpus shifted only slightly from the previous preprocessing run, so the
numbers below are close to before but now current.

Same pipeline as Hindi (`common/train_tokenizer.py`, `nepali/tokenizer/train_tokenizer.py`):
SentencePiece BPE trained from scratch on `nepali/data/splits/train.txt` only, vocab size
chosen by sweeping candidates and evaluating each on held-out `nepali/data/splits/val.txt`;
`test.txt` untouched. `byte_fallback=True` again gives **0.0% UNK rate** at every candidate.
Same candidate range as Hindi (5k/7k/9k/10k/16k/24k):

| Vocab size | Fertility (tok/word) | Avg chars/token | UNK rate | Fertility gain per +1k vocab (vs. previous) |
|---:|---:|---:|---:|---:|
| 5,000 | 1.7252 | 3.738 | 0.0% | — |
| 7,000 | 1.6005 | 4.029 | 0.0% | 0.0623/1k |
| 9,000 | 1.5224 | 4.236 | 0.0% | 0.0390/1k |
| 10,000 | 1.4934 | 4.318 | 0.0% | 0.0290/1k |
| 16,000 | 1.3848 | 4.657 | 0.0% | 0.0181/1k |
| 24,000 | 1.3126 | 4.913 | 0.0% | 0.0090/1k |

**Same diminishing-returns pattern as Hindi**: fertility-improvement rate per +1,000 vocab
falls from 0.062 (5k→7k) to 0.009 (16k→24k) — roughly a 7x drop — so the sweep's raw "lowest
fertility" rule would again mechanically pick the largest candidate (24,000) tested rather
than reflecting a genuine trade-off point. Notably, **Nepali's fertility is higher than
Hindi's at every matched vocab size** (e.g. 1.493 vs. 1.323 at 10,000) — the tokenizer needs
more subword pieces per word for the same vocabulary budget, consistent with Nepali's richer
suffix-chaining morphology and its smaller, more source-diverse corpus (more Wikipedia
content, less single-domain news style) giving the trainer a broader, less repetitive
vocabulary to compress.

**Parameter-budget trade-off**: identical embedding-cost table to Hindi's (same candidate
vocab sizes, same `d_model=256` illustrative assumption for a ~25M-parameter model) — see
"Tokenizer training — Hindi" above. At 24,000, the embedding table alone would still consume
roughly a quarter (tied) to half (untied) of a 25M-parameter budget.

**Selected vocab size: 10,000** — same reasoning as Hindi: top of the TA-recommended 5k–10k
band, captures **56%** of the total fertility improvement available across the tested range
(1.7252 → 1.3126) using only **26%** of the tested vocab-size range, and keeps the embedding
table to a modest, defensible share of the parameter budget rather than a quarter-to-half of
it.

**Confirmed final choice: 10,000.** Same as Hindi — the tokenizer files saved on disk
(`nepali/tokenizer/nepali_tokenizer.model` / `.vocab`) have been retrained at vocab_size=10,000
(`common/finalize_tokenizer.py`) on the current (post-repetitive/near-dedup-filter) splits;
the `final_eval_on_full_val` / `token_frequency_stats` / example-tokenization figures in
`nepali/statistics/tokenizer_stats.json` below are all measured on that model.

### Token-frequency statistics and tokenization examples (final 10,000 model)

Measured on 200,000 sampled lines from `train.txt` (10,330,982 tokens): 9,880 of 10,000
vocabulary entries (98.8%) were used at least once; the top 100 tokens account for **27.5%**
of all occurrences — a notably flatter distribution than Hindi's 39.3%, i.e. Nepali's token
usage is more evenly spread rather than dominated by a small set of function words.

**Top 20 tokens by frequency (Nepali, 10,000-vocab model):**

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

Notably, several of Nepali's top-20 pieces are **bound case/postposition suffixes without a
leading space marker** (`को`, `मा`, `का`, `ले`, `लाई`, `बाट`) rather than whole standalone
words — a direct, visible signature of Nepali's suffix-chaining morphology fusing case markers
onto noun/verb stems, which Hindi's largely-analytic postposition system (mostly separate
space-delimited words, e.g. `▁के`/`▁में`/`▁की`) does not show at the same vocab size. This is
consistent with Nepali's higher fertility at every matched vocab size reported above.

Example tokenizations (final full-val evaluation: fertility 1.4947 tok/word, avg 4.315
chars/token, 0.0% UNK, over 200,000 val lines / 10,382,990 tokens):

| Text | Pieces | Count |
|---|---|---:|
| नेपाल हिमालयको काखमा रहेको सुन्दर देश हो। | ▁नेपाल / ▁हिमाल / यको / ▁का / ख / मा / ▁रहेको / ▁सुन्दर / ▁देश / ▁हो / । | 11 |
| प्रधानमन्त्रीले आज काठमाडौंमा एक महत्वपूर्ण बैठकको अध्यक्षता गर्नुभयो। | ▁प्रधानमन्त्रीले / ▁आज / ▁काठमाडौंमा / ▁एक / ▁महत्वपूर्ण / ▁बैठकको / ▁अध्यक्ष / ता / ▁गर्नुभयो / । | 10 |
| क्रिकेट विश्वकपको फाइनल खेलमा नेपालले उत्कृष्ट प्रदर्शन गर्यो। | ▁क्रिकेट / ▁विश्वक / पको / ▁फाइनल / ▁खेलमा / ▁नेपालले / ▁उत्कृष्ट / ▁प्रदर्शन / ▁गर्यो / । | 10 |

At 10k vocab, several long inflected forms still tokenize as a single whole-word piece,
including case/postposition suffixes fused onto the stem (प्रधानमन्त्रीले, काठमाडौंमा,
बैठकको, नेपालले) — but more stem+suffix splits are visible than in Hindi's examples at the
same vocab size (▁हिमाल+यको+▁का+ख+मा, ▁विश्वक+पको, ▁अध्यक्ष+ता), directly illustrating why
Nepali's fertility (1.49) is meaningfully higher than Hindi's (1.32) at the same 10,000-token
budget.

## Compliance and collection notes

- All fetches (sitemaps, RSS feeds, and article pages) are gated by `robots_utils.py`'s
  `RobotsCache`, which fetches and parses each site's live `robots.txt` per request — not
  just at the one-time site-selection stage — including a fix for a UTF-8 BOM bug that was
  silently making some sites' robots.txt parse as empty (discovered on `livehindustan.com`)
  and a second, related bug where a BOM at the start of a *sitemap* response (not just
  robots.txt) made a valid sitemap look like non-XML content and get silently skipped
  (discovered on `punjabkesari.in`).
- Per-site crawl-delay is read from each site's own robots.txt where declared (e.g.
  `ratopati.com`'s 20s) and otherwise defaults to a 2-second minimum; multiple worker
  threads per site pace themselves independently at that same delay (the standard
  "N workers × per-connection delay" model, comparable to Scrapy's
  `CONCURRENT_REQUESTS_PER_DOMAIN`), rather than one worker firing requests faster than the
  declared delay.
- Non-article content (videos, photo galleries, web-stories, live-blogs) is filtered out of
  the candidate URL pool by path pattern before fetching, not just after extraction.

## Pending for this deliverable

- [x] Run `hindi/data/preprocess.py` to produce cleaned, deduplicated `train`/`val`/`test`
      splits and the real (post-cleaning) manual/downloaded token split. **Done** — see
      "Data cleaning and deduplication — Hindi" above.
- [x] Run `nepali/data/preprocess.py` (same pipeline, 400M-token download cap, both manual
      sources used in full). **Done** — see "Data cleaning and deduplication — Nepali" above.
- [x] Add and apply a safety filter (profanity removal) to both languages —
      `common/badwords_hindi.txt` / `common/badwords_nepali.txt`, exact-whole-word matching.
      **Done** — see the updated comparison tables in both languages' "Data cleaning and
      deduplication" sections above.
- [x] Add repetitive-document removal and MinHash/LSH near-duplicate removal to the cleaning
      pipeline and re-run preprocessing for both languages. **Done** — see the updated "Data
      cleaning and deduplication" sections above. (A from-scratch perplexity-filtering stage
      was also implemented and tested, then deliberately removed per explicit instruction.)
- [x] Retrain Hindi tokenizer sweep (5k/7k/9k/10k/16k/24k) on the current (post-filter)
      splits and report vocabulary size, fertility, avg chars/token, UNK-rate, and
      token-frequency statistics. **Done, current** — see "Tokenizer training — Hindi" above.
      Final vocab size: **10,000**, trade-off justified against the ~25M-parameter budget.
- [x] Retrain Nepali tokenizer sweep (same candidate range) on the current splits and report
      the same statistics. **Done, current** — see "Tokenizer training — Nepali" above. Final
      vocab size: **10,000**, same trade-off reasoning.
- [ ] Retrain both languages' tokenizers on the current (post-repetitive/near-dedup-filter)
      splits, so the saved `.model`/`.vocab` files and reported sweep/token-frequency/example
      statistics match the current corpus. Not yet done — pending instruction.
- [x] Update this report with the current preprocessing numbers. **Done**.
