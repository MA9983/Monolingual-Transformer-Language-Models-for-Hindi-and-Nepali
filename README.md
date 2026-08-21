# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Individual project for *Language Models and Agents* (Monsoon 2026). The goal is to build,
train, and analyze **two completely independent** decoder-only Transformer language models
from scratch — one for a higher-resource Indian language, one for a lower-resource Indian
language — covering corpus collection, tokenizer training, architecture implementation,
pretraining, reasoning finetuning, and evaluation. The two models never share data, tokenizer,
vocabulary, or weights.

| | Language | Role |
|---|---|---|
| **Model H** | Hindi | Higher-resource — large public corpora (IndicCorp, Wikipedia) and abundant online news text. |
| **Model L** | Nepali | Lower-resource — chosen from the assignment's allowed lower-resource list (Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, Nepali, Sindhi). Genuinely resource-constrained online relative to Hindi: manual data collection required substantially more source-hunting effort to reach a comparable token count (see `report/phase1_report.md`). |

Both languages are Devanagari-script, which lets shared script-level utilities (Unicode
normalization, script-ratio quality filtering) be reused safely without conflating the two
languages' actual vocabularies or training data — every pipeline still runs fully
independently per language.

## Project status

- **Phase 1 (Data Collection and Tokenizer Construction) — complete.** See
  [`report/phase1_report.md`](report/phase1_report.md) for full methodology, statistics, and
  figures.
- **Phase 2 (Model Implementation, Pretraining, Evaluation)** and **Phase 3 (Reasoning
  Finetuning, Attention Analysis, Final Report)** — not started yet; `*/model/`, `*/train/`,
  and `*/eval/` directories are scaffolded but currently empty.

### Phase 1 results at a glance

| | Hindi (Model H) | Nepali (Model L) |
|---|---:|---:|
| Final corpus size | 506.81M tokens | 500.24M tokens |
| Manual fraction | 21.07% | 20.04% |
| Manual sources | News scrape | News scrape + Wikipedia |
| Train / val / test split | 95% / 2.5% / 2.5% | 95% / 2.5% / 2.5% |
| Tokenizer | SentencePiece BPE, vocab 10,000 | SentencePiece BPE, vocab 10,000 |
| Fertility (tokens/word, held-out val) | 1.3226 | 1.4947 |

Nepali's manual fraction sits closer to the assignment's 20% floor than Hindi's — MinHash
near-duplicate removal (see below) found substantially more real duplicate content on
Nepali's manual side (news + Wikipedia) than on Hindi's, narrowing its margin above the
requirement. Both still comfortably meet it.

Full before/after cleaning tables, source breakdowns, the tokenizer vocab-size sweep and its
trade-off justification, and top-20 token-frequency tables are in
[`report/phase1_report.md`](report/phase1_report.md) with supporting figures in
[`report/plot/`](report/plot/).

## Repository layout

```
common/                     Shared code used by both languages independently
                             (no shared data, vocab, or model state)
  download_indiccorp.py       Stream IndicCorp v2 from HuggingFace to a token-count target
  news_scraper.py              Sitemap/RSS-driven, robots.txt-compliant news-site scraper
  wikipedia_scraper.py         BFS/category-recursive ne.wikipedia.org crawler (Nepali only)
  robots_utils.py               robots.txt fetching/parsing (crawl-delay, AI-crawler policy)
  text_cleaning.py              Unicode normalization, quality filtering, repetitive-document
                                   removal, exact dedup, MinHash/LSH near-dedup, safety filter
  badwords_hindi.txt            Devanagari profanity wordlist (Hindi safety filter)
  badwords_nepali.txt           Devanagari profanity wordlist (Nepali safety filter)
  preprocess_corpus.py          Shared clean -> repetitive-removal -> safety-filter ->
                                   exact-dedup -> near-dedup -> split pipeline
  train_tokenizer.py            SentencePiece BPE training + vocab-size sweep + eval stats
  finalize_tokenizer.py         Retrain/save the final tokenizer at a chosen vocab size

hindi/                       Model H — fully self-contained
  data/
    raw/                       Downloaded + manually scraped raw text (gitignored, see below)
    processed/                 Combined cleaned+deduplicated corpus (audit artifact, gitignored)
    splits/                    train.txt / val.txt / test.txt (gitignored)
    preprocess.py               Wires up Hindi's sources into the shared pipeline
  tokenizer/                  train_tokenizer.py + hindi_tokenizer.model / .vocab
  statistics/                 download_stats.json, preprocessing_stats.json, tokenizer_stats.json
  model/ train/ eval/ configs/   Scaffolded for Phase 2, currently empty

nepali/                      Model L — mirrors hindi/ exactly, fully independent

report/
  phase1_report.md            Full Phase 1 methodology, statistics, and cross-language analysis
  hindi_report.md              Hindi-only deep dive: vocab size, token-frequency stats,
                                 avg chars/token, tokenization examples, UNK stats
  nepali_report.md             Same, for Nepali
  plot/                        Generated figures (generate_plots.py + PNGs) referenced by the reports

requirements.txt             Python dependencies
```

## Data and large artifacts

Per the assignment's policy, raw/processed/split corpora and model checkpoints are **not**
committed to git (see `.gitignore`) — only lightweight code, statistics JSON files, and the
tokenizer model/vocab files are tracked. Large artifacts are hosted on Kaggle Datasets
instead of Google Drive, due to insufficient available Google Drive storage for corpora of
this size:

- Hindi raw dataset: https://www.kaggle.com/datasets/mohdahmad2868/hindi-raw-dataset
- Hindi processed + split dataset: https://www.kaggle.com/datasets/mohdahmad2868/hindi-processed-dataset
- Nepali raw dataset: https://www.kaggle.com/datasets/mohdahmad2868/nepali-raw-dataset
- Nepali processed + split dataset: https://www.kaggle.com/datasets/mohdahmad2868/nepali-processed-dataset

Model checkpoints (Phase 2 onward) will be linked here once trained.

## Reproduction steps (Phase 1)

```bash
pip install -r requirements.txt
```

1. **Collect data** (per language — see `report/phase1_report.md` for the exact site lists,
   token caps, and robots.txt compliance policy used):
   ```bash
   python3 common/download_indiccorp.py   # downloaded source, per language
   python3 common/news_scraper.py         # manual news scrape, per language
   python3 common/wikipedia_scraper.py    # manual Wikipedia scrape, Nepali only
   ```
2. **Clean, deduplicate, safety-filter, and split** each language independently:
   ```bash
   python3 hindi/data/preprocess.py
   python3 nepali/data/preprocess.py
   ```
   This applies, in order: Unicode NFC normalization, Devanagari-script quality filtering,
   repetitive-document removal (Gopher-style top-n-gram-ratio heuristic), profanity
   safety-filtering (`common/badwords_*.txt`), content-hash exact deduplication, and
   MinHash/LSH near-duplicate deduplication (manually scraped sources only — see
   `common/text_cleaning.py`'s `NearDuplicateDetector` for why it's not applied to the
   multi-million-line downloaded source). Writes deterministic `train`/`val`/`test` splits
   and a combined `data/processed/cleaned_corpus.txt` audit file. Per-source and aggregate
   statistics are written to `*/statistics/preprocessing_stats.json`.

   A from-scratch bigram-language-model perplexity filter was also implemented and tested as
   an additional stage, then deliberately removed per instruction — the pipeline above does
   not include it.
3. **Train tokenizers** (SentencePiece BPE, from scratch, one per language, no shared
   vocabulary):
   ```bash
   python3 hindi/tokenizer/train_tokenizer.py
   python3 nepali/tokenizer/train_tokenizer.py
   ```
   Sweeps candidate vocab sizes (5k/7k/9k/10k/16k/24k) and evaluates each on the held-out
   `val.txt` split (fertility, avg chars/token, UNK rate). The vocab size actually shipped
   (10,000 for both languages) was chosen by weighing the sweep's diminishing returns against
   the assignment's ~25M-parameter model budget rather than the sweep script's automatic
   "lowest fertility" rule (which always favors the largest candidate once UNK rate is 0% at
   every size) — see "Tokenizer training" in `report/phase1_report.md` for the full
   justification — and finalized at that chosen size with:
   ```python
   from common.finalize_tokenizer import finalize

   finalize(language="hindi", train_path="hindi/data/splits/train.txt",
            val_path="hindi/data/splits/val.txt", out_dir="hindi/tokenizer",
            stats_path="hindi/statistics/tokenizer_stats.json", vocab_size=10000)
   # same call for nepali/, with vocab_size=10000
   ```
4. **Generate report figures**:
   ```bash
   python3 report/plot/generate_plots.py
   ```

Phase 2 (model implementation and pretraining) and Phase 3 (reasoning finetuning and attention
analysis) reproduction steps will be added here as those phases are completed.
