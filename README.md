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
- **Phase 2 (Model Implementation, Pretraining, Evaluation) — complete.** Both models
  implemented from scratch in PyTorch, pretrained independently to 30,000 steps, and
  evaluated (intrinsic LM metrics, generation quality, attention analysis). See
  [`report/phase2_report.md`](report/phase2_report.md) for full methodology, results, and
  analysis.
- **Phase 3 (Reasoning Finetuning, Attention Analysis, Final Report)** — not started yet.

### Phase 2 results at a glance

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Parameters | 25,019,328 | 25,019,328 |
| Architecture | 6-layer, `d_model=512`, 8 heads, RoPE, pre-norm, tied embeddings | Same (independent weights) |
| Pretraining steps / tokens | 30,000 steps / ~491.5M tokens | 30,000 steps / ~491.5M tokens |
| Validation perplexity | 24.24 | 40.91 |
| Validation bits-per-byte (BPB) | 0.4665 | 0.4688 |
| BLEU-4 / chrF++ / ROUGE-L (greedy) | 38.34 / 48.94 / 0.242 | 37.42 / 48.77 / 0.067 |
| Distinct-1 / Distinct-2 | 0.300 / 0.548 | 0.381 / 0.556 |

The perplexity gap is almost entirely a tokenizer-fertility artifact (Nepali needs more
subword pieces per word at the same 10k vocab) rather than a real quality gap — bits-per-byte,
the tokenizer-invariant metric, is under 0.5% apart between the two models. The ROUGE-L gap
traces to a specific greedy-decoding mode-collapse pattern in Model L, not a general fluency
issue. Full discussion, per-decoding-setting tables, generated samples, and attention-head
analysis are in [`report/phase2_report.md`](report/phase2_report.md).

## Architecture (Model H and Model L)

Both models share the exact same architecture *shape* — only the trained weights differ, since
each is trained independently on its own language's data. A decoder-only (GPT-style)
Transformer built entirely from PyTorch primitives (`nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout`) — no `nn.Transformer*`, no `nn.MultiheadAttention`, no
HuggingFace model classes, per the assignment's hard constraint.

- **Input representation**: token ids → learned embedding (`vocab_size × d_model`), scaled by
  `sqrt(d_model)`, plus embedding dropout. No positional term is added at the input.
- **Positional encoding — RoPE (Rotary Position Embedding)**, not learned-absolute or
  sinusoidal: query/key vectors are rotated by an angle proportional to sequence position,
  inside every attention layer, so relative position falls directly out of the attention score
  computation rather than being injected once at the input and left for the network to
  propagate through depth. Precomputed cos/sin tables are capped at `context_length` (256);
  the implementation raises `ValueError` on any longer input rather than silently
  extrapolating past what it was trained on.
- **Multi-head causal self-attention**: a single fused `Linear(d_model, 3*d_model)` projects to
  Q/K/V, reshaped to `(B, n_heads, T, d_head)` with `d_head = d_model / n_heads = 512/8 = 64`;
  RoPE applied to Q and K only; scaled dot-product attention (`1/sqrt(d_head)` scaling) under
  an additive `-inf` upper-triangular causal mask (position `t` sees only positions `≤ t`,
  verified empirically by checking that changing token `t+1` never changes position `t`'s
  logits); heads concatenated and passed through an output projection.
- **Transformer block**: pre-norm LayerNorm → attention → residual add → pre-norm LayerNorm →
  feed-forward (`Linear(d_model→d_ff)` → GELU → `Linear(d_ff→d_model)`) → residual add.
  Pre-norm was chosen over post-norm because it keeps the residual stream itself unnormalized,
  letting gradients flow back through the residual path largely unimpeded at this depth.
- **Output head**: `Linear(d_model, vocab_size, bias=False)`, **weight-tied** to the input
  token embedding — saves `vocab_size × d_model` = 5.12M parameters (~a fifth of the 25M
  budget) versus a separate output matrix.
- **Objective**: standard causal-LM cross-entropy between the logits at position `t` and the
  token at position `t+1`, averaged over all positions.

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 10,000 (each language's own Phase 1 SentencePiece tokenizer) |
| `d_model` | 512 |
| `n_layers` | 6 |
| `n_heads` | 8 (`d_head` = 64) |
| `d_ff` | 2,208 (chosen to land almost exactly on the ~25M target — a 4×`d_model` ratio would give 2,048 and 24.04M params instead) |
| `context_length` | 256 |
| Activation | GELU |
| Norm placement | Pre-norm |
| Dropout (embed / attn / resid) | 0.1 / 0.1 / 0.1 |
| **Exact parameter count** | **25,019,328** (both models, via `GPTLanguageModel.num_parameters()`) |

Full justification (depth/width trade-off given the compute budget, RoPE-vs-alternatives
comparison, per-component "what would break if removed" reasoning) is in
[`report/phase2_report.md`](report/phase2_report.md#1-architecture-section-21).

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

hindi/                       Model H — fully self-contained (no cross-imports with nepali/,
                               no shared model/train/eval module either)
  data/
    raw/                       Downloaded + manually scraped raw text (gitignored, see below)
    processed/                 Combined cleaned+deduplicated corpus (audit artifact, gitignored)
    splits/                    train.txt / val.txt / test.txt (gitignored)
    tokenized/                 train.bin / val.bin / test.bin, uint16 token arrays (gitignored)
    preprocess.py               Wires up Hindi's sources into the shared pipeline
  tokenizer/                  train_tokenizer.py + hindi_tokenizer.model / .vocab
  configs/                    model_config.json, training_config.json (exact hyperparameters,
                                 verified parameter count)
  model/                      embeddings.py, attention.py, block.py, rope.py, gpt.py —
                                 decoder-only Transformer from scratch (PyTorch primitives only)
  train/                      prepare_data.py (tokenize splits), dataset.py (memmap
                                 windowing), trainer.py (AdamW + LR schedule + checkpointing)
  eval/                       lm_metrics.py (PPL/BPB), generation_metrics.py (BLEU/chrF/
                                 ROUGE-L/diversity), run_generation_eval.py (real held-out
                                 eval), attention_analysis.py, plotting.py
  model/checkpoints/          Trained checkpoint (gitignored, 287MB — see Google Drive links
                                 below)
  statistics/
    phase2_eval_outputs/      Training logs, loss curves, PPL/BPB + generation results,
                                 attention heatmaps/entropy/distance, all figures (tracked)

nepali/                      Model L — mirrors hindi/ exactly, fully independent

report/
  phase1_report.md            Full Phase 1 methodology, statistics, and cross-language analysis
  hindi_report.md              Hindi-only deep dive: vocab size, token-frequency stats,
                                 avg chars/token, tokenization examples, UNK stats
  nepali_report.md             Same, for Nepali
  plot/                        Phase 1 figures (generate_plots.py + PNGs)
  phase2_report.md             Full Phase 2 methodology, results, and H-vs-L analysis
  phase2_plot/                 Phase 2 cross-language figures (generate_ppl_bpb_comparison.py
                                 + PNG) — the only Phase 2 plot needing both languages' results
                                 at once, so it's generated locally, not inside either
                                 language's single-language Kaggle run

kaggle_upload/                Kaggle API automation: credentials (gitignored), the training
                                 notebook, and per-language kernel-push folders used to run
                                 both models' pretraining/evaluation on Kaggle GPUs

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

### Model checkpoints (Phase 2)

Both models' final trained checkpoints (287MB each, step 30,000) are currently hosted as
private Kaggle datasets:

- Hindi (Model H): https://www.kaggle.com/datasets/mohdahmad2868/phase2-hindi-checkpoint-30000
- Nepali (Model L): https://www.kaggle.com/datasets/mohdahmad2868/phase2-nepali-checkpoint-30000

> **Pending**: the assignment specifically requires checkpoints on **Google Drive** with
> shareable links here. That upload has not been done yet — the Kaggle links above are a
> working interim source (checkpoints are downloadable from them the same way they were
> pulled down for local evaluation) but should not be treated as satisfying the Drive
> requirement until this note is removed.

**Checkpoint format** (both languages, identical schema — see `Trainer.save_checkpoint` /
`load_checkpoint` in `hindi/train/trainer.py`): each `checkpoint_step<N>.pt` is a single
`torch.save`d dict with —

| Key | Contents |
|---|---|
| `step` | Training step this checkpoint was saved at (30,000 for both final checkpoints) |
| `model_state_dict` | Full model weights (`GPTLanguageModel.state_dict()`) |
| `optimizer_state_dict` | AdamW state (moment estimates, etc.) — needed to resume training exactly, not needed just to run inference |
| `scheduler_state_dict` | LR scheduler state — same, resume-only |
| `train_config` | The exact training hyperparameters this run used (see `configs/training_config.json`) |
| `loss_history` | List of `(step, train_loss)` tuples for the whole run |
| `val_loss_history` | List of `(step, val_loss)` tuples, one per `eval_interval_steps` |

To just run inference/evaluation (as `hindi/eval/run_generation_eval.py` does), only
`model_state_dict` is actually needed — construct a `GPTLanguageModel` from
`configs/model_config.json` and call `model.load_state_dict(checkpoint["model_state_dict"])`.

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

## Reproduction steps (Phase 2)

Architecture, training, and evaluation code is fully duplicated per language (no shared
`model`/`train`/`eval` module) — every command below is run once per language directory.

1. **Tokenize the splits** into flat `uint16` binary arrays (adds document-boundary EOS
   tokens; only needs Phase 1's `data/splits/*.txt` and tokenizer, no GPU):
   ```bash
   python3 hindi/train/prepare_data.py
   python3 nepali/train/prepare_data.py
   ```
2. **Pretrain** (AdamW, linear warmup + cosine decay, checkpoint every 1,000 steps,
   auto-resumes from the latest checkpoint if interrupted — both models were actually
   trained on Kaggle T4×2 GPUs via `kaggle_upload/phase2_train_eval.ipynb`, which wraps the
   same `Trainer` class with zero-hardcoded-path dataset auto-discovery):
   ```python
   from trainer import Trainer
   # construct GPTLanguageModel from configs/model_config.json, TokenBlockDataset from
   # data/tokenized/{train,val}.bin, then:
   trainer = Trainer(model, train_ds, val_ds, train_config, checkpoint_dir, device)
   trainer.train(resume=True, log_csv_path="statistics/phase2_eval_outputs/train_log.csv")
   ```
3. **Evaluate** intrinsic LM metrics, generation quality, and attention, against the trained
   checkpoint (inference only, no GPU strictly required for a 25M-parameter model):
   ```bash
   python3 hindi/eval/run_generation_eval.py   # real held-out BLEU/chrF/ROUGE-L + samples
   python3 nepali/eval/run_generation_eval.py
   # PPL/BPB, attention heatmaps/entropy/distance are produced inside
   # kaggle_upload/phase2_train_eval.ipynb's evaluation cells (see that notebook for the
   # exact calls into hindi/eval/lm_metrics.py and hindi/eval/attention_analysis.py)
   ```
4. **Generate the Model H vs. Model L comparison chart** (the one Phase 2 figure that needs
   both languages' results at once, so it's run locally after both trainings finish rather
   than inside either language's single-language Kaggle run):
   ```bash
   python3 report/phase2_plot/generate_ppl_bpb_comparison.py
   ```

Full results, all figures, and the complete written analysis are in
[`report/phase2_report.md`](report/phase2_report.md).

Phase 3 (reasoning finetuning and attention analysis) reproduction steps will be added here
once that phase is completed.

## Known limitations / pending work

- **Google Drive checkpoint links** — required by the assignment, not yet done (see "Model
  checkpoints" above). Kaggle dataset links are a working interim substitute.
- **Generation-quality evaluation** now uses real held-out prefix/reference pairs sampled from
  each language's `data/splits/test.txt` (fixed as of this report), scored under greedy
  decoding and temperatures 0.5/1.0/1.5 — see `report/phase2_report.md` Section 4 for the
  methodology and a concrete finding (a greedy-decoding mode-collapse pattern specific to
  Model L).
- **Attention-head local/content-based discussion** is written up quantitatively in
  `report/phase2_report.md` Section 5 (entropy/distance grid analysis), not just the raw
  heatmap images.
- Two leftover artifacts at the repo root, `kaggle_phase2_upload/` and
  `kaggle_phase2_upload.zip`, are an earlier, now-superseded packaging attempt (before the
  Kaggle-API-automation and auto-discovery setup described above) — not referenced by any
  current workflow, kept only pending a cleanup decision.
- Phase 3 (reasoning finetuning, post-finetune attention comparison, final consolidated
  report) has not started.
