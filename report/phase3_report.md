# Final Project Report: Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

This is the final, consolidated report for the whole project, submitted with Phase 3
(the final project deadline). It brings together the key results from all three phases —
data collection (Phase 1), model architecture/pretraining/evaluation (Phase 2), and
reasoning finetuning/attention analysis (Phase 3) — in plain language, with the important
figures and tables from each phase included directly. Full phase-by-phase detail remains in
[`phase1_report.md`](phase1_report.md) and [`phase2_report.md`](phase2_report.md) for anyone
who wants the complete write-up of a specific phase.

**Model H** = Hindi (higher-resource language). **Model L** = Nepali (lower-resource
language, from the assignment's allowed list). Both are completely independent decoder-only
Transformer language models, built from scratch — separate data, separate tokenizer, separate
weights, at every stage.

## Full project pipeline (preprocessing → training, all three phases)

Every stage below runs **twice, completely independently** — once for Hindi, once for
Nepali — with no shared data, vocabulary, or weights at any point.

```mermaid
flowchart TD
    A[Raw text sources<br/>IndicCorp v2 download + news/Wikipedia scraping] --> B[Cleaning & preprocessing<br/>Unicode normalization, quality filter,<br/>repetitive-doc removal, safety filter,<br/>exact + near-dedup]
    B --> C[Train / val / test split<br/>95% / 2.5% / 2.5%]
    C --> D[Tokenizer training<br/>SentencePiece BPE, vocab=10,000,<br/>own tokenizer per language]
    D --> E[Tokenize splits to .bin arrays<br/>prepare_data.py]
    E --> F[Pretraining from scratch<br/>decoder-only Transformer, RoPE,<br/>AdamW, 30,000 steps]
    F --> G[Pretrained checkpoint<br/>checkpoint_step30000.pt]
    G --> H[Phase 2 evaluation<br/>PPL / BPB, generation quality,<br/>attention analysis]
    G --> I[Synthetic reasoning dataset<br/>generate_reasoning_data.py<br/>pairwise / transitive3 / chain]
    I --> J[Reasoning finetuning<br/>loss masked to answer span,<br/>1,000 steps from the pretrained checkpoint]
    J --> K[Finetuned checkpoint<br/>finetune_step1000.pt]
    K --> L[Phase 3 evaluation<br/>pretrained-vs-finetuned accuracy,<br/>error analysis, attention comparison]
```

**Where each stage's code lives** (same layout for both `hindi/` and `nepali/`):

| Stage | Code | Phase |
|---|---|---|
| Data collection | `common/download_indiccorp.py`, `common/news_scraper.py`, `common/wikipedia_scraper.py` | 1 |
| Cleaning, splitting | `<lang>/data/preprocess.py` | 1 |
| Tokenizer training | `<lang>/tokenizer/train_tokenizer.py` | 1 |
| Tokenize to binary | `<lang>/train/prepare_data.py` | 2 |
| Model architecture | `<lang>/model/{embeddings,attention,block,rope,gpt}.py` | 2 |
| Pretraining | `<lang>/train/trainer.py`, driven by `kaggle_upload/phase2_train_eval.ipynb` | 2 |
| Pretrained-model evaluation | `<lang>/eval/{lm_metrics,generation_metrics,attention_analysis,plotting}.py` | 2 |
| Reasoning dataset generation | `<lang>/reasoning/generate_reasoning_data.py` | 3 |
| Finetuning | `<lang>/finetune/{dataset,finetune_trainer}.py`, `run_finetune.py` | 3 |
| Finetuned-model evaluation | `<lang>/finetune/{run_reasoning_eval,run_attention_comparison}.py` | 3 |

---

# Part I — Phase 1: Data Collection and Tokenizer Construction

Two independent text corpora were collected — one per language — combining a downloaded
source (IndicCorp v2) with manually scraped news and (for Nepali) Wikipedia text, then
cleaned, deduplicated, safety-filtered, and split.

## 1.0 Phase 1 pipeline

```mermaid
flowchart TD
    A1[IndicCorp v2 download] --> C[Combine sources]
    A2[News-site scraping] --> C
    A3[Wikipedia scraping<br/>Nepali only] --> C
    C --> D[Unicode NFC normalization +<br/>Devanagari script-ratio quality filter]
    D --> E[Repetitive-document removal<br/>Gopher-style top-n-gram ratio]
    E --> F[Safety filter<br/>profanity wordlist]
    F --> G[Exact deduplication<br/>content-hash]
    G --> H[Near-duplicate deduplication<br/>MinHash / LSH, manual sources only]
    H --> I[Train / val / test split<br/>95% / 2.5% / 2.5%]
    I --> J[Tokenizer vocab-size sweep<br/>5k/7k/9k/10k/16k/24k on held-out val]
    J --> K[Final tokenizer<br/>SentencePiece BPE, vocab=10,000]
```

## 1.1 Corpus statistics

| | Hindi (Model H) | Nepali (Model L) |
|---|---:|---:|
| Final corpus size | 506.81M tokens | 500.24M tokens |
| Manual fraction | 21.07% | 20.04% |
| Manual sources | News scrape | News scrape + Wikipedia |
| Train / val / test split | 95% / 2.5% / 2.5% | 95% / 2.5% / 2.5% |

Nepali needed a **second manual source (Wikipedia)** on top of news scraping, and its manual
fraction sits closer to the assignment's 20% floor — because MinHash near-duplicate removal
found substantially more real duplicate content in Nepali's manually-collected pool than in
Hindi's, meaning Nepali's *raw*, before-cleaning pool of genuinely distinct text was thinner
to begin with, even though the two final corpora ended up almost the same size.

![Final corpus composition: downloaded vs. manual tokens, both languages](plot/corpus_composition.png)
![Train/validation/test split sizes, both languages](plot/split_sizes.png)

## 1.2 Tokenizer

A separate SentencePiece BPE tokenizer was trained from scratch per language (never shared,
never pretrained), vocabulary size 10,000 for both, chosen by sweeping candidate sizes and
weighing diminishing fertility returns against the Phase 2 ~25M-parameter budget.

| | Hindi | Nepali |
|---|---:|---:|
| Vocabulary size | 10,000 | 10,000 |
| Fertility (tokens/word, held-out val) | 1.3227 | 1.4934 |
| Avg characters/token | 3.856 | 4.318 |
| UNK rate | 0.0% | 0.0% |

Both hit 0.0% UNK rate because `byte_fallback=True` was enabled — any character sequence the
tokenizer hasn't seen falls back to raw bytes instead of an unknown-token placeholder.
**Nepali's tokenizer needs more subword pieces per word (1.49 vs. 1.32)** at the same
vocabulary size — a real, measured difference that turns out to matter a lot in Phase 2's
perplexity comparison (Part II).

![Tokenizer fertility vs. vocabulary size sweep, both languages](plot/fertility_vs_vocab_size.png)

---

# Part II — Phase 2: Model Architecture, Pretraining, and Evaluation

## 2.0 Phase 2 pipeline

```mermaid
flowchart TD
    A[Phase 1 train/val/test splits +<br/>trained tokenizer] --> B[Tokenize splits to uint16<br/>.bin arrays, prepare_data.py]
    B --> C[Build GPTLanguageModel<br/>embeddings + RoPE + causal attention<br/>+ transformer blocks, from scratch]
    C --> D[Pretrain: AdamW, linear warmup<br/>+ cosine decay, 30,000 steps,<br/>checkpoint every 1,000 steps]
    D --> E[Pretrained checkpoint<br/>checkpoint_step30000.pt]
    E --> F1[Intrinsic LM eval<br/>perplexity, bits-per-byte]
    E --> F2[Generation quality eval<br/>BLEU-4, chrF++, ROUGE-L,<br/>greedy + temp 0.5/1.0/1.5]
    E --> F3[Attention analysis<br/>heatmaps, entropy, mean distance]
```

## 2.1 Architecture

A decoder-only (GPT-style) Transformer, built entirely from PyTorch primitives
(`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`) — no `nn.Transformer*`, no
`nn.MultiheadAttention`, no HuggingFace model classes. Both languages use the identical
architecture *shape*; only the trained weights differ.

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 10,000 (each language's own tokenizer) |
| `d_model` | 512 |
| `n_layers` | 6 |
| `n_heads` | 8 (`d_head` = 64) |
| `d_ff` | 2,208 |
| `context_length` | 256 |
| Positional encoding | RoPE (Rotary Position Embedding) |
| Activation | GELU |
| Norm placement | Pre-norm |
| Weight tying | Yes (input embedding ↔ output head) |
| **Exact parameter count** | **25,019,328** (both models) |

**Why RoPE**: instead of adding a position signal once at the input, RoPE rotates the
query/key vectors by an angle based on position, inside every attention layer — so relative
position is re-introduced fresh at every layer rather than only once at the start. Full
justification for every architecture choice (RoPE, pre-norm, weight tying, head count,
`d_ff` tuning, etc.) is in `phase2_report.md` Sections 1.1-1.6.

## 2.2 Pretraining

Both models trained independently with AdamW, linear warmup into cosine-decay learning rate,
for **30,000 steps** (≈491.5M tokens processed — roughly one full pass over each ~500M-token
corpus), with checkpointing every 1,000 steps in a resume-capable format (weights, optimizer
state, scheduler state, training step, config).

![Hindi pretraining loss curve](../hindi/statistics/phase2_eval_outputs/loss_curve.png)
![Nepali pretraining loss curve](../nepali/statistics/phase2_eval_outputs/loss_curve.png)

Both curves start near `ln(10,000) ≈ 9.2` nats (a completely untrained, randomly-guessing
model) and fall to a plateau by the end of training, with train and validation loss staying
close throughout — no overfitting, expected for a 25M-parameter model on a ~500M-token corpus.

## 2.3 Language-modeling evaluation

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Validation perplexity | 24.24 | 40.91 |
| Validation bits-per-byte (BPB) | 0.4665 | 0.4688 |

![Model H vs Model L: validation perplexity and bits-per-byte](../report/phase2_plot/ppl_bpb_comparison.png)

**The perplexity gap is misleading on its own.** Perplexity is measured per *token*, and
Nepali's tokenizer needs more tokens per word (Part I) — so each Nepali token is a smaller,
easier-feeling unit to guess wrong, inflating its per-token perplexity. Bits-per-byte
measures the same thing per *byte* instead, which is fair regardless of how each tokenizer
happens to chop up the text — and there, Hindi and Nepali are under 0.5% apart. **The two
models are nearly equally good language models; the raw perplexity gap is mostly a
tokenizer artifact, not a real capability gap.**

## 2.4 Generation quality

Real held-out prefixes sampled from each language's `test.txt`, scored under greedy decoding
and temperature sampling (0.5 / 1.0 / 1.5):

| Setting (greedy) | Hindi | Nepali |
|---|---:|---:|
| BLEU-4 | 38.34 | 37.42 |
| chrF++ | 48.94 | 48.77 |
| ROUGE-L F1 | 0.242 | 0.067 |
| Distinct-1 / Distinct-2 | 0.300 / 0.548 | 0.381 / 0.556 |

chrF++ (character-level, script-agnostic) is nearly tied — consistent with the BPB finding
above. ROUGE-L is much lower for Nepali, but this traces to a specific finding: Nepali's
**greedy** decoding tends to fall back on one dominant, frequently-repeated training-corpus
sentence regardless of the prompt (a decoding-time "mode collapse"), not a general fluency
problem — temperature-sampled Nepali generations are visibly more varied. Full generated
samples and discussion are in `phase2_report.md` Section 4.

## 2.5 Attention analysis (pretrained)

![Hindi attention entropy per layer/head](../hindi/statistics/phase2_eval_outputs/attention_entropy.png)
![Hindi mean attention distance per layer/head](../hindi/statistics/phase2_eval_outputs/attention_distance.png)

Both models show an early-layer-local, later-layer-broader attention pattern (early layers
attend to nearby positions, later layers reach further). Full per-head breakdown is in
`phase2_report.md` Section 5. This is the *pretrained* baseline that Part III below compares
against after Phase 3 finetuning.

---

# Part III — Phase 3: Reasoning Finetuning and Attention Analysis

## 3.0 Phase 3 pipeline

```mermaid
flowchart TD
    A[Generate synthetic reasoning dataset<br/>pairwise / transitive3 / chain<br/>held-out entity names for test] --> B[Train / val / test .jsonl<br/>4,000 / 500 / 500 examples]
    B --> C[Load Phase 2 pretrained checkpoint<br/>tokenizer/vocab kept fixed]
    C --> D[Finetune: loss masked to<br/>answer span only, 1,000 steps]
    D --> E[Finetuned checkpoint<br/>finetune_step1000.pt]
    E --> F1[Reasoning accuracy eval<br/>pretrained vs. finetuned,<br/>exact match on held-out test set]
    F1 --> F1b[Error analysis<br/>failure patterns, value-gap correlation]
    E --> F2[Attention comparison<br/>pretrained vs. finetuned heatmaps,<br/>entropy, mean distance]
```

## 3.1 What is the "synthetic dataset" and what did we build?

**"Synthetic" just means made up by a script, not collected from the real world.** Instead of
finding real comparison sentences on the internet, a program invents its own practice
questions — it picks two or three made-up people (or objects), gives them random numbers
(height, age, price, or quantity), and asks a comparison question. Because the script itself
chose the numbers, it always knows the correct answer — no manual labeling needed.

Example (Hindi): *"अशोक की ऊंचाई 160 सेमी है, जबकि मोहन की ऊंचाई 152 सेमी है। इनमें से कौन ज्यादा
लंबा है?"* ("Ashok's height is 160cm, Mohan's is 152cm. Who is taller?") → correct answer:
अशोक (Ashok).

Three question types, matching the assignment's own examples, each drawing its numbers from
one of four attributes — height (140-200cm), age (5-90 years), price (10-10,000 rupees), or
quantity (1-100) — with height/age asked about people and price/quantity about everyday
objects (book, bicycle, mango, etc.), so the same three reasoning patterns get tested across
genuinely different real-world contexts, not just one repeated scenario:

1. **Pairwise** — compare two entities on one attribute.
   *"अशोक की ऊंचाई 160 सेमी है, जबकि मोहन की ऊंचाई 152 सेमी है। इनमें से कौन ज्यादा लंबा है?"*
   → अशोक (Ashok is taller)
2. **Transitive3** — given three entities' values, which is biggest/smallest?
   *"नेहा की लंबाई 151 सेमी, ललिता की लंबाई 195 सेमी, और अशोक की लंबाई 170 सेमी है। इनमें सबसे लंबा
   कौन है?"* → ललिता (Lalita, the tallest of the three)
3. **Chain** — given "A is bigger than B" and "B is bigger than C" as two separate facts,
   which is bigger, A or C? (needs two logical steps, not just reading one number off the page)
   *"गौरव संजय से बड़ा है। संजय नेहा से बड़ा है। गौरव और नेहा में से कौन बड़ा है?"* → गौरव (chaining
   the two given facts: Gaurav > Sanjay > Neha, so Gaurav > Neha)

| | Hindi | Nepali |
|---|---:|---:|
| Train / val / test examples | 4,000 / 500 / 500 | 4,000 / 500 / 500 |
| Task-type split (train+val+test) | pairwise 1,706 / transitive3 1,596 / chain 1,698 | same distribution |
| Distinct template sentences | 16 (8 pairwise, 4 transitive3, 4 chain) | 16 (same breakdown) |
| Attributes | height, age, price, quantity | height, age, price, quantity |
| Name pool | 50 people + 20 objects | 50 people + 20 objects |
| Train/test name overlap | **0** | **0** |

**Why entity names, specifically, were held out (rather than holding out a whole template or
relation pattern)**: the assignment allows either method. Names were chosen here because the
*reasoning pattern itself* (how to compare two numbers, how to chain two facts) is exactly the
skill the finetuning is meant to teach — holding out a whole template or pattern would mean
the test set checks something the model was never even shown *how* to do, which tests
something closer to zero-shot generalization to a new task than to "did finetuning teach the
intended skill." Holding out names instead keeps the *patterns* fully covered in training
while still making it impossible for the model to pass by memorizing "this specific name pair
→ this specific answer" — it has to generalize the comparison logic to entities it has never
seen.

Both languages have their own independent generator, name pool, and natural phrasing (not
translated from each other) — `hindi/reasoning/generate_reasoning_data.py`,
`nepali/reasoning/generate_reasoning_data.py`, both deterministic (seed 1337), so re-running
them reproduces the exact same dataset.

## 3.2 How finetuning worked

Starting from each language's Phase 2 pretrained checkpoint, tokenizer/vocabulary kept fixed
throughout (as required), model architecture completely unchanged from Phase 2.

**How the "only train on the answer" masking actually works**: each example is tokenized as
`[question tokens][answer tokens][end-of-sentence marker]`, one continuous sequence, exactly
as the model reads it during generation. But when computing the training loss, every position
that would be predicting a *question* token is excluded — the model is never penalized for how
it "would have" continued the question itself, only for whether it gets the answer right,
given the whole question as context. Concretely: for the example *"...कौन लंबा है? → अशोक"*,
loss is computed only on the token(s) making up "अशोक" plus the closing marker — nothing
earlier. This matters because the question text is something the model already knows how to
produce fluently (it's ordinary language, same as Phase 2 training); wasting training signal
re-teaching it to write the question would dilute the signal actually meant to teach the new
skill.

| Finetuning hyperparameter | Value | Why |
|---|---|---|
| Learning rate | 1e-4 | Lower than pretraining's 3e-4 — finetuning an already-good model needs smaller updates to avoid overwriting what it already knows |
| Weight decay | 0.01 | Lighter than pretraining's 0.1 — less regularization needed for a short, narrow finetuning run |
| Batch size | 32 | Smaller than pretraining's 64 — reasonable for short, variable-length examples on CPU |
| Epochs / total steps | 8 / 1,000 | 4,000 examples × 8 passes ÷ batch 32 = 1,000 steps — enough repetition for a small templated dataset to converge, nowhere near pretraining's compute |
| Warmup steps | 50 | Short warmup appropriate for a short run (vs. pretraining's 1,000) |
| Checkpoint interval | every 250 steps | Same resume-capable format as pretraining (weights, optimizer, scheduler, step, config) |

![Hindi finetuning loss curve](../hindi/statistics/phase3_finetune_outputs/finetune_loss_curve.png)
![Nepali finetuning loss curve](../nepali/statistics/phase3_finetune_outputs/finetune_loss_curve.png)

Training loss falls to near-zero (Hindi 0.0375, Nepali 0.0077) while validation loss plateaus
around 0.21-0.22 and stays flat for both — expected, since the answer is always one short
name from a small closed set (an easier target than ordinary language modeling), and the flat
(not worsening) validation loss shows the model isn't badly overfitting. The real test of
whether it learned to *reason*, rather than memorize, is the held-out test accuracy below.

## 3.3 Reasoning accuracy: pretrained vs. finetuned

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| **Pretrained-only accuracy** | 0.0% | 0.0% |
| **Finetuned accuracy** | **65.0%** | **43.4%** |
| — pairwise | 50.8% | 42.1% |
| — transitive3 | 41.4% | 42.1% |
| — chain | 100.0% | 45.9% |

**How accuracy was measured**: for each of the 500 held-out test questions, the finetuned
model generates an answer with greedy decoding (always picks its single most likely next
token — no randomness, so the score is fully reproducible), the output is cut at the first
end-of-sentence marker, and it's compared to the correct answer with **exact string match** —
no partial credit, no fuzzy matching. 500 test examples per language is a large enough sample
that the overall accuracy numbers (65.0%, 43.4%) are stable, meaningful percentages, not
noise from a handful of lucky or unlucky guesses.

Both models go from 0% (pretrained models were never taught to answer with a bare short
name) to a clearly-above-chance accuracy **on entity names never seen during training** —
real evidence of learned reasoning, not memorization. Hindi does noticeably better overall,
especially on chain reasoning (100% vs. 46%) — plausibly because chain questions use only 4
fixed template sentences (very repetitive structure) that Hindi's finetuning locked onto
completely, while the more numerically-varied pairwise/transitive3 tasks remain harder for
both models. Neither model gets close to 100% on the harder tasks, which matches the
well-documented general weakness of small (25M-parameter) language models at precise
multi-entity numeric comparison.

## 3.4 Error analysis

Failed test examples show a **clear, consistent mistake pattern**, not random garbage:

| Language | Prompt (shortened) | Correct answer | Model's answer |
|---|---|---|---|
| Hindi | संजय (199cm), गौरव (140cm), अशोक (179cm) — सबसे लंबा कौन? | संजय | राजेश *(not in the prompt!)* |
| Hindi | अशोक (160cm) vs मोहन (152cm) — कौन लंबा? | अशोक | राजेश *(not in the prompt!)* |
| Nepali | विकास (199cm), अनिता (140cm), नारायण (179cm) — सबैभन्दा अग्लो? | विकास | सीता *(not in the prompt!)* |

**In every failure examined, the model's wrong answer is a real, valid-looking name — but one
that doesn't even appear in that specific question.** This is a **grounding failure, not a
format failure**: the model reliably learned *what kind* of answer to give (always a single
name), but on some unseen name combinations it fails to correctly track *which* entity from
*this particular* question the numbers belonged to — especially with three entities to track
(transitive3) instead of two.

**A second, quantitative pattern: the model struggles more when the compared numbers are
close together.** Measuring the gap between the two largest values in every pairwise test
question:

| | Mean value gap when correct | Mean value gap when wrong |
|---|---:|---:|
| Hindi | 1,289.9 | 525.4 |
| Nepali | 1,571.7 | 436.2 |

Correct answers have a gap roughly **2.5x (Hindi) to 3.6x (Nepali)** larger than wrong ones.
This makes intuitive sense: if two heights are 199cm vs. 140cm, the difference is obvious;
if they're 178cm vs. 174cm, distinguishing them correctly requires more precise numeric
comparison, which is exactly where a 25M-parameter model's numeric-reasoning limits show up.
Between this and the grounding-failure pattern above, the picture is consistent: the model
has learned the *task* correctly, and fails specifically on the *hardest instances of it* —
close numeric comparisons and (for transitive3) tracking three entities at once — rather than
failing randomly or misunderstanding what's being asked.

## 3.5 Attention analysis: pretrained vs. finetuned

Using the same attention toolkit from Phase 2 (Part II.5), compared on the same
comparative-reasoning sentence, before and after finetuning:

| | Hindi: pretrained → finetuned | Nepali: pretrained → finetuned |
|---|---|---|
| Mean attention entropy (lower = more focused) | 2.168 → 1.932 bits (↓) | 2.380 → 2.020 bits (↓) |
| Mean attention distance (higher = looks further back) | 4.222 → 4.690 (↑) | 4.514 → 5.269 (↑) |

![Hindi pretrained attention heatmap, layer 0](../hindi/statistics/phase3_finetune_outputs/pretrained_attn_heatmap_layer0_head0.png)
![Hindi finetuned attention heatmap, layer 0](../hindi/statistics/phase3_finetune_outputs/finetuned_attn_heatmap_layer0_head0.png)

**The same shift happens in both languages**: finetuning makes attention **more focused**
(entropy drops) **and reaches further back** (distance increases) — the model learned to
hunt more precisely for the two specific numbers being compared, wherever they sit in the
sentence. Nepali's shift is actually *larger* than Hindi's in both measurements, even though
Hindi ends up more accurate — meaning Nepali's attention adapted at least as much to the
task, but its slightly weaker starting language model (Part II.3) limited how well that
adaptation translated into correct final answers.

**Breaking this down layer by layer** reveals the two languages don't just differ in *how
much* attention shifted, but *where*:

| Layer | Hindi distance shift | Nepali distance shift |
|---:|---:|---:|
| 0 (earliest) | +0.32 | +0.20 |
| 1 | -0.39 | -0.12 |
| 2 | **+0.95** | +0.96 |
| 3 | **+1.27** | +0.76 |
| 4 | +1.01 | +0.19 |
| 5 (latest) | -0.34 | **+2.54** |

**Hindi's biggest adaptation is concentrated in the middle layers (2-3)** — distance jumps by
around +1.0 to +1.3 there, while the very first and last layers barely change or even shrink
slightly. **Nepali's biggest adaptation is concentrated almost entirely in the last layer**
(+2.54, by far the single largest shift either language shows anywhere), with the middle
layers changing more modestly. This is a genuine structural difference in *how* the two
models learned to solve the same task: Hindi appears to have built the "find and compare the
two numbers" behavior into its middle layers, closer to where the numbers are first read,
while Nepali pushed most of that adaptation all the way to its final layer, right before
producing the answer — a less distributed, more last-minute strategy that may be part of why
it's less reliable (Section 3.3) despite adapting its attention just as much overall.

## 3.6 So, in plain terms — what actually happened in Phase 3?

Stepping back from the numbers and tables above, here's the short version of the story:

- **Before this phase, neither model could do this at all.** Ask the plain pretrained model
  "who is taller, X or Y?" and it wouldn't reliably answer with just a name — it would just
  keep writing more sentences, the way it was trained to. That's the 0% starting point.
- **A small amount of focused extra training (about half an hour on an ordinary computer, no
  special hardware) was enough to teach both models the skill**, and — this is the important
  part — it worked on people and objects the models had genuinely never seen before, not just
  on the exact examples they were trained on. That's real learning, not the model "cheating"
  by memorizing a lookup table of names and answers.
- **Neither model became perfect at it**, and when they get it wrong, they don't fail
  randomly — they consistently give a real, plausible-sounding name that just happens to be
  the *wrong* one for that specific question, and they're more likely to make this mistake
  when the two numbers being compared are close together (e.g. 178cm vs. 174cm) than when
  they're far apart (e.g. 199cm vs. 140cm). In other words: the models understand the *task*
  correctly; they sometimes lose track of *which specific thing* the numbers were about,
  especially when the comparison itself is a close call.
- **Hindi ended up noticeably better at this than Nepali** (65% vs. 43% correct), which lines
  up with something we already knew from Phase 2 — Hindi's underlying model was slightly
  better at the language in general, and that small head start carried over into learning
  this new skill too.
- **Looking inside the models, both actually changed in the same sensible way**: after this
  training, both started paying attention more sharply to specific parts of the sentence
  (rather than spreading attention evenly over everything) and reaching further back to
  connect the two numbers being compared — exactly the kind of change you'd hope to see if a
  model is genuinely learning to "find the two relevant numbers and compare them," rather
  than picking up some unrelated shortcut.

Put simply: **the extra training worked, it generalized to new examples, its mistakes make
sense rather than looking random, and the stronger-language model (Hindi) turned that
advantage into better reasoning too** — a clean, consistent, evidence-backed picture across
every angle this phase looked at (accuracy, errors, and attention).

---

# Part IV — Final Synthesis: Answering the Assignment's Four Required Questions

### 1. How did data scale and quality differ between Model H and Model L?

Final corpus sizes ended up almost identical (506.81M vs. 500.24M tokens, under 1.3% apart)
— but only because Nepali needed substantially more manual-collection effort (news scraping
*plus* a Wikipedia crawl, vs. news scraping alone for Hindi) to reach that size, and even
then its manually-collected text had a higher near-duplicate rate (~6.2% on Wikipedia) than
Hindi's. The underlying *availability* of distinct Nepali text online is genuinely smaller;
matching final token counts required more work, not less real scarcity.

### 2. How do language-modeling and reasoning results compare across the two resource tiers?

On the fair, tokenizer-independent metric (bits-per-byte) and on chrF++ generation quality,
Hindi and Nepali are **nearly tied** — under 0.5% and under 0.2 points apart respectively.
Where they clearly diverge is **reasoning**: Hindi's finetuned accuracy (65.0%) is well above
Nepali's (43.4%). Raw perplexity (24.24 vs. 40.91) looks like a bigger gap than the
underlying models actually have.

### 3. What tokenizer/corpus factors most affected the lower-resource model?

Two concrete, measured factors: **(a) tokenizer fertility** — Nepali needs 1.4934
tokens/word vs. Hindi's 1.3227 at the same 10,000-word vocabulary, inflating perplexity
without a real quality gap; **(b) source-text redundancy** — Nepali's manually-collected
pool had a measurably higher near-duplicate rate, pointing to a genuinely thinner pool of
distinct available text for the lower-resource language.

### 4. What evidence explains the observed differences?

The perplexity gap is explained by tokenizer fertility, not model quality (BPB proves this).
The reasoning-accuracy gap tracks with Hindi's small remaining LM-quality edge after
accounting for tokenization, giving finetuning a slightly stronger foundation to build on.
Attention evidence shows both models adapted in the *same direction* during finetuning (more
focused, longer-range) — Nepali's adaptation was, if anything, larger — so the accuracy gap
isn't because Nepali failed to learn the right attention pattern, but because its weaker base
language model translated that adaptation into correct answers less reliably. Finally, the
error analysis shows both languages fail in the *same specific way* (right answer format,
wrong specific entity) — a shared limitation of a 25M-parameter model doing precise
multi-entity tracking, not a language-specific weakness.

---

# Deliverables checklist (whole project)

- Dataset collection and preprocessing code (both languages) — Phase 1
- Separate tokenizers and vocabularies (both languages) — Phase 1
- Transformer implementation (MHA, positional embeddings, causal mask, student-implemented) — Phase 2
- Google Drive links to pretrained checkpoints (H and L) — in `README.md`
- Google Drive links to finetuned checkpoints (H and L) — **still pending upload**, checkpoints exist locally (gitignored)
- Training and finetuning logs — `*/statistics/phase2_eval_outputs/`, `*/statistics/phase3_finetune_outputs/`
- Evaluation scripts and results — PPL/BPB, generation quality, reasoning accuracy, all with real held-out data
-  Attention visualizations — pretrained (Phase 2) and pretrained-vs-finetuned (Phase 3), both languages
-  Final report (this document)
-  README with reproduction steps and all Drive links — Phase 2 steps done; Phase 3 steps + finetuned-checkpoint links still to be added

# Reproducibility (whole project)

**Phase 1** (data + tokenizer): see `phase1_report.md` and the top-level `README.md`.

**Phase 2** (architecture, pretraining, evaluation): see `phase2_report.md` and the
top-level `README.md`.

**Phase 3** (this phase):

```bash
# 1. Generate the synthetic reasoning dataset (deterministic, seed=1337)
python3 hindi/reasoning/generate_reasoning_data.py
python3 nepali/reasoning/generate_reasoning_data.py

# 2. Finetune from the Phase 2 pretrained checkpoint
python3 hindi/finetune/run_finetune.py
python3 nepali/finetune/run_finetune.py

# 3. Evaluate pretrained vs. finetuned accuracy on the held-out test set
python3 hindi/finetune/run_reasoning_eval.py
python3 nepali/finetune/run_reasoning_eval.py

# 4. Compare pretrained vs. finetuned attention on a reasoning prompt
python3 hindi/finetune/run_attention_comparison.py
python3 nepali/finetune/run_attention_comparison.py
```

All outputs land in `*/statistics/phase3_finetune_outputs/`. Finetuning configs (learning
rate, epochs, etc.) are documented in `*/finetune/finetune_config.json`.

---

# Bonus (Optional) — Ablation: No Positional Embeddings

Per the assignment's optional bonus, one language (**Hindi**) was retrained with positional
information removed entirely, and compared against the standard model on the full Phase 2
evaluation suite. This is kept in a separate top-level `bonus/` folder — not inside `hindi/`
— so it's never confused with the main graded architecture; per the assignment, this ablation
is explicitly *not* part of the main graded project.

## B.1 What was actually changed

**Exactly one thing, and nothing else.** `bonus/model/attention.py` is a copy of
`hindi/model/attention.py` with the RoPE rotation deleted — no `rope.py` import, no
`self.rope`, no rotation applied to Q/K before the attention scores are computed. Every other
line (Q/K/V projections, head reshaping, `1/sqrt(d_head)` scaling, causal masking, output
projection) is unchanged, and every other file (`embeddings.py`, `block.py`, `gpt.py`) is
untouched. `d_model=512`, `n_layers=6`, `n_heads=8`, `d_ff=2208`, `context_length=256`, same
tokenizer, same Hindi training data, same training hyperparameters (30,000 steps, AdamW,
warmup+cosine) as the standard model — the only variable that differs is whether queries and
keys carry any positional signal at all.

**Parameter count**: identical, **25,019,328** — RoPE has zero learnable parameters (it's a
fixed rotation computed from position, not a trained weight), so removing it doesn't change
the model's capacity or size in any way. This keeps the comparison to genuinely one variable.

**What breaks in principle, before even running it**: self-attention itself has no built-in
sense of order — computing `softmax(QK^T / sqrt(d_head))V` treats the input as an unordered
*set* of tokens, not a *sequence*, unless something (RoPE, learned positional embeddings,
sinusoidal encodings) tells it where each token sits. With that removed, the model can still
use word *identity* and *co-occurrence* to predict the next token (which is why it won't fail
completely), but it loses any notion of *word order*, *distance between words*, or *which
occurrence of a repeated word* is being referred to — for example, it cannot distinguish
"राम ने श्याम को मारा" (Ram hit Shyam) from "श्याम ने राम को मारा" (Shyam hit Ram): both contain
the identical set of tokens, and only word order tells you who did what to whom.

## B.2 Results: standard Hindi model vs. no-positional-embeddings ablation

Both models trained for the identical 30,000 steps on the identical Hindi data.

| Metric | Standard Hindi | No-positional-embeddings ablation | Change |
|---|---:|---:|---:|
| Perplexity | 24.24 | 27.23 | +12% (mild) |
| Bits-per-byte | 0.4665 | 0.4836 | +3.7% (mild) |
| **BLEU-4** (greedy) | 38.34 | **4.21** | **−89% (collapse)** |
| **chrF++** (greedy) | 48.94 | **25.38** | **−48% (collapse)** |
| **ROUGE-L** (greedy) | 0.242 | **0.0** | **complete collapse** |
| Distinct-1 / Distinct-2 | 0.300 / 0.548 | 0.571 / 0.792 | higher (see B.3 — not a genuine improvement) |
| Repetition rate (4-gram) | 0.013 | 0.036 | higher |
| Mean attention entropy | 1.046 bits | 1.079 bits | ~unchanged |
| Mean attention distance | ~0.99 | 1.011 | ~unchanged |

**The split is stark and informative**: perplexity and bits-per-byte — both *average, single-next-token*
prediction quality measures — degrade only mildly. But BLEU-4, chrF++, and ROUGE-L — all of
which score an entire *multi-token generated sequence* against a reference — collapse
completely. This gap is itself evidence for what position information is actually for: it
barely matters for guessing the *single* most likely next word most of the time (which is
often obvious from local word co-occurrence alone), but is essential for producing a
*coherent multi-word sequence* that holds together over many steps.

**Qualitative sample (greedy decoding, ablated model)**:
> प्रधानमंत्री ने आज → *"...लखनऊ लखनऊ लखनऊ लखनऊ में कोरोना से संक्रमित मरीजों की संख्या..."*

The model repeats **"लखनऊ" (Lucknow) four times in a row** — something the standard model
never does. This is a direct, visible symptom: without a positional signal, the model has no
reliable way to track "I already just produced this word," so it can loop on a locally
plausible word repeatedly. This also explains the counter-intuitive **higher** Distinct-1/2
scores in the table above: those metrics count *unique* n-grams over the *whole* generated
sample, and don't penalize *local, bursty* repetition (four "लखनऊ" in a row still only costs
one repeated n-gram in a global uniqueness count) — so "more diverse but also more
repetitive" is not a contradiction, it describes the same degenerate pattern from two angles.

**The genuinely surprising finding — attention heatmaps look qualitatively similar anyway**:

![Ablated model, layer 0 head 0](../bonus/statistics/attn_heatmap_layer0_head0.png)
![Ablated model, layer 5 head 0](../bonus/statistics/attn_heatmap_layer5_head0.png)

Both the aggregate entropy/distance numbers *and* the heatmap shapes for the ablated model are
close to the standard model's (Part II.5) — early-layer near-self-attention, late-layer
routing back to the sentence's first token (भारत). This is not what a naive prediction would
expect ("no position → attention pattern should look random/uniform"), and it has a concrete
explanation: **causal masking itself leaks a weak, implicit positional signal**, even with zero
explicit positional encoding. A query at the *first* position can only ever see one key (itself);
a query at the *last* position can see all five — the sheer number of valid keys available differs
by position, which is itself a (crude) function of position. On a short, 5-token example sentence,
that alone is enough for the model to partially reconstruct "early vs. late in the sentence,"
producing a heatmap that looks superficially similar to the standard model's. What this
implicit signal is clearly *not* enough for is precise word-order tracking over a longer
generated sequence — which is exactly where BLEU/chrF/ROUGE and the repetition failure above
show the real breakdown.

**Breaking the comparison down layer by layer** shows the aggregate numbers above are hiding a
real, systematic shift, not just noise:

| Layer | Entropy: standard → ablated | Distance: standard → ablated |
|---:|---:|---:|
| 0 (earliest) | 0.913 → **0.656** (lower) | 0.885 → **0.440** (much lower) |
| 1 | 1.052 → 1.126 | 0.919 → 0.983 |
| 2 | 1.043 → **1.247** (higher) | 0.895 → **1.131** (higher) |
| 3 | 1.132 → **1.247** (higher) | 1.102 → **1.223** (higher) |
| 4 | 1.068 → 1.077 | 1.089 → 1.075 |
| 5 (latest) | 1.067 → 1.124 | 1.072 → 1.210 |

Without positional embeddings, **layer 0 becomes even more tightly self-focused** than in the
standard model (distance nearly halves, 0.885→0.440) — with no positional cue to work with,
the earliest layer has even less basis for looking anywhere but the current token itself.
**Layers 2-3 compensate by becoming more diffuse and longer-range** than their standard-model
counterparts, as if searching more broadly for content-based associations to substitute for
the relative-position shortcut RoPE would normally provide. So the model isn't just "attending
the same way, minus position" — it's visibly *reorganizing which layers do what* to partially
compensate, even though (per B.2) that compensation isn't enough to rescue generation quality.

## B.3 What broke without position information

Bringing B.1's theoretical prediction and B.2's actual results together:

1. **Word-order sensitivity is gone.** Self-attention without any positional signal treats a
   sentence as an unordered set of tokens — the model can still use *which words co-occur*, but
   not *in what order*, so it cannot reliably distinguish "राम ने श्याम को मारा" from
   "श्याम ने राम को मारा." This doesn't show up much in perplexity (predicting *a* plausible
   next word from context is still largely solvable from co-occurrence alone) but shows up
   completely in BLEU/chrF/ROUGE-L, which specifically score whether generated text matches a
   reference **in the right order** — hence the 89% BLEU-4 collapse despite only a 12%
   perplexity increase.
2. **The model can't reliably track "what have I already said."** The repeated
   "लखनऊ लखनऊ लखनऊ लखनऊ" failure is the direct, visible symptom: normally, position lets the
   model represent "this is the 4th word I've generated" distinctly from "this is the 5th," so
   it doesn't confuse where it is in its own output. Without that, the model can loop on a
   locally-plausible word.
3. **What *doesn't* break**: local, single-step word prediction stays largely intact (mild PPL/BPB
   degradation), and even the attention *pattern* on a short example looks superficially similar
   to the standard model, because causal masking alone leaks a weak implicit positional cue (the
   number of valid keys a query can see grows with position). This is not a strong enough signal
   to support coherent multi-step generation, but it is enough to partially mimic the standard
   model's early-local/late-global attention shape on short inputs — a genuinely informative,
   non-obvious result: the *headline* failure of this ablation is in **generation quality**, not
   in the attention visualization itself.

**Bottom line**: positional embeddings are not needed for a Transformer to guess plausible individual
words from context, but they are essential for producing fluent, non-repetitive, correctly-ordered
multi-word text — exactly the distinction this ablation makes directly visible, on the same data,
same architecture otherwise, same training budget as the standard Hindi model.

## B.4 Reproducibility

```bash
# Ablated architecture (RoPE deleted from attention.py, everything else identical)
bonus/model/{embeddings,attention,block,gpt}.py

# Trained on Kaggle (same hindi-processed-dataset, same training_config.json, 30,000 steps):
# kaggle_upload/kernel_push_bonus/, using bonus/train/{prepare_data,dataset,trainer}.py

# Post-training evaluation (PPL/BPB, generation quality) ran inside the same Kaggle notebook,
# reusing bonus/eval/{lm_metrics,generation_metrics}.py unchanged.

# Attention analysis (re-run locally against the downloaded checkpoint):
python3 bonus/eval/run_attention_analysis.py
```

Checkpoint: `bonus/checkpoints/checkpoint_step30000.pt` (gitignored, same schema as the
standard model's checkpoints). Full config: `bonus/configs/model_config.json`.

## B.5 In plain terms — what this experiment actually shows

- Take the real, working Hindi model from Phase 2, delete exactly one thing (the mechanism
  that tells it where each word sits in a sentence), change nothing else, and train it the
  same way on the same data. What happens?
- **It still sounds okay for a single next word.** Ask it to guess one plausible next word
  given some context, and it does that only slightly worse than before (perplexity up ~12%).
  That's because guessing one likely word from nearby words doesn't strictly require knowing
  word *order* — just which words tend to appear near each other.
- **But ask it to write several words in a row, and it falls apart.** The standard model's
  BLEU score (how closely generated text matches a real reference, word by word, in order)
  was 38.3; this one scores 4.2 — a collapse, not a dip. And watching it generate text directly
  shows exactly why: it gets stuck repeating the same word four times in a row
  ("लखनऊ लखनऊ लखनऊ लखनऊ"), because without a sense of position it can't reliably tell that it
  already just said that word.
- **Even its internal attention reorganizes to cope** — the earliest layer becomes even more
  narrowly self-focused, while the middle layers spread out further, as if trying to find
  another way to recover some sense of structure. It's a real, visible adaptation — just not
  one that's good enough to fix the underlying problem.
- **The takeaway**: positional information isn't needed for a language model to sound locally
  fluent for a word or two, but it's essential for holding a whole sentence together correctly
  — which is exactly the kind of thing this single, controlled change makes directly visible,
  rather than just theoretically expected.
