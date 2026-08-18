# Phase 1 Report — Data Collection and Tokenizer Construction

**Model H (higher-resource):** Hindi
**Model L (lower-resource):** Nepali

This report covers **data collection only** (downloaded + manually scraped), as collected
so far, **before the preprocessing/cleaning pass**. Raw-text character counts below are
converted to an estimated token count using a **4.5 characters/token** planning assumption
(Devanagari BPE fertility typically falls in the 4–5 chars/token range; the real figure
will be measured once the tokenizer is trained in a later step). Final, post-cleaning
statistics (deduplication, quality filtering, train/val/test splits, real tokenizer-based
token counts) will be reported separately once `hindi/data/preprocess.py` and
`nepali/data/preprocess.py` are run.

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

## Hindi — current raw totals

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

## Nepali — current raw totals

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

## Manual fraction — currently ~18%, not yet finalized

Both languages currently sit at **~18% manual** on a **raw, pre-cleaning** basis, just under
the assignment's 20% floor. This number is **not final**: `common/preprocess_corpus.py`
applies Unicode normalization, script/quality filtering, and deduplication independently to
every source, and downloaded vs. manual content is not guaranteed to be filtered out at the
same rate (e.g. IndicCorp's web-crawled text tends to contain more exact/near-duplicate
boilerplate than freshly-scraped news articles, which would raise the manual fraction after
cleaning; the reverse is possible too). The real, final manual fraction will only be known
once preprocessing has actually run, and will be reported in the Phase 1 dataset-statistics
update. A capping mechanism to force the post-cleaning ratio to a specific floor was
prototyped and explicitly **rejected** per user instruction — the current approach is to use
all collected downloaded and manual data as-is and report the real resulting fraction
honestly, rather than truncating either source.

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

- [ ] Run `hindi/data/preprocess.py` and `nepali/data/preprocess.py` to produce cleaned,
      deduplicated `train`/`val`/`test` splits and the real (post-cleaning) manual/downloaded
      token split.
- [ ] Train tokenizers (`hindi/tokenizer/train_tokenizer.py`,
      `nepali/tokenizer/train_tokenizer.py`) on the resulting `train` split and report
      vocabulary size, fertility, and UNK-rate statistics.
- [ ] Update this report (or add a companion `phase1_dataset_statistics.md`) with the final,
      post-cleaning numbers.
