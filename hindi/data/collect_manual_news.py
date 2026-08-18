"""Manual-collection portion of the Hindi corpus: scrape Hindi news
articles to satisfy the project's >=20% manual-data requirement.

Site list was chosen after manually inspecting each candidate's robots.txt.
Excluded: bbc.com/hindi (robots.txt states an explicit "no AI
training/dataset creation" policy), bhaskar.com and amarujala.com (robots.txt
explicitly disallows the `anthropic-ai` user agent), news18.com (robots.txt
explicitly disallows `ClaudeBot`/`Claude-Web`). The sites below have neither
restriction as of the check date (17 Aug 2026) and allow user-agent `*`
broadly; robots.txt is still enforced live and per-request by RobotsCache
inside news_scraper.py, not just at this static list-selection stage.

hindustantimes.com was tried and dropped: it is a primarily English
publication whose robots.txt only declares sitemaps for its Telugu/Bangla
editions (no Hindi-specific sitemap), so crawling it would mean fetching
tens of thousands of English pages just to have nearly all of them filtered
out by the language check -- wasteful for us and needless load on their
servers for content we'd discard anyway. livehindustan.com is the properly
Hindi-language sister publication and has genuinely deep dated archives
(observed: ~10,000 URLs per half-month sitemap, 363 historical sub-sitemaps
in its index), so it does the equivalent job.

Target: ~120M raw tokens (~540M characters at 4.5 chars/token) -- raised
from the original 110M cap per the user's follow-up instruction, after
110M was reached on 18 Aug 2026.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from news_scraper import NewsCollector  # noqa: E402

TARGET_TOKENS = 120_000_000
CHARS_PER_TOKEN = 4.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

SITES = [
    {"name": "jagran", "homepage": "https://www.jagran.com"},
    {"name": "abplive", "homepage": "https://www.abplive.com"},
    {"name": "livehindustan", "homepage": "https://www.livehindustan.com"},
    # Second pass: jagran/abplive/livehindustan's readily-discoverable
    # archives got mostly exhausted after many hours of scraping, so more
    # sites were added for fresh headroom. punjabkesari and aajtak add
    # real pools (punjabkesari has 15 declared sitemaps including several
    # regional ones; aajtak's newssitemap-style listing is shallower but
    # still real). Also checked and excluded: patrika.com (disallows
    # ClaudeBot and anthropic-ai), zeenews.india.com (disallows
    # ClaudeBot/claude-web), navbharattimes.indiatimes.com (only
    # category/section pages found, no deep article archive), samaylive.com
    # (unreachable).
    #
    # indiatv.in was tried and dropped per user instruction: despite 326
    # monthly sitemaps back to 2018-01 (a large discoverable pool) and
    # individually-fast article responses (0.2-0.3s when tested directly),
    # its worker threads repeatedly stalled mid-run in practice (per-site
    # chars going flat for many minutes at a time despite a large queue),
    # dragging down overall throughput. abplive and livehindustan cover the
    # same role reliably instead.
    {"name": "punjabkesari", "homepage": "https://www.punjabkesari.in"},
    {"name": "aajtak", "homepage": "https://www.aajtak.in"},
    # Third pass: jagran/abplive/livehindustan/punjabkesari/aajtak are now
    # mostly exhausted of backlog too (each restart only turns up a small
    # trickle of newly-published articles), so throughput without indiatv
    # had become bottlenecked by organic publication rate rather than
    # scraping speed. timesnowhindi.com has 37 real monthly sitemaps back
    # to 2023-08 and no AI-bot restrictions. Also checked and excluded:
    # jansatta.com (403 on robots.txt), haribhoomi.com and
    # hindi.oneindia.com (both disallow ClaudeBot/anthropic-ai),
    # livehindi.tv (unreachable).
    {"name": "timesnowhindi", "homepage": "https://www.timesnowhindi.com"},
    # Fourth pass: timesnowhindi's bulk queue also drained after carrying
    # most of the growth from 90M to 110M tokens. hindi.opindia.com has a
    # real WordPress post-sitemap archive (85 sub-sitemaps back to 2019)
    # and no AI-bot restrictions. thelallantop.com was checked and
    # dropped: its sitemap index is all topic/category/live-event
    # sitemaps, no dated post archive. ndtv (khabar.ndtv.com) 403s on
    # robots.txt; webdunia.com and dw.com both disallow anthropic-ai.
    {"name": "opindia_hindi", "homepage": "https://hindi.opindia.com"},
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw", "manual", "news_scrape")

if __name__ == "__main__":
    collector = NewsCollector(SITES, lang_code="hi", target_chars=TARGET_CHARS,
                               out_dir=OUT_DIR, chars_per_token=CHARS_PER_TOKEN)
    collector.collect()
