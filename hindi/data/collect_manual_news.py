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

Target: ~110M raw tokens (~495M characters at 4.5 chars/token) -- the
user's explicit hard cap for Hindi manual scraping.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from news_scraper import NewsCollector  # noqa: E402

TARGET_TOKENS = 110_000_000
CHARS_PER_TOKEN = 4.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

SITES = [
    {"name": "jagran", "homepage": "https://www.jagran.com"},
    {"name": "abplive", "homepage": "https://www.abplive.com"},
    {"name": "livehindustan", "homepage": "https://www.livehindustan.com"},
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw", "manual", "news_scrape")

if __name__ == "__main__":
    collector = NewsCollector(SITES, lang_code="hi", target_chars=TARGET_CHARS,
                               out_dir=OUT_DIR, chars_per_token=CHARS_PER_TOKEN)
    collector.collect()
