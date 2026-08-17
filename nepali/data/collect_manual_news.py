"""Manual-collection portion of the Nepali corpus: scrape Nepali news
articles to satisfy the project's >=20% manual-data requirement.

Site list was chosen after manually inspecting each candidate's robots.txt;
none of the sites below disallow AI/anthropic user agents or state an
anti-AI-training policy as of the check date (17 Aug 2026). robots.txt is
still enforced live and per-request by RobotsCache inside news_scraper.py,
not just at this static list-selection stage. Ratopati declares a 20s
crawl-delay, which RobotsCache picks up and enforces automatically, so it
will naturally contribute far fewer articles per hour than the others.

Target: same ~150M raw token (~675M character) ambition as Hindi, on a
best-effort basis -- Nepali news publishing volume online is smaller, so
this may fall short; the actual collected amount is reported honestly in
the Phase 1 statistics/report rather than padded.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from news_scraper import NewsCollector  # noqa: E402

TARGET_TOKENS = 150_000_000
CHARS_PER_TOKEN = 4.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

SITES = [
    {"name": "onlinekhabar", "homepage": "https://www.onlinekhabar.com"},
    {"name": "ekantipur", "homepage": "https://ekantipur.com"},
    {"name": "setopati", "homepage": "https://www.setopati.com"},
    {"name": "nagariknews", "homepage": "https://nagariknews.nagariknetwork.com"},
    {"name": "ratopati", "homepage": "https://ratopati.com"},
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw", "manual", "news_scrape")

if __name__ == "__main__":
    collector = NewsCollector(SITES, lang_code="ne", target_chars=TARGET_CHARS,
                               out_dir=OUT_DIR, chars_per_token=CHARS_PER_TOKEN)
    collector.collect()
