"""Clean, deduplicate, and split the Hindi corpus (IndicCorp download +
manually scraped news) into train/val/test. See
common/preprocess_corpus.py for the shared pipeline; this file only wires
up Hindi's specific source files and output paths.

The downloaded source is capped at ~400M tokens (1.8B chars); the manual
news-scrape source is used in full (after cleaning, repetitive-document
removal, the safety filter, exact dedup, and MinHash near-dedup), landing
the corpus at ~507M tokens total with manual comfortably above the
assignment's 20% floor -- the real post-cleaning figures are in
hindi/statistics/preprocessing_stats.json.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from preprocess_corpus import run  # noqa: E402

HERE = os.path.dirname(__file__)
DOWNLOADED_CAP_CHARS = 400_000_000 * 4.5  # ~400M tokens

SOURCES = [
    {
        "name": "indiccorp_downloaded",
        "type": "downloaded",
        "granularity": "line",
        "path": os.path.join(HERE, "raw", "downloaded", "indiccorp", "hindi_indiccorp_raw.txt"),
        "max_kept_chars": DOWNLOADED_CAP_CHARS,
    },
    {
        "name": "news_scrape_manual",
        "type": "manual",
        "granularity": "article",
        "path": os.path.join(HERE, "raw", "manual", "news_scrape", "scraped_articles.txt"),
        "near_dedup": True,
    },
]

if __name__ == "__main__":
    run(
        language="hindi",
        sources=SOURCES,
        splits_dir=os.path.join(HERE, "splits"),
        processed_dir=os.path.join(HERE, "processed"),
        stats_path=os.path.join(HERE, "..", "statistics", "preprocessing_stats.json"),
        badwords_path=os.path.join(HERE, "..", "..", "common", "badwords_hindi.txt"),
    )
