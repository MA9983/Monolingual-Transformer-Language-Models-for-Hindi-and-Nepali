"""Clean, deduplicate, and split the Hindi corpus (IndicCorp download +
manually scraped news) into train/val/test. See
common/preprocess_corpus.py for the shared pipeline; this file only wires
up Hindi's specific source files and output paths.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from preprocess_corpus import run  # noqa: E402

HERE = os.path.dirname(__file__)

SOURCES = [
    {
        "name": "indiccorp_downloaded",
        "type": "downloaded",
        "granularity": "line",
        "path": os.path.join(HERE, "raw", "downloaded", "indiccorp", "hindi_indiccorp_raw.txt"),
    },
    {
        "name": "news_scrape_manual",
        "type": "manual",
        "granularity": "article",
        "path": os.path.join(HERE, "raw", "manual", "news_scrape", "scraped_articles.txt"),
    },
]

if __name__ == "__main__":
    run(
        language="hindi",
        sources=SOURCES,
        splits_dir=os.path.join(HERE, "splits"),
        processed_dir=os.path.join(HERE, "processed"),
        stats_path=os.path.join(HERE, "..", "statistics", "preprocessing_stats.json"),
    )
