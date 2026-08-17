"""Clean, deduplicate, and split the Nepali corpus (IndicCorp download +
manually scraped news) into train/val/test. See
common/preprocess_corpus.py for the shared pipeline; this file only wires
up Nepali's specific source files and output paths.

Note the quality filter in text_cleaning.is_valid_line() checks for
Devanagari-script ratio, which is correct for Nepali too (Nepali is
written in Devanagari, same Unicode block as Hindi) -- it is a script
check, not a language check, so it does not conflate the two languages'
text or vocabularies.
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
        "path": os.path.join(HERE, "raw", "downloaded", "indiccorp", "nepali_indiccorp_raw.txt"),
    },
    {
        "name": "news_scrape_manual",
        "type": "manual",
        "granularity": "article",
        "path": os.path.join(HERE, "raw", "manual", "news_scrape", "scraped_articles.txt"),
    },
    {
        "name": "wikipedia_scrape_manual",
        "type": "manual",
        "granularity": "article",
        "path": os.path.join(HERE, "raw", "manual", "wikipedia_scrape", "scraped_wikipedia.txt"),
    },
]

if __name__ == "__main__":
    run(
        language="nepali",
        sources=SOURCES,
        splits_dir=os.path.join(HERE, "splits"),
        processed_dir=os.path.join(HERE, "processed"),
        stats_path=os.path.join(HERE, "..", "statistics", "preprocessing_stats.json"),
    )
