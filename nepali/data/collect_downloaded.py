"""Download the public (IndicCorp v2) portion of the Nepali corpus.

Same 500M-token best-effort target and chars/token assumption as the Hindi
collector (see hindi/data/collect_downloaded.py), applied to IndicCorp's
`npi_Deva` split. Nepali is the lower-resource language in this project, so
if `ne.txt` is exhausted before the target is reached, the shortfall is
recorded in the stats file (target_reached=False) and reported honestly in
the Phase 1 report rather than padded with another source.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from download_indiccorp import stream_indiccorp_to_target  # noqa: E402

TARGET_TOKENS = 500_000_000
CHARS_PER_TOKEN = 4.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

SOURCE_FILES = ["ne.txt"]

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "raw", "downloaded", "indiccorp", "nepali_indiccorp_raw.txt")
STATS_PATH = os.path.join(HERE, "..", "statistics", "download_stats.json")

if __name__ == "__main__":
    stats = stream_indiccorp_to_target(SOURCE_FILES, OUT_PATH, TARGET_CHARS, CHARS_PER_TOKEN)
    stats["language"] = "nepali"
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
