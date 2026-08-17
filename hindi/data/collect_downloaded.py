"""Download the public (IndicCorp v2) portion of the Hindi corpus.

Target: ~500M raw tokens, estimated as characters / 4.5 (project-wide
assumption ahead of tokenizer training -- Devanagari BPE fertility is
typically in the 4-5 chars/token range). Streams only as much of the
IndicCorp `hin_Deva` split as needed to hit that target, so the full
~27GB source file does not have to be downloaded.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from download_indiccorp import stream_indiccorp_to_target  # noqa: E402

TARGET_TOKENS = 500_000_000
CHARS_PER_TOKEN = 4.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

# hi-1/2/3 are shards of the same hin_Deva split; fall back to later shards
# only if the first is exhausted before the target is reached.
SOURCE_FILES = ["hi-1.txt", "hi-2.txt", "hi-3.txt"]

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "raw", "downloaded", "indiccorp", "hindi_indiccorp_raw.txt")
STATS_PATH = os.path.join(HERE, "..", "statistics", "download_stats.json")

if __name__ == "__main__":
    stats = stream_indiccorp_to_target(SOURCE_FILES, OUT_PATH, TARGET_CHARS, CHARS_PER_TOKEN)
    stats["language"] = "hindi"
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
