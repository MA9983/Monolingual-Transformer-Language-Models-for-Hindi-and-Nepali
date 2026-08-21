"""Train the Nepali SentencePiece (BPE) tokenizer from scratch on
nepali/data/splits/train.txt, choosing vocabulary size via a fertility/
UNK-rate sweep on nepali/data/splits/val.txt. See
common/train_tokenizer.py for the shared implementation.

Completely independent of hindi/tokenizer/train_tokenizer.py: separate
input files, separate SentencePieceTrainer call, separate output
model/vocab -- no state or vocabulary is shared between the two languages.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from train_tokenizer import sweep_and_train  # noqa: E402

HERE = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(HERE, "..", "data", "splits", "train.txt")
VAL_PATH = os.path.join(HERE, "..", "data", "splits", "val.txt")
STATS_PATH = os.path.join(HERE, "..", "statistics", "tokenizer_stats.json")

# swept across that band plus two larger points (16k, 24k) so the
# selection is empirically justified against a wider comparison, not just
# asserted. Same range as Hindi (final corpus sizes ended up comparable,
# ~500M vs ~507M tokens post-cleaning).
CANDIDATE_VOCAB_SIZES = [5000, 7000, 9000, 10000, 16000, 24000]

SAMPLE_TEXTS = [
    "नेपाल हिमालयको काखमा रहेको सुन्दर देश हो।",
    "प्रधानमन्त्रीले आज काठमाडौंमा एक महत्वपूर्ण बैठकको अध्यक्षता गर्नुभयो।",
    "क्रिकेट विश्वकपको फाइनल खेलमा नेपालले उत्कृष्ट प्रदर्शन गर्यो।",
]

if __name__ == "__main__":
    report = sweep_and_train(
        language="nepali",
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        out_dir=HERE,
        candidate_vocab_sizes=CANDIDATE_VOCAB_SIZES,
        sample_texts=SAMPLE_TEXTS,
    )
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
