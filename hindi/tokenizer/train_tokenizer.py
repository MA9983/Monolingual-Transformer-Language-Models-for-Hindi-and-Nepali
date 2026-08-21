"""Train the Hindi SentencePiece (BPE) tokenizer from scratch on
hindi/data/splits/train.txt, choosing vocabulary size via a fertility/
UNK-rate sweep on hindi/data/splits/val.txt. See
common/train_tokenizer.py for the shared implementation.
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

# Per TA guidance the best vocab size is expected in the 5k-10k range;
# swept across that band plus two larger points (16k, 24k) so the
# selection is empirically justified against a wider comparison, not just
# asserted.
CANDIDATE_VOCAB_SIZES = [5000, 7000, 9000, 10000, 16000, 24000]

SAMPLE_TEXTS = [
    "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "प्रधानमंत्री ने आज नई दिल्ली में एक महत्वपूर्ण बैठक की अध्यक्षता की।",
    "क्रिकेट विश्व कप के फाइनल मैच में भारत ने शानदार जीत दर्ज की।",
]

if __name__ == "__main__":
    report = sweep_and_train(
        language="hindi",
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
