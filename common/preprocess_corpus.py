"""Shared cleaning -> dedup -> train/val/test split pipeline, run once
independently per language (see hindi/data/preprocess.py and
nepali/data/preprocess.py) so the two corpora never share any processing
state or vocabulary.

Streaming / O(1)-memory by design: every record is read, cleaned, and
written out immediately rather than accumulated in a Python list, and the
train/val/test split assignment is a deterministic hash of each record's
own (cleaned) content rather than a materialized shuffle -- this keeps
memory flat regardless of corpus size (targeting hundreds of millions of
tokens here) while still being reproducible and effectively random.
"""

import json
import os

from text_cleaning import (
    Deduplicator,
    contains_bad_word,
    content_hash,
    is_valid_line,
    load_badwords_set,
    normalize_unicode,
)

CHARS_PER_TOKEN = 4.5
SPLIT_RATIOS = {"train": 0.95, "val": 0.025, "test": 0.025}


def iter_records(path, granularity):
    """Yield raw text records from a source file. `granularity="line"`
    treats every non-empty line as one record (IndicCorp's native format).
    `granularity="article"` treats blank-line-separated blocks as one
    record each (how collect_manual_news.py writes scraped articles), so
    a whole article stays together rather than being split mid-document.
    """
    if granularity == "line":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    elif granularity == "article":
        with open(path, encoding="utf-8") as f:
            buf = []
            for line in f:
                if line.strip() == "":
                    if buf:
                        yield "\n".join(buf).strip()
                        buf = []
                else:
                    buf.append(line.rstrip("\n"))
            if buf:
                yield "\n".join(buf).strip()
    else:
        raise ValueError(f"unknown granularity: {granularity}")


def split_bucket(text):
    """Deterministic pseudo-random train/val/test assignment from a hash
    of the record's own content, so re-running preprocessing is
    reproducible and no in-memory shuffle/index over millions of records
    is ever needed.
    """
    h = int(content_hash(text)[:8], 16) % 1000
    cum = 0
    for name, ratio in SPLIT_RATIOS.items():
        cum += ratio * 1000
        if h < cum:
            return name
    return "test"


def clean_record(text, granularity):
    """Normalize and quality-filter one record. For articles, filters at
    the paragraph level first (drops boilerplate/short lines trafilatura
    may have let through) and keeps the article only if enough valid
    content remains; returns None to signal "drop this record".
    """
    text = normalize_unicode(text)
    if granularity == "article":
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        valid_paragraphs = [p for p in paragraphs if is_valid_line(p)]
        if not valid_paragraphs:
            return None
        joined = " ".join(valid_paragraphs)
        if len(joined) < 200:
            return None
        return joined
    else:
        if not is_valid_line(text):
            return None
        return text


def run(language, sources, splits_dir, processed_dir, stats_path, badwords_path=None):
    """Process every source for one language end to end and write:
    - splits_dir/{train,val,test}.txt: one cleaned, deduplicated, safety-
      filtered record per line, ready for tokenizer training / pretraining.
    - stats_path: per-source and aggregate statistics (raw vs. kept
      counts, manual vs. downloaded token split, split sizes) for the
      Phase 1 report.

    `badwords_path`, if given, is a one-term-per-line Devanagari profanity
    list (see common/text_cleaning.py:load_badwords_set); any record
    containing one of its terms as a whole word is dropped, tracked
    separately from the invalid/duplicate counts.
    """
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    badwords = load_badwords_set(badwords_path) if badwords_path else None
    dedup = Deduplicator()
    split_files = {name: open(os.path.join(splits_dir, f"{name}.txt"), "w", encoding="utf-8")
                    for name in SPLIT_RATIOS}
    # Combined, pre-split cleaned+deduplicated corpus (every kept record from
    # every source, in the order processed) -- a transparency/audit artifact
    # showing exactly what survived cleaning before the train/val/test
    # assignment was applied, distinct from (and redundant with, by design)
    # the three split files.
    processed_file = open(os.path.join(processed_dir, "cleaned_corpus.txt"), "w", encoding="utf-8")
    stats = {
        "language": language,
        "chars_per_token_assumption": CHARS_PER_TOKEN,
        "sources": [],
        "splits": {name: {"records": 0, "chars": 0} for name in SPLIT_RATIOS},
    }
    manual_chars = downloaded_chars = 0

    try:
        for src in sources:
            if not os.path.exists(src["path"]):
                print(f"[{language}] [warn] missing source file, skipping: {src['path']}", flush=True)
                continue
            # Optional explicit cap (source["max_kept_chars"]): stop once this
            # source's own kept (post-cleaning) chars reach the limit, rather
            # than consuming the whole file. Used to hold the downloaded
            # source to a fixed size so the corpus lands at a target overall
            # manual fraction without touching the manual side at all.
            max_kept_chars = src.get("max_kept_chars")
            raw_records = kept_records = invalid_records = dup_records = unsafe_records = 0
            kept_chars = raw_chars = invalid_chars = dup_chars = unsafe_chars = 0
            capped = False
            for raw in iter_records(src["path"], src["granularity"]):
                if max_kept_chars is not None and kept_chars >= max_kept_chars:
                    capped = True
                    break
                raw_records += 1
                raw_chars += len(raw)
                cleaned = clean_record(raw, src["granularity"])
                if cleaned is None:
                    invalid_records += 1
                    invalid_chars += len(raw)
                    continue
                if contains_bad_word(cleaned, badwords):
                    unsafe_records += 1
                    unsafe_chars += len(cleaned)
                    continue
                if dedup.is_duplicate(cleaned):
                    dup_records += 1
                    dup_chars += len(cleaned)
                    continue
                bucket = split_bucket(cleaned)
                split_files[bucket].write(cleaned + "\n")
                processed_file.write(cleaned + "\n")
                kept_records += 1
                kept_chars += len(cleaned)
                stats["splits"][bucket]["records"] += 1
                stats["splits"][bucket]["chars"] += len(cleaned)
                if src["type"] == "manual":
                    manual_chars += len(cleaned)
                else:
                    downloaded_chars += len(cleaned)
                if raw_records % 2_000_000 == 0:
                    print(f"[{language}/{src['name']}] ...{raw_records:,} raw records scanned, "
                          f"{kept_records:,} kept so far", flush=True)
            stats["sources"].append({
                "name": src["name"], "type": src["type"], "path": src["path"],
                "raw_records": raw_records, "kept_records": kept_records,
                "invalid_records": invalid_records, "duplicate_records": dup_records,
                "unsafe_records": unsafe_records,
                "raw_chars": raw_chars, "estimated_raw_tokens": raw_chars / CHARS_PER_TOKEN,
                "invalid_chars_removed": invalid_chars, "duplicate_chars_removed": dup_chars,
                "unsafe_chars_removed": unsafe_chars,
                "kept_chars": kept_chars, "estimated_kept_tokens": kept_chars / CHARS_PER_TOKEN,
                "chars_reduction_from_cleaning_dedup": raw_chars - kept_chars,
                "pct_chars_reduction_from_cleaning_dedup": (
                    (raw_chars - kept_chars) / raw_chars if raw_chars else 0.0),
                "capped_at_max_kept_chars": capped,
            })
            cap_note = f" (capped at {max_kept_chars:,} chars)" if capped else ""
            print(f"[{language}/{src['name']}] done: raw={raw_records:,} kept={kept_records:,} "
                  f"invalid={invalid_records:,} dup={dup_records:,} unsafe={unsafe_records:,} "
                  f"kept_chars={kept_chars:,} "
                  f"(~{kept_chars/CHARS_PER_TOKEN/1e6:.1f}M est. tokens){cap_note}", flush=True)
    finally:
        for f in split_files.values():
            f.close()
        processed_file.close()

    total_chars = manual_chars + downloaded_chars
    stats.update({
        "total_kept_chars": total_chars,
        "estimated_total_tokens": total_chars / CHARS_PER_TOKEN,
        "manual_chars": manual_chars,
        "downloaded_chars": downloaded_chars,
        "manual_fraction": (manual_chars / total_chars) if total_chars else 0.0,
        "manual_fraction_meets_20pct_requirement": (manual_chars / total_chars >= 0.20) if total_chars else False,
        "unique_records_seen": len(dedup),
    })
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats
