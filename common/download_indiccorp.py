"""Shared utility to stream monolingual text out of the public IndicCorp v2
corpus (ai4bharat/IndicCorpV2 on HuggingFace) without downloading full
multi-gigabyte files when only a prefix is needed.

IndicCorp v2 is CC-0 licensed and served as plain UTF-8 text files (one
sentence/document per line) behind HuggingFace's resolve/CDN redirect, so a
streamed GET with early termination is sufficient -- no `datasets` library
or local Parquet conversion required.
"""

import codecs
import os
import time

import requests

HF_BASE_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/{}"


def stream_indiccorp_to_target(files, out_path, target_chars, chars_per_token=4.5,
                                chunk_size=1 << 20, log_every_chars=100_000_000):
    """Stream one or more IndicCorp language files, writing raw lines to
    `out_path` until `target_chars` characters have been collected (or the
    file list is exhausted).

    Progress is measured directly on raw character counts, following the
    project convention of estimating tokens as characters / chars_per_token
    ahead of tokenizer training. Stops mid-file as soon as the target is
    met, so a large source file does not have to be downloaded in full.

    Returns a stats dict (per-file and totals) that the caller can persist
    as part of the dataset provenance record.
    """
    total_chars = 0
    total_lines = 0
    total_bytes = 0
    last_logged = 0
    stats = {"files": []}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as out:
        for fname in files:
            if total_chars >= target_chars:
                break
            url = HF_BASE_URL.format(fname)
            file_chars = 0
            file_lines = 0
            file_bytes = 0
            buffer = ""
            # Incremental decoder correctly carries a partial multi-byte UTF-8
            # sequence over to the next chunk instead of mangling it at the
            # chunk boundary (Devanagari codepoints are 3 bytes in UTF-8, so a
            # 1MB chunk_size will regularly split one mid-character).
            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            print(f"[{fname}] starting stream from {url}", flush=True)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    buffer += decoder.decode(chunk)
                    *complete_lines, buffer = buffer.split("\n")
                    for line in complete_lines:
                        line = line.strip()
                        if not line:
                            continue
                        out.write(line + "\n")
                        n = len(line)
                        file_chars += n
                        total_chars += n
                        file_lines += 1
                        total_lines += 1
                    if total_chars - last_logged >= log_every_chars:
                        last_logged = total_chars
                        elapsed = time.time() - t0
                        print(f"  progress: {total_chars/1e9:.2f}B chars "
                              f"(~{total_chars/chars_per_token/1e6:.1f}M est. tokens), "
                              f"{total_bytes/1e9:.2f} GB downloaded, {elapsed/60:.1f} min elapsed",
                              flush=True)
                    if total_chars >= target_chars:
                        break
                buffer += decoder.decode(b"", final=True)
                trailing = buffer.strip()
                if trailing and total_chars < target_chars:
                    out.write(trailing + "\n")
                    file_chars += len(trailing)
                    total_chars += len(trailing)
                    file_lines += 1
                    total_lines += 1
            stats["files"].append({
                "file": fname,
                "chars": file_chars,
                "lines": file_lines,
                "bytes_downloaded": file_bytes,
            })
            print(f"[{fname}] done: +{file_chars:,} chars, {file_lines:,} lines, "
                  f"{file_bytes/1e9:.2f} GB downloaded", flush=True)

    stats.update({
        "source": "ai4bharat/IndicCorpV2",
        "license": "CC-0",
        "total_chars": total_chars,
        "total_lines": total_lines,
        "total_bytes_downloaded": total_bytes,
        "estimated_tokens": total_chars / chars_per_token,
        "chars_per_token_assumption": chars_per_token,
        "target_chars": target_chars,
        "target_reached": total_chars >= target_chars,
        "elapsed_seconds": time.time() - t0,
        "out_path": out_path,
    })
    return stats
