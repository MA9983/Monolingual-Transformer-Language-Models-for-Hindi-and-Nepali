"""Shared cleaning/quality-filtering utilities applied identically (but
independently -- no data is ever mixed) to both languages' corpora.

Kept deliberately simple and script-based (as opposed to a heavier
ML-based quality classifier) since the input here is either an
already-curated public corpus (IndicCorp) or extracted main-article text
(trafilatura already stripped nav/boilerplate/ads), so the main remaining
risks are: encoding inconsistency, off-script/mixed-language noise,
near-empty lines, and exact/near duplicates.
"""

import hashlib
import re
import unicodedata
from collections import Counter

from datasketch import MinHash, MinHashLSH

DEVANAGARI_START, DEVANAGARI_END = 0x0900, 0x097F


def normalize_unicode(text):
    """Canonicalize to NFC so visually-identical Devanagari sequences
    (e.g. a vowel sign expressed as separate combining codepoints vs. a
    precomposed one) collapse to a single representation before counting,
    deduplicating, or tokenizing them.
    """
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def devanagari_letter_ratio(text):
    """Fraction of alphabetic characters that fall in the Devanagari
    Unicode block, used as the primary script/language purity signal.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for ch in letters if DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END)
    return dev / len(letters)


def is_valid_line(text, min_chars=20, min_devanagari_ratio=0.6, min_alpha_ratio=0.4):
    """Reject lines that are too short, not predominantly Devanagari script,
    or mostly punctuation/digits/markup rather than prose (e.g. leftover
    navigation labels, timestamps, ad slugs).
    """
    text = text.strip()
    if len(text) < min_chars:
        return False
    if devanagari_letter_ratio(text) < min_devanagari_ratio:
        return False
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha / len(text) < min_alpha_ratio:
        return False
    return True


def content_hash(text):
    """Hash of whitespace-normalized, lowercased text for exact/near-exact
    duplicate detection (near-exact in the sense that pure whitespace or
    casing differences -- irrelevant for Devanagari but harmless in
    general -- do not count as distinct records).
    """
    norm = normalize_whitespace(text).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


DEVANAGARI_WORD_RE = re.compile(r"[ऀ-ॿ]+")


def load_badwords_set(path):
    """Read a one-term-per-line badwords list (# comments / blank lines
    ignored) into a set for exact whole-word lookup.

    Deliberately NOT a substring/regex-\\b match: Python's \\b treats
    Devanagari combining vowel signs (matra) as non-word characters, so a
    naive \\b-anchored regex would false-match a listed root (e.g. the
    listed word "मुठ्ठ") as a "whole word" inside a completely unrelated,
    benign word that merely starts with the same consonant sequence before
    its vowel sign (e.g. "मुठ्ठी", meaning "fist") -- verified directly
    during development. Extracting whole Devanagari-codepoint runs as
    tokens and comparing by exact set membership avoids that false-positive
    class entirely, at the cost of needing common inflected spellings
    listed explicitly (already done in badwords_hindi.txt/badwords_nepali.txt).
    """
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line)
    return terms


def contains_bad_word(text, badwords):
    if not badwords:
        return False
    return any(w in badwords for w in DEVANAGARI_WORD_RE.findall(text))


def is_repetitive_text(text, ngram_size=4, max_top_ngram_ratio=0.3):
    """Flag documents dominated by one repeated phrase (e.g. scraped
    boilerplate/spam that survived extraction) using the standard
    "top n-gram repetition ratio" heuristic from the Gopher/MassiveText
    quality filters (Rae et al. 2021): among all overlapping word n-grams
    in the text, if the single most common one accounts for more than
    `max_top_ngram_ratio` of all n-gram occurrences, the text is treated
    as repetitive junk rather than genuine prose. Pure frequency counting,
    no external model.
    """
    words = text.split()
    total_ngrams = len(words) - ngram_size + 1
    if total_ngrams < 4:
        return False
    counts = Counter(tuple(words[i:i + ngram_size]) for i in range(total_ngrams))
    top_count = counts.most_common(1)[0][1]
    return (top_count / total_ngrams) > max_top_ngram_ratio


class NearDuplicateDetector:
    """MinHash/LSH-based fuzzy near-duplicate detection (Jaccard similarity
    over overlapping word shingles), catching near-identical documents that
    differ only in minor edits -- e.g. syndicated wire stories or an
    article republished with a changed byline/date -- which exact
    content-hash matching (Deduplicator) cannot catch since it requires an
    identical hash of the whole normalized text.

    Applied selectively (not to every source): only wired into
    manually-scraped article sources (tens/hundreds of thousands of
    records, so a ~1KB-per-record signature at num_perm=128 stays well
    under a few hundred MB total) rather than the multi-million-line
    downloaded source, where the memory cost of one signature per record
    would be prohibitive and where upstream curation already makes
    near-duplicates rare. num_perm=128 (datasketch's own default) matters
    for accuracy, not just precision: at a low num_perm (e.g. 32, tried
    first during development) MinHash's similarity *estimate* has enough
    variance that a genuine near-duplicate with true Jaccard 0.94 was
    estimated at 0.84 -- just under a 0.85 threshold -- and missed;
    num_perm=128 estimated the same pair at 0.93, correctly above it.
    """

    def __init__(self, num_perm=128, threshold=0.85, shingle_size=3):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._next_key = 0

    def _minhash(self, text):
        m = MinHash(num_perm=self.num_perm)
        words = text.split()
        n = self.shingle_size
        shingles = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
        for shingle in shingles:
            m.update(shingle.encode("utf-8"))
        return m

    def is_near_duplicate(self, text):
        m = self._minhash(text)
        if self.lsh.query(m):
            return True
        self.lsh.insert(f"r{self._next_key}", m)
        self._next_key += 1
        return False


class Deduplicator:
    """Tracks content hashes across an arbitrarily large stream without
    keeping the original text in memory twice.
    """

    def __init__(self):
        self.seen = set()

    def is_duplicate(self, text):
        h = content_hash(text)
        if h in self.seen:
            return True
        self.seen.add(h)
        return False

    def __len__(self):
        return len(self.seen)
