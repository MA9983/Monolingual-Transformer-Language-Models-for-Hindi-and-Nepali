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
