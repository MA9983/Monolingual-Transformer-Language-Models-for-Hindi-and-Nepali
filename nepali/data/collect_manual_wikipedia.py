"""Manual-collection supplement for the Nepali corpus: a BFS link-following
crawl of ne.wikipedia.org (see common/wikipedia_scraper.py for why this
counts as manual collection and how it stays robots.txt-compliant).

This exists because news-site scraping alone (collect_manual_news.py)
exhausts at ~92K tokens across all 7 vetted Nepali news sites -- their
entire discoverable pool via sitemaps/RSS, not a "needs more time" problem
-- which is nowhere near enough to keep manual data at >=20% of a corpus
anywhere near the ~500M-token target. Wikipedia was chosen, after
discussion, as the way to close that gap: it is explicitly one of the
assignment's own examples of manual collection ("scraping + cleaning
pages you gather yourself"), scales far beyond what the news sites can
offer (Nepali Wikipedia has on the order of 40,000 articles), and is
naturally already in Devanagari, real Nepali prose.

Target: best-effort, sized to what a background crawl can realistically
reach given ~1 request/2s from a single domain (no round-robin speedup
possible, unlike the multi-site news crawler) -- left running well past
any single check-in, same as the other Phase 1 background collection
jobs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from wikipedia_scraper import WikipediaCollector  # noqa: E402

TARGET_TOKENS = 40_000_000
CHARS_PER_TOKEN = 4.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

# Deliberately topically diverse seeds so the BFS frontier doesn't get
# stuck in one densely-interlinked cluster (e.g. all geography) before
# exploring others (science, sport, religion, ...). Also serves as the
# recovery seed list if frontier.txt is ever lost (e.g. the crawl was
# interrupted before the frontier-persistence fix landed) and all of an
# earlier, smaller seed set has already been visited -- a broader list
# here means recovery doesn't stall with an empty frontier.
SEED_TITLES = [
    "नेपाल", "भारत", "काठमाडौं", "विज्ञान", "इतिहास", "भूगोल", "राजनीति",
    "साहित्य", "नेपाली भाषा", "धर्म", "खेलकुद", "अर्थशास्त्र", "स्वास्थ्य",
    "शिक्षा", "प्रविधि", "संस्कृति", "गणित", "जीवविज्ञान", "भौतिक शास्त्र", "कला",
    "पोखरा", "ललितपुर", "भक्तपुर", "विराटनगर", "धरान", "बुटवल", "जनकपुर",
    "हिमालय", "सगरमाथा", "गण्डकी प्रदेश", "कोशी प्रदेश", "लुम्बिनी प्रदेश",
    "बौद्ध धर्म", "हिन्दू धर्म", "इस्लाम", "इसाई धर्म", "किराँत धर्म",
    "नेपाल संघीय संसद", "नेपाल कम्युनिस्ट पार्टी", "नेपाली कांग्रेस",
    "पृथ्वीनारायण शाह", "गौतम बुद्ध", "भानुभक्त आचार्य", "मदन भण्डारी",
    "क्रिकेट", "फुटबल", "ओलम्पिक खेल", "एशियाली खेलकुद",
    "रसायन शास्त्र", "जीवविज्ञान", "खगोल विज्ञान", "कम्प्युटर विज्ञान",
    "अर्थतन्त्र", "कृषि", "पर्यटन", "जलविद्युत", "यातायात",
    "संगीत", "चलचित्र", "नेपाली संगीत", "लोक संस्कृति",
    "महाभारत", "रामायण", "वेद", "उपनिषद्",
    "विश्व इतिहास", "द्वितीय विश्वयुद्ध", "प्रथम विश्वयुद्ध", "शीत युद्ध",
    "संयुक्त राज्य अमेरिका", "चीन", "जापान", "बेलायत", "फ्रान्स", "जर्मनी",
    "पृथ्वी", "सौर्यमण्डल", "ब्रह्माण्ड", "वातावरण", "जलवायु परिवर्तन",
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw", "manual", "wikipedia_scrape")

if __name__ == "__main__":
    collector = WikipediaCollector(
        wiki_domain="ne.wikipedia.org",
        lang_code="ne",
        seed_titles=SEED_TITLES,
        target_chars=TARGET_CHARS,
        out_dir=OUT_DIR,
        chars_per_token=CHARS_PER_TOKEN,
    )
    collector.collect()
