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

Target: set high (per the user's ~110-120M-token manual-data goal for
both languages) but almost certainly unreachable from this source alone
-- ne.wikipedia.org has on the order of 40,000 articles total, so even
crawling every single one of them (impossible in practice too, since many
are short stubs and some are unreachable from these seeds) caps out
somewhere around 15-20M tokens, not 110-120M. Setting the target this
high is still the right move mechanically: it just means the crawler
keeps going until it has exhausted essentially the whole reachable wiki
rather than stopping early at an artificially low number, which is
exactly what "get as much manual Nepali data as this source can give"
means in practice. Nepali news scraping (collect_manual_news.py) adds
more on top, but nowhere near enough to close a ~100M-token gap -- the
combined realistic ceiling for Nepali manual data is far below Hindi's,
and that shortfall should be reported honestly rather than papered over.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from wikipedia_scraper import WikipediaCollector  # noqa: E402

TARGET_TOKENS = 115_000_000
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
    # Fourth batch: the frontier was running low after ~3 hours of
    # crawling (down to ~2,300 pending titles), so this adds fresh,
    # topically distant entry points -- biography-heavy and
    # district/administrative clusters tend to be under-explored by the
    # earlier subject-area seeds above.
    "जनजाति", "नेपालका जिल्ला", "प्रदेश नं. १", "मधेश प्रदेश", "सुदूरपश्चिम प्रदेश",
    "कर्णाली प्रदेश", "बागमती प्रदेश", "नेपाली उपन्यास", "नेपाल भाषा", "मैथिली भाषा",
    "भोजपुरी भाषा", "तामाङ जाति", "गुरुङ जाति", "मगर जाति", "थारू जाति", "शेर्पा",
    "नेवार जाति", "राई जाति", "लिम्बू जाति", "दलित", "जात व्यवस्था",
    "नेपालको संविधान", "स्थानीय तह", "प्रतिनिधि सभा", "राष्ट्रपति", "प्रधानमन्त्री",
    "नेपाल प्रहरी", "नेपाली सेना", "अन्तर्राष्ट्रिय सम्बन्ध", "संयुक्त राष्ट्रसंघ",
    "दक्षिण एशियाली क्षेत्रीय सहयोग संगठन", "बहुराष्ट्रिय कम्पनी", "बैंकिङ",
    "नेपाल राष्ट्र बैंक", "शेयर बजार", "उद्योग", "पर्यावरण संरक्षण", "वन्यजन्तु",
    "राष्ट्रिय निकुञ्ज", "चितवन राष्ट्रिय निकुञ्ज", "अन्नपूर्ण संरक्षण क्षेत्र",
    "औषधि विज्ञान", "मनोविज्ञान", "समाजशास्त्र", "मानवशास्त्र", "पुरातत्व",
    "भाषाविज्ञान", "दर्शनशास्त्र", "पत्रकारिता", "प्रकाशन", "पुस्तकालय",
    "नेपाली चलचित्र उद्योग", "ललितकला", "वास्तुकला", "फोटोग्राफी", "नृत्य",
    # Fifth batch: the frontier hit 0 and the crawl stopped naturally
    # (~24,400 articles visited) -- this is expected, matching the
    # ~15-20M-token ceiling flagged earlier for a wiki this size. Named
    # people, specific districts, and specific historical
    # events/religious sites tend to be genuinely under-linked from the
    # broad subject-area seeds above, so they're the best remaining bet
    # for finding any still-unvisited pockets of the site.
    "शेरबहादुर देउवा", "केपी शर्मा ओली", "पुष्पकमल दाहाल", "गिरिजाप्रसाद कोइराला",
    "बीपी कोइराला", "ज्ञानेन्द्र शाह", "वीरेन्द्र वीर विक्रम शाह", "माधवकुमार नेपाल",
    "लक्ष्मीप्रसाद देवकोटा", "पारिजात", "भूपी शेरचन", "बालकृष्ण सम",
    "पारस खड्का", "संदीप लामिछाने", "नेपाल क्रिकेट राष्ट्रिय टोली",
    "नेपाल राष्ट्रिय फुटबल टोली", "काठमाडौं जिल्ला", "चितवन जिल्ला",
    "रुपन्देही जिल्ला", "मोरङ जिल्ला", "झापा जिल्ला", "कास्की जिल्ला",
    "स्याङ्जा जिल्ला", "गोरखा जिल्ला", "धादिङ जिल्ला", "नुवाकोट जिल्ला",
    "सिन्धुपाल्चोक जिल्ला", "दोलखा जिल्ला", "सोलुखुम्बु जिल्ला", "मुस्ताङ जिल्ला",
    "नेपाल एकीकरण", "राणा शासन", "२०६२।०६३ को जनआन्दोलन", "माओवादी जनयुद्ध",
    "२०७२ नेपाल भूकम्प", "पशुपतिनाथ मन्दिर", "स्वयम्भूनाथ", "बौद्धनाथ", "लुम्बिनी",
    "जनकपुरधाम", "मुक्तिनाथ", "हलेसी महादेव", "पर्यटन उद्योग", "हस्तकला",
    "नेपाल विद्युत प्राधिकरण", "नेपाल टेलिकम", "नेपाल वायुसेवा निगम",
    "त्रिभुवन विश्वविद्यालय", "काठमाडौं विश्वविद्यालय", "पोखरा विश्वविद्यालय",
    # Sixth batch: category pages (श्रेणी: = Category:), added after
    # confirming plain link-following BFS had genuinely plateaued (a
    # 44-title reseed found almost nothing new). Category pages list
    # member articles and subcategories explicitly rather than relying on
    # inline links, and common/wikipedia_scraper.py now recurses into
    # subcategories found this way (while never saving a category page's
    # own listing as "content") -- this should reach corners of the wiki
    # the pure link-graph crawl missed entirely.
    "श्रेणी:नेपाल", "श्रेणी:इतिहास", "श्रेणी:विज्ञान", "श्रेणी:भूगोल",
    "श्रेणी:व्यक्ति", "श्रेणी:जीवनी", "श्रेणी:संस्कृति", "श्रेणी:खेलकुद",
    "श्रेणी:राजनीति", "श्रेणी:धर्म", "श्रेणी:भाषा", "श्रेणी:साहित्य",
    "श्रेणी:शिक्षा", "श्रेणी:प्रविधि", "श्रेणी:अर्थतन्त्र",
    "श्रेणी:नेपालका जिल्ला", "श्रेणी:नेपालका सहरहरू", "श्रेणी:नेपालका नदीहरू",
    "श्रेणी:नेपालका पहाडहरू", "श्रेणी:नेपाली राजनीतिज्ञ", "श्रेणी:नेपाली लेखक",
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
