"""BFS link-following crawler for one Wikipedia language edition, used to
substantially grow the manual-collection pool for a lower-resource
language beyond what news-site scraping alone can supply (see
nepali/data/collect_manual_wikipedia.py).

Classification note (manual, not downloaded): this fetches individual
rendered article pages one at a time and extracts their text, exactly the
"scraping + cleaning pages you gather yourself" method the assignment's
own examples list as manual collection -- it is deliberately NOT a
download of the pre-packaged Wikipedia XML/HTML dumps (dumps.wikimedia.org),
which would just be another bulk "downloaded" corpus like IndicCorp.

robots.txt note: ne.wikipedia.org's robots.txt disallows /w/ (the action
API, except a narrow mobileview exception) and /api/ (the REST API,
except its own doc page) for the default user-agent group, and its
declared sitemap endpoint (/w/rest.php/site/v1/sitemap) 403s for
non-search-engine bots regardless of User-Agent. Regular /wiki/<Title>
article pages are NOT disallowed (only /wiki/Special:* and various
per-wiki admin/deletion-log page patterns are), so this crawler discovers
new titles the only robots.txt-compliant way available: following the
normal mainspace wikilinks found in each fetched article's HTML, the same
way a human reader would browse from page to page.
"""

import json
import os
import re
import time
import urllib.parse

import requests
import trafilatura
from langdetect import DetectorFactory, detect

from news_scraper import USER_AGENT, _content_hash
from robots_utils import RobotsCache

DetectorFactory.seed = 0

REQUEST_TIMEOUT = 20
MIN_ARTICLE_CHARS = 300
# Body/prose content (Parsoid-rendered) links with an absolute URL,
# e.g. href="https://ne.wikipedia.org/wiki/Title" -- distinct from the
# relative /wiki/Title hrefs used by navigation chrome (sidebar, header).
# Matching both is what actually finds enough outbound links per article
# for the BFS frontier to grow; matching only the relative form (as an
# earlier version of this function did) starves the frontier almost
# immediately since real body links are essentially all absolute here.
REDLINK_MARKERS = ('redlink=1', 'class="new"', "class='new'")


def _extract_mainspace_links(html, base_wiki_prefix):
    """Return de-duplicated mainspace article titles (no namespace prefix,
    which MediaWiki always separates with ':' regardless of language --
    e.g. Nepali's "Special:" is "विशेष:") linked from this page's HTML.
    Skips "redlinks" (links to articles that don't exist yet -- fetching
    one just returns an edit-this-page prompt, not article content).
    """
    titles = []
    seen = set()
    for match in re.finditer(r'<a\b[^>]*href="([^"]*)"[^>]*>', html):
        tag, href = match.group(0), match.group(1)
        if href.startswith("/wiki/"):
            path = href
        elif href.startswith(base_wiki_prefix + "/wiki/"):
            # Same-domain absolute link (how Parsoid renders body-content
            # wikilinks); strip the domain to get a plain /wiki/... path.
            path = href[len(base_wiki_prefix):]
        else:
            # Anything else, including interwiki sitelinks to other
            # language editions (e.g. https://de.wikipedia.org/wiki/...),
            # which use the identical /wiki/Title path shape on a
            # different domain and must not be treated as same-site.
            continue
        if any(marker in tag for marker in REDLINK_MARKERS):
            continue
        title_enc = path[len("/wiki/"):].split("?", 1)[0].split("#", 1)[0]
        title = urllib.parse.unquote(title_enc).replace("_", " ")
        if not title or ":" in title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


class WikipediaCollector:
    """BFS crawl of one Wikipedia language edition. Mirrors
    NewsCollector's checkpointed output format (scraped_*.txt,
    *_meta.jsonl, seen_urls.txt, progress.json) so it plugs into
    preprocess.py as just another "manual" granularity="article" source.
    """

    def __init__(self, wiki_domain, lang_code, seed_titles, target_chars, out_dir, chars_per_token=4.5):
        self.base = f"https://{wiki_domain}"
        self.lang_code = lang_code
        self.target_chars = target_chars
        self.chars_per_token = chars_per_token
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.text_path = os.path.join(out_dir, "scraped_wikipedia.txt")
        self.meta_path = os.path.join(out_dir, "scraped_wikipedia_meta.jsonl")
        self.seen_titles_path = os.path.join(out_dir, "seen_titles.txt")
        self.progress_path = os.path.join(out_dir, "progress.json")
        self.robots = RobotsCache(USER_AGENT)

        self.seen_titles = set()
        if os.path.exists(self.seen_titles_path):
            with open(self.seen_titles_path, encoding="utf-8") as f:
                self.seen_titles = set(line.rstrip("\n") for line in f if line.strip())

        self.seen_hashes = set()
        self.total_chars = 0
        self.total_articles = 0
        if os.path.exists(self.meta_path):
            with open(self.meta_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    self.seen_hashes.add(rec["hash"])
                    self.total_chars += rec["n_chars"]
                    self.total_articles += 1

        self.frontier = [t for t in seed_titles if t not in self.seen_titles]

    def _save_progress(self, extra=None):
        prog = {
            "total_chars": self.total_chars,
            "total_articles": self.total_articles,
            "estimated_tokens": self.total_chars / self.chars_per_token,
            "target_chars": self.target_chars,
            "frontier_remaining": len(self.frontier),
        }
        if extra:
            prog.update(extra)
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(prog, f, ensure_ascii=False, indent=2)

    def _article_url(self, title):
        return self.base + "/wiki/" + urllib.parse.quote(title.replace(" ", "_"))

    def collect(self, max_articles=None):
        t0 = time.time()
        headers = {"User-Agent": USER_AGENT}
        print(f"[wikipedia/{self.lang_code}] resume state: {self.total_chars:,} chars "
              f"(~{self.total_chars/self.chars_per_token/1e6:.2f}M est. tokens), "
              f"{self.total_articles:,} articles, {len(self.seen_titles):,} titles already visited, "
              f"{len(self.frontier):,} seeds queued", flush=True)

        text_f = open(self.text_path, "a", encoding="utf-8")
        meta_f = open(self.meta_path, "a", encoding="utf-8")
        seen_f = open(self.seen_titles_path, "a", encoding="utf-8")
        last_log = self.total_chars
        next_allowed = 0.0

        try:
            while self.frontier and self.total_chars < self.target_chars:
                if max_articles and self.total_articles >= max_articles:
                    break
                title = self.frontier.pop(0)
                if title in self.seen_titles:
                    continue
                self.seen_titles.add(title)
                seen_f.write(title + "\n")

                url = self._article_url(title)
                delay = self.robots.crawl_delay(url)
                wait = next_allowed - time.time()
                if wait > 0:
                    time.sleep(wait)
                next_allowed = time.time() + delay

                if not self.robots.can_fetch(url):
                    continue
                try:
                    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        continue
                except Exception:
                    continue

                new_titles = _extract_mainspace_links(r.text, self.base)
                for nt in new_titles:
                    if nt not in self.seen_titles:
                        self.frontier.append(nt)

                text = trafilatura.extract(r.content, favor_recall=False,
                                            include_comments=False, include_tables=False)
                if not text or len(text) < MIN_ARTICLE_CHARS:
                    continue
                h = _content_hash(text)
                if h in self.seen_hashes:
                    continue
                try:
                    detected = detect(text[:1000])
                except Exception:
                    continue
                if detected != self.lang_code:
                    continue

                self.seen_hashes.add(h)
                text_f.write(text.strip() + "\n\n")
                meta_f.write(json.dumps({
                    "title": title, "url": url, "hash": h, "n_chars": len(text),
                }, ensure_ascii=False) + "\n")
                text_f.flush()
                meta_f.flush()
                seen_f.flush()

                self.total_chars += len(text)
                self.total_articles += 1

                if self.total_chars - last_log >= 500_000:
                    last_log = self.total_chars
                    elapsed = (time.time() - t0) / 60
                    print(f"[wikipedia/{self.lang_code}] progress: {self.total_chars/1e6:.2f}M chars "
                          f"(~{self.total_chars/self.chars_per_token/1e6:.2f}M est. tokens), "
                          f"{self.total_articles:,} articles, frontier={len(self.frontier):,}, "
                          f"{elapsed:.1f} min elapsed", flush=True)
                    self._save_progress()
        finally:
            text_f.close()
            meta_f.close()
            seen_f.close()
            self._save_progress({"finished": True, "elapsed_seconds": time.time() - t0})
            print(f"[wikipedia/{self.lang_code}] stopped: {self.total_chars:,} chars, "
                  f"{self.total_articles:,} articles, "
                  f"target_reached={self.total_chars >= self.target_chars}", flush=True)
