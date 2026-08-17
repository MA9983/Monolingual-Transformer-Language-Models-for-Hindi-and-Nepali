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
import queue
import re
import threading
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
# There is only one domain here (unlike the news crawler's several sites),
# so this is the only lever for aggregate throughput: each worker paces
# itself independently at the domain's own crawl-delay, same model as
# NewsCollector.WORKERS_PER_SITE. Wikipedia's own robots.txt explicitly
# welcomes "friendly, low-speed bots" on article pages, and its
# infrastructure is built for vastly higher load than this, so a handful
# of concurrent well-paced connections stays well inside that spirit.
NUM_WORKERS = 6
# Body/prose content (Parsoid-rendered) links with an absolute URL,
# e.g. href="https://ne.wikipedia.org/wiki/Title" -- distinct from the
# relative /wiki/Title hrefs used by navigation chrome (sidebar, header).
# Matching both is what actually finds enough outbound links per article
# for the BFS frontier to grow; matching only the relative form (as an
# earlier version of this function did) starves the frontier almost
# immediately since real body links are essentially all absolute here.
REDLINK_MARKERS = ('redlink=1', 'class="new"', "class='new'")
# Nepali Wikipedia's localized name for the Category: namespace. Plain
# link-following BFS plateaus once it has covered the densely
# cross-linked "core" of the wiki (confirmed: reseeding with 44 fresh
# topic titles found almost nothing new) because category *membership*
# is often not reflected in inline article-to-article links -- a category
# page lists its member articles and subcategories explicitly, reaching
# corners of the site pure link-following misses. Category pages are
# themselves mainspace-adjacent (/wiki/श्रेणी:Title, not /wiki/Special:...),
# so robots.txt does not disallow fetching them.
CATEGORY_PREFIX = "श्रेणी:"


def _extract_mainspace_links(html, base_wiki_prefix, include_categories=False):
    """Return de-duplicated article titles linked from this page's HTML:
    mainspace titles (no namespace prefix, which MediaWiki always
    separates with ':' regardless of language -- e.g. Nepali's "Special:"
    is "विशेष:"), plus, when include_categories=True, CATEGORY_PREFIX
    titles too (so the BFS can recurse into subcategories). Skips
    "redlinks" (links to articles that don't exist yet -- fetching one
    just returns an edit-this-page prompt, not article content).
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
        if not title or title in seen:
            continue
        is_category = title.startswith(CATEGORY_PREFIX)
        if ":" in title and not (include_categories and is_category):
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
        # Discovered-but-not-yet-visited titles, persisted as we find them.
        # Without this, killing the process (OOM, Ctrl-C, a Colab
        # disconnect) silently throws away the entire in-memory BFS
        # frontier -- on resume there would be nothing left to crawl
        # beyond whatever seed titles hadn't been visited yet, even with
        # tens of thousands of discovered titles sitting unexplored.
        self.frontier_path = os.path.join(out_dir, "frontier.txt")
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

        # frontier_queue is thread-safe (queue.Queue); `queued` tracks
        # membership so the same title is never enqueued twice by two
        # workers that discover it on different pages at the same time --
        # both need self.lock since NUM_WORKERS threads touch them.
        self.frontier_queue = queue.Queue()
        self.queued = set()
        pending = []
        if os.path.exists(self.frontier_path):
            with open(self.frontier_path, encoding="utf-8") as f:
                for line in f:
                    t = line.rstrip("\n")
                    if t and t not in self.seen_titles and t not in self.queued:
                        self.queued.add(t)
                        pending.append(t)
        for t in seed_titles:
            if t not in self.seen_titles and t not in self.queued:
                self.queued.add(t)
                pending.append(t)
        for t in pending:
            self.frontier_queue.put(t)
        self._initial_frontier_size = len(pending)
        self.lock = threading.Lock()

    def _save_progress(self, extra=None):
        prog = {
            "total_chars": self.total_chars,
            "total_articles": self.total_articles,
            "estimated_tokens": self.total_chars / self.chars_per_token,
            "target_chars": self.target_chars,
            "frontier_remaining": self.frontier_queue.qsize(),
        }
        if extra:
            prog.update(extra)
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(prog, f, ensure_ascii=False, indent=2)

    def _article_url(self, title):
        return self.base + "/wiki/" + urllib.parse.quote(title.replace(" ", "_"))

    def _worker(self, stop_event, busy, max_articles, t0):
        """One of NUM_WORKERS threads sharing the single Wikipedia domain.
        `busy` is a 1-element list used as a shared mutable counter (under
        self.lock) of workers currently mid-fetch, so an idle worker can
        tell "queue is empty and nothing else is about to add to it"
        (safe to exit) apart from "queue is empty but another worker
        might discover more links any moment" (should keep waiting).
        """
        headers = {"User-Agent": USER_AGENT}
        next_allowed = 0.0
        while not stop_event.is_set():
            try:
                title = self.frontier_queue.get(timeout=1.0)
            except queue.Empty:
                with self.lock:
                    if busy[0] == 0:
                        return
                continue

            with self.lock:
                self.queued.discard(title)
                already_seen = title in self.seen_titles
                if not already_seen:
                    self.seen_titles.add(title)
                    self.seen_f.write(title + "\n")
                    self.seen_f.flush()
                busy[0] += 1
            if already_seen:
                with self.lock:
                    busy[0] -= 1
                continue

            try:
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

                new_titles = _extract_mainspace_links(r.text, self.base, include_categories=True)
                with self.lock:
                    for nt in new_titles:
                        if nt not in self.seen_titles and nt not in self.queued:
                            self.queued.add(nt)
                            self.frontier_queue.put(nt)
                            self.frontier_f.write(nt + "\n")
                    self.frontier_f.flush()

                if title.startswith(CATEGORY_PREFIX):
                    # Category pages are a link index (member articles +
                    # subcategories), not prose -- already queued above,
                    # never saved as "content".
                    continue

                text = trafilatura.extract(r.content, favor_recall=False,
                                            include_comments=False, include_tables=False)
                if not text or len(text) < MIN_ARTICLE_CHARS:
                    continue
                h = _content_hash(text)
                if h in self.seen_hashes:  # cheap unlocked pre-check
                    continue
                try:
                    detected = detect(text[:1000])
                except Exception:
                    continue
                if detected != self.lang_code:
                    continue

                with self.lock:
                    if h in self.seen_hashes:
                        continue
                    self.seen_hashes.add(h)
                    self.text_f.write(text.strip() + "\n\n")
                    self.meta_f.write(json.dumps({
                        "title": title, "url": url, "hash": h, "n_chars": len(text),
                    }, ensure_ascii=False) + "\n")
                    self.text_f.flush()
                    self.meta_f.flush()

                    self.total_chars += len(text)
                    self.total_articles += 1

                    if self.total_chars - self.last_log >= 500_000:
                        self.last_log = self.total_chars
                        elapsed = (time.time() - t0) / 60
                        print(f"[wikipedia/{self.lang_code}] progress: {self.total_chars/1e6:.2f}M chars "
                              f"(~{self.total_chars/self.chars_per_token/1e6:.2f}M est. tokens), "
                              f"{self.total_articles:,} articles, "
                              f"frontier={self.frontier_queue.qsize():,}, "
                              f"{elapsed:.1f} min elapsed", flush=True)
                        self._save_progress()

                    if (self.total_chars >= self.target_chars
                            or (max_articles and self.total_articles >= max_articles)):
                        stop_event.set()
            finally:
                with self.lock:
                    busy[0] -= 1

    def collect(self, max_articles=None):
        t0 = time.time()
        print(f"[wikipedia/{self.lang_code}] resume state: {self.total_chars:,} chars "
              f"(~{self.total_chars/self.chars_per_token/1e6:.2f}M est. tokens), "
              f"{self.total_articles:,} articles, {len(self.seen_titles):,} titles already visited, "
              f"{self._initial_frontier_size:,} seeds queued, {NUM_WORKERS} workers", flush=True)

        self.text_f = open(self.text_path, "a", encoding="utf-8")
        self.meta_f = open(self.meta_path, "a", encoding="utf-8")
        self.seen_f = open(self.seen_titles_path, "a", encoding="utf-8")
        self.frontier_f = open(self.frontier_path, "a", encoding="utf-8")
        self.last_log = self.total_chars
        stop_event = threading.Event()
        busy = [0]

        try:
            threads = [threading.Thread(target=self._worker, args=(stop_event, busy, max_articles, t0),
                                         daemon=True)
                       for _ in range(NUM_WORKERS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            self.text_f.close()
            self.meta_f.close()
            self.seen_f.close()
            self.frontier_f.close()
            self._save_progress({"finished": True, "elapsed_seconds": time.time() - t0})
            print(f"[wikipedia/{self.lang_code}] stopped: {self.total_chars:,} chars, "
                  f"{self.total_articles:,} articles, "
                  f"target_reached={self.total_chars >= self.target_chars}", flush=True)
