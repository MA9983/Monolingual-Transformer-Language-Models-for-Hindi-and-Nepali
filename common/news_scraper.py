"""Respectful, checkpointed news-site crawler used for the manual/scraped
portion (>=20% of tokens) of each language's corpus.

Design choices, and why:
- URL discovery is sitemap-based: robots.txt's declared Sitemap: entries are
  fetched and, if they are sitemap *indexes*, recursively expanded to their
  child sitemaps. This is how a handful of entry points turn into the
  large historical URL universe needed for real corpus volume, without ever
  guessing or brute-forcing URLs.
- Every fetch is gated by RobotsCache.can_fetch() (see robots_utils.py) and
  paced by the site's declared crawl-delay (or a polite default minimum),
  scheduled round-robin across sites so no single domain is hammered.
- Sites that block AI/anthropic crawlers or state an anti-AI-training policy
  in robots.txt are excluded upstream, at the per-language site-list level
  (see hindi/data/collect_manual_news.py and nepali/.../collect_manual_news.py)
  -- this module assumes every site it is given has already cleared that bar.
- Progress (seen URLs + running totals) is checkpointed to disk continuously
  so a multi-hour crawl can be killed and resumed without re-fetching or
  double-counting anything, mirroring the checkpoint-resume discipline the
  project requires for model training.
"""

import hashlib
import json
import os
import queue
import threading
import time
from urllib.parse import urljoin, urlparse

import feedparser
import requests
import trafilatura
from langdetect import DetectorFactory, detect
from lxml import etree

from robots_utils import RobotsCache

DetectorFactory.seed = 0  # deterministic langdetect

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 "
    "(LMA-course-project research crawler; academic dataset collection; "
    "contact: ahmadmohd311@gmail.com)"
)
REQUEST_TIMEOUT = 20
NON_ARTICLE_PATTERNS = (
    "/video/", "/videos/", "/photo/", "/photos/", "/photo-gallery/",
    "/gallery/", "/web-stories/", "/webstories/", "/live-blog/", "/liveblog/",
    "/live-updates", "/shorts/", "/podcast/", "/audio/", "/login", "/search",
    "/tag/", "/topic/", "/author/", "/rashifal", "/horoscope", "/astrology",
    ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".pdf",
)
MIN_ARTICLE_CHARS = 300
MAX_SITEMAP_DEPTH = 3
# Each worker independently paces itself at the site's own crawl-delay
# (e.g. 1 request/2s each), so N workers on one site achieve roughly N
# times the aggregate throughput while no single connection ever exceeds
# the polite per-connection rate -- the same model Scrapy's
# CONCURRENT_REQUESTS_PER_DOMAIN uses. Kept modest (not dozens) so this
# stays "friendly low-speed" in aggregate, not a burst hammering one site.
WORKERS_PER_SITE = 3

# Many sites host a sitemap/feed without declaring it in robots.txt, so these
# common paths are always tried in addition to whatever robots.txt lists.
FALLBACK_SITEMAP_PATHS = [
    "/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
    "/wp-sitemap.xml", "/news-sitemap.xml",
]
FALLBACK_FEED_PATHS = ["/feed", "/feed/", "/rss", "/rss.xml", "/feeds/all.xml"]


def _fetch_bytes(url, headers):
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _parse_sitemap_xml(raw_bytes):
    """Return (child_sitemap_urls, page_urls) from one sitemap XML document."""
    try:
        root = etree.fromstring(raw_bytes)
    except etree.XMLSyntaxError:
        return [], []
    tag = etree.QName(root.tag).localname
    ns = {"sm": root.nsmap.get(None, "")} if root.nsmap.get(None) else {}
    locs = [el.text.strip() for el in root.iter() if etree.QName(el.tag).localname == "loc" and el.text]
    if tag == "sitemapindex":
        return locs, []
    return [], locs


def _looks_like_xml(raw_bytes):
    head = raw_bytes[:200].lstrip().lower()
    return head.startswith(b"<?xml") or head.startswith(b"<urlset") or head.startswith(b"<sitemapindex")


def discover_article_urls(homepage, robots_cache, max_urls, log_prefix=""):
    """Discover candidate article URLs for one site from every source we
    know how to try: robots.txt-declared sitemaps, common sitemap paths not
    declared in robots.txt, and common RSS/Atom feed paths. Sitemap indexes
    are recursively expanded up to MAX_SITEMAP_DEPTH. Returns a
    de-duplicated list capped at max_urls.

    Real sites vary a lot here -- some have deep dated archives, some only
    expose a rolling recent window, some have broken/404ing sitemap links
    despite listing them, some have no sitemap at all. This function tries
    everything plausible and lets the caller see (via the printed log) what
    actually worked for each site, rather than assuming one recipe fits all.
    """
    headers = {"User-Agent": USER_AGENT}
    seen_sitemaps = set()
    declared = list(robots_cache.sitemaps(homepage))
    fallback = [urljoin(homepage, p) for p in FALLBACK_SITEMAP_PATHS]
    frontier = [(u, 0) for u in declared] + [(u, 0) for u in fallback if u not in declared]

    found = []
    found_set = set()
    while frontier and len(found) < max_urls:
        sitemap_url, depth = frontier.pop(0)
        if sitemap_url in seen_sitemaps or depth > MAX_SITEMAP_DEPTH:
            continue
        seen_sitemaps.add(sitemap_url)
        if not robots_cache.can_fetch(sitemap_url):
            continue
        try:
            raw = _fetch_bytes(sitemap_url, headers)
        except Exception as e:
            print(f"{log_prefix}  [sitemap fetch failed] {sitemap_url}: {e}", flush=True)
            continue
        if not _looks_like_xml(raw):
            continue  # fallback-guessed path resolved to an HTML page, not a real sitemap
        children, pages = _parse_sitemap_xml(raw)
        for c in children:
            frontier.append((c, depth + 1))
        for p in pages:
            if p in found_set:
                continue
            path = urlparse(p).path.lower()
            if any(pat in path for pat in NON_ARTICLE_PATTERNS):
                continue
            found_set.add(p)
            found.append(p)
            if len(found) >= max_urls:
                break
        if pages or children:
            print(f"{log_prefix}  [sitemap] {sitemap_url} -> +{len(pages)} urls, "
                  f"+{len(children)} child sitemaps (total collected: {len(found)})", flush=True)

    if len(found) < max_urls:
        for feed_path in FALLBACK_FEED_PATHS:
            feed_url = urljoin(homepage, feed_path)
            if not robots_cache.can_fetch(feed_url):
                continue
            try:
                r = requests.get(feed_url, headers=headers, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    continue
                parsed = feedparser.parse(r.content)
            except Exception:
                continue
            new_count = 0
            for entry in getattr(parsed, "entries", []):
                link = entry.get("link")
                if not link or link in found_set:
                    continue
                path = urlparse(link).path.lower()
                if any(pat in path for pat in NON_ARTICLE_PATTERNS):
                    continue
                found_set.add(link)
                found.append(link)
                new_count += 1
            if new_count:
                print(f"{log_prefix}  [feed] {feed_url} -> +{new_count} urls "
                      f"(total collected: {len(found)})", flush=True)
    return found


def _content_hash(text):
    return hashlib.sha1(text[:500].encode("utf-8", errors="ignore")).hexdigest()


class NewsCollector:
    """Round-robin, checkpointed crawler across a list of sites for one
    language. See collect() for the main loop.
    """

    def __init__(self, sites, lang_code, target_chars, out_dir, chars_per_token=4.5):
        """sites: list of {"name": str, "homepage": str} dicts, all
        pre-cleared for robots.txt / anti-AI-crawling compliance.
        lang_code: langdetect ISO code the extracted text must match (e.g.
        'hi' or 'ne').
        """
        self.sites = sites
        self.lang_code = lang_code
        self.target_chars = target_chars
        self.chars_per_token = chars_per_token
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.text_path = os.path.join(out_dir, "scraped_articles.txt")
        self.meta_path = os.path.join(out_dir, "scraped_articles_meta.jsonl")
        self.seen_urls_path = os.path.join(out_dir, "seen_urls.txt")
        self.progress_path = os.path.join(out_dir, "progress.json")
        self.robots = RobotsCache(USER_AGENT)

        self.seen_urls = set()
        if os.path.exists(self.seen_urls_path):
            with open(self.seen_urls_path, encoding="utf-8") as f:
                self.seen_urls = set(line.strip() for line in f if line.strip())
        # Resume counts are derived from meta.jsonl -- the append-only,
        # per-article source of truth -- rather than trusted from
        # progress.json, which is only written periodically and would
        # silently undercount everything if a previous run was killed
        # (e.g. Ctrl-C, OOM, a Colab disconnect) between two of its
        # checkpoints.
        self.seen_hashes = set()
        self.total_chars = 0
        self.total_articles = 0
        self.per_site_chars = {s["name"]: 0 for s in sites}
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
                    self.per_site_chars[rec["site"]] = self.per_site_chars.get(rec["site"], 0) + rec["n_chars"]

    def _save_progress(self, extra=None):
        prog = {
            "total_chars": self.total_chars,
            "total_articles": self.total_articles,
            "estimated_tokens": self.total_chars / self.chars_per_token,
            "target_chars": self.target_chars,
            "per_site_chars": self.per_site_chars,
        }
        if extra:
            prog.update(extra)
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump(prog, f, ensure_ascii=False, indent=2)

    def _fetch_and_extract(self, url):
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        # Pass raw bytes rather than r.text: several sites (e.g.
        # onlinekhabar.com) send `Content-Type: text/html` with no charset,
        # which makes requests fall back to ISO-8859-1 and mangle the
        # Devanagari into mojibake. trafilatura does its own HTML-aware
        # encoding detection (meta charset tags, etc.) on raw bytes, which
        # gets this right.
        text = trafilatura.extract(r.content, favor_recall=False,
                                    include_comments=False, include_tables=False)
        return text

    def _worker(self, site, url_queue, stop_event, t0):
        """One of WORKERS_PER_SITE threads for this site: pulls URLs off
        its site's queue, paces itself independently at the site's own
        crawl-delay, and commits kept articles under self.lock. Multiple
        workers per site is what turns the per-site crawl-delay into real
        aggregate throughput instead of a single-threaded ceiling.
        """
        next_allowed = 0.0
        while not stop_event.is_set():
            try:
                url = url_queue.get_nowait()
            except queue.Empty:
                return

            wait = next_allowed - time.time()
            if wait > 0:
                time.sleep(wait)
            delay = self.robots.crawl_delay(site["homepage"])
            next_allowed = time.time() + delay

            with self.lock:
                if url in self.seen_urls:
                    continue
                self.seen_urls.add(url)
                self.seen_f.write(url + "\n")
                self.seen_f.flush()

            if not self.robots.can_fetch(url):
                continue
            try:
                text = self._fetch_and_extract(url)
            except Exception:
                continue
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
                    "url": url, "site": site["name"], "hash": h, "n_chars": len(text),
                }, ensure_ascii=False) + "\n")
                self.text_f.flush()
                self.meta_f.flush()

                self.total_chars += len(text)
                self.total_articles += 1
                self.per_site_chars[site["name"]] += len(text)

                if self.total_chars - self.last_log >= 2_000_000:
                    self.last_log = self.total_chars
                    elapsed = (time.time() - t0) / 60
                    print(f"[progress] {self.total_chars/1e6:.1f}M chars "
                          f"(~{self.total_chars/self.chars_per_token/1e6:.2f}M est. tokens), "
                          f"{self.total_articles:,} articles, {elapsed:.1f} min elapsed, "
                          f"per-site: {self.per_site_chars}", flush=True)
                    self._save_progress()

                if self.total_chars >= self.target_chars:
                    stop_event.set()

    def collect(self):
        t0 = time.time()
        print(f"[{self.lang_code}] resume state: {self.total_chars:,} chars "
              f"({self.total_chars/self.chars_per_token/1e6:.1f}M est. tokens), "
              f"{self.total_articles:,} articles, {len(self.seen_urls):,} urls already seen",
              flush=True)
        if self.total_chars >= self.target_chars:
            print(f"[{self.lang_code}] target already reached, nothing to do.", flush=True)
            return

        # Discover a large batch of candidate URLs per site up front.
        site_queues = {}
        for site in self.sites:
            remaining_est_chars = self.target_chars - self.total_chars
            max_urls = max(2000, int(remaining_est_chars / 1200))  # ~1200 chars/article floor estimate
            urls = discover_article_urls(site["homepage"], self.robots, max_urls,
                                          log_prefix=f"[{site['name']}]")
            urls = [u for u in urls if u not in self.seen_urls]
            q = queue.Queue()
            for u in urls:
                q.put(u)
            site_queues[site["name"]] = q
            print(f"[{site['name']}] {len(urls):,} new candidate urls queued "
                  f"({WORKERS_PER_SITE} workers)", flush=True)

        self.lock = threading.Lock()
        self.last_log = self.total_chars
        self.text_f = open(self.text_path, "a", encoding="utf-8")
        self.meta_f = open(self.meta_path, "a", encoding="utf-8")
        self.seen_f = open(self.seen_urls_path, "a", encoding="utf-8")
        stop_event = threading.Event()

        try:
            threads = []
            for site in self.sites:
                for _ in range(WORKERS_PER_SITE):
                    t = threading.Thread(target=self._worker,
                                          args=(site, site_queues[site["name"]], stop_event, t0),
                                          daemon=True)
                    t.start()
                    threads.append(t)
            for t in threads:
                t.join()
        finally:
            self.text_f.close()
            self.meta_f.close()
            self.seen_f.close()
            self._save_progress({"finished": True, "elapsed_seconds": time.time() - t0})
            print(f"[{self.lang_code}] stopped: {self.total_chars:,} chars, "
                  f"{self.total_articles:,} articles, "
                  f"target_reached={self.total_chars >= self.target_chars}", flush=True)
