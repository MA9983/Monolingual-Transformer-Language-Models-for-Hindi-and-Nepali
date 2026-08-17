"""Small helper around urllib.robotparser for polite, robots.txt-compliant
crawling: per-domain caching of the parsed robots.txt, fetch permission
checks, and crawl-delay lookup.
"""

import urllib.robotparser
from urllib.parse import urlparse

import requests

DEFAULT_MIN_DELAY_SECONDS = 2.0


class RobotsCache:
    """Caches one urllib.robotparser.RobotFileParser per domain so robots.txt
    is fetched once per site, not once per URL.
    """

    def __init__(self, user_agent):
        self.user_agent = user_agent
        self._parsers = {}

    def _get_parser(self, url):
        domain = urlparse(url).scheme + "://" + urlparse(url).netloc
        if domain not in self._parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(domain + "/robots.txt")
            try:
                # Fetch and parse manually (rather than rp.read(), which uses
                # urllib's urlopen) so a leading UTF-8 BOM -- present on some
                # sites' robots.txt and enough to make robotparser silently
                # discard every directive, including Sitemap: entries -- can
                # be stripped before urllib.robotparser ever sees the text.
                resp = requests.get(domain + "/robots.txt", timeout=15,
                                     headers={"User-Agent": self.user_agent})
                text = resp.content.decode("utf-8-sig", errors="ignore")
                rp.parse(text.splitlines())
            except Exception:
                rp = None  # treat unreadable robots.txt as permissive
            self._parsers[domain] = rp
        return self._parsers[domain]

    def can_fetch(self, url):
        """Return True if this crawler's user agent is allowed to fetch url."""
        rp = self._get_parser(url)
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def crawl_delay(self, url):
        """Return the site's declared crawl-delay in seconds, or the
        project's default polite minimum, whichever is larger.
        """
        rp = self._get_parser(url)
        delay = None
        if rp is not None:
            try:
                delay = rp.crawl_delay(self.user_agent)
            except Exception:
                delay = None
        return max(delay or 0, DEFAULT_MIN_DELAY_SECONDS)

    def sitemaps(self, homepage_url):
        """Return the Sitemap: URLs declared in the site's robots.txt."""
        rp = self._get_parser(homepage_url)
        if rp is None:
            return []
        try:
            return rp.site_maps() or []
        except Exception:
            return []
