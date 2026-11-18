"""Polite HTTP fetching: per-domain rate limiting, robots.txt, conditional GETs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests


@dataclass
class FetchResult:
    status: int
    text: str | None
    etag: str | None = None
    last_modified: str | None = None
    final_url: str | None = None


class Fetcher:
    def __init__(self, user_agent: str, timeout: float = 15, per_domain_delay: float = 1.0, respect_robots: bool = True):
        self.user_agent = user_agent
        self.timeout = timeout
        self.per_domain_delay = per_domain_delay
        self.respect_robots = respect_robots
        self._local = threading.local()
        self._domain_locks: dict[str, threading.Lock] = {}
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._guard = threading.Lock()

    @property
    def session(self) -> requests.Session:
        # requests.Session is not guaranteed thread-safe; keep one per thread.
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update({"User-Agent": self.user_agent, "Accept-Language": "en"})
            self._local.session = s
        return self._local.session

    def _domain_lock(self, domain: str) -> threading.Lock:
        with self._guard:
            return self._domain_locks.setdefault(domain, threading.Lock())

    def _wait_turn(self, domain: str) -> None:
        with self._domain_lock(domain):
            wait = self._last_hit.get(domain, 0) + self.per_domain_delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_hit[domain] = time.monotonic()

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        with self._domain_lock(parts.netloc):
            if base not in self._robots:
                parser = RobotFileParser()
                try:
                    resp = self.session.get(f"{base}/robots.txt", timeout=self.timeout)
                    if resp.status_code >= 400:
                        parser = None  # no robots.txt -> everything allowed
                    else:
                        parser.parse(resp.text.splitlines())
                except requests.RequestException:
                    parser = None
                self._robots[base] = parser
        parser = self._robots[base]
        return parser is None or parser.can_fetch(self.user_agent, url)

    def get(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        self._wait_turn(urlsplit(url).netloc)
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        resp = self.session.get(url, headers=headers, timeout=self.timeout)
        if resp.status_code == 304:
            return FetchResult(304, None, etag, last_modified, resp.url)
        resp.raise_for_status()
        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding
        return FetchResult(
            resp.status_code,
            resp.text,
            resp.headers.get("ETag"),
            resp.headers.get("Last-Modified"),
            resp.url,
        )
