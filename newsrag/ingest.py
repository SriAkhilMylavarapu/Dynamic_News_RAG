"""Ingestion pipeline: poll feeds -> dedupe -> scrape -> cluster into stories -> chunk -> index."""

from __future__ import annotations

import logging
import queue
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import feedparser
import requests

from .config import Feed, Settings
from .extract import clean_summary, extract_article
from .fetch import Fetcher
from .models import Article, IngestReport
from .store import Store
from .text import canonical_url, chunk_text, content_hash, entry_datetime, url_id
from .vectors import Embedder, VectorIndex

log = logging.getLogger(__name__)

MIN_SUMMARY_CHARS = 120
INDEX_BATCH = 25  # articles are saved in batches as they're scraped, so an interrupted run keeps its work

# Video, photo and audio pages carry little or no article text.
_MEDIA_PATH = re.compile(
    r"/(videos?|video-news|photos?|photogallery|photo-gallery|gallery|galleries|slideshows?|"
    r"podcasts?|audio|web-?stories|visual-?stories|shorts|reels)(/|$)",
    re.I,
)


@dataclass
class Candidate:
    id: str
    url: str
    title: str
    summary: str
    published_at: datetime | None
    feed: Feed


def looks_like_article(url: str) -> bool:
    """Section fronts like https://site.com/india/ show up in feeds; skip them."""
    path = urlsplit(url).path.strip("/")
    if _MEDIA_PATH.search("/" + path):
        return False
    return bool(path) and (path.count("/") >= 1 or len(path) > 30 or any(c.isdigit() for c in path))


def embedding_text(title: str, body: str) -> str:
    # Prefixing the headline gives each chunk the article's topic, which helps
    # chunks from the middle of an article that never repeat the subject.
    return f"{title}\n\n{body}"


class Ingestor:
    def __init__(self, settings: Settings, store: Store, index: VectorIndex, embedder: Embedder, fetcher: Fetcher | None = None):
        self.settings = settings
        self.store = store
        self.index = index
        self.embedder = embedder
        self.fetcher = fetcher or Fetcher(
            settings.user_agent, settings.request_timeout, settings.per_domain_delay, settings.respect_robots
        )

    # -------------------------------------------------------------- stage 1 --

    def poll_feed(self, feed: Feed, report: IngestReport) -> list[Candidate]:
        state = self.store.feed_state(feed.url)
        try:
            res = self.fetcher.get(feed.url, state["etag"] if state else None, state["last_modified"] if state else None)
        except requests.RequestException as exc:
            report.feeds_failed += 1
            report.errors.append(f"feed {feed.url}: {exc}")
            self.store.save_feed_state(feed.url, state["etag"] if state else None, state["last_modified"] if state else None, f"error: {exc}")
            return []
        if res.status == 304:
            report.feeds_unchanged += 1
            self.store.save_feed_state(feed.url, res.etag, res.last_modified, "not modified")
            return []

        parsed = feedparser.parse(res.text)
        if parsed.bozo and not parsed.entries:
            report.feeds_failed += 1
            report.errors.append(f"feed {feed.url}: unparseable ({parsed.bozo_exception})")
            self.store.save_feed_state(feed.url, None, None, "unparseable")
            return []
        self.store.save_feed_state(feed.url, res.etag, res.last_modified, f"ok: {len(parsed.entries)} entries")

        out = []
        for entry in parsed.entries[: self.settings.max_items_per_feed]:
            link = entry.get("link")
            title = (entry.get("title") or "").strip()
            if not link or not title or not looks_like_article(link):
                continue
            out.append(
                Candidate(
                    id=url_id(link),
                    url=link,
                    title=title,
                    summary=clean_summary(entry.get("summary") or entry.get("description")),
                    published_at=entry_datetime(entry),
                    feed=feed,
                )
            )
        return out

    # -------------------------------------------------------------- stage 2 --

    def scrape(self, cand: Candidate) -> tuple[str, bool] | None:
        """Returns (text, is_full_text) or None if there's nothing usable."""
        try:
            if self.fetcher.allowed(cand.url):
                res = self.fetcher.get(cand.url)
                if res.text:
                    body = extract_article(res.text, res.final_url or cand.url)
                    if body:
                        return body, True
        except requests.RequestException as exc:
            log.debug("scrape failed for %s: %s", cand.url, exc)
        if len(cand.summary) >= MIN_SUMMARY_CHARS:
            return cand.summary, False
        return None

    # -------------------------------------------------------------- run -----

    def run(self, feeds: list[Feed]) -> IngestReport:
        report = IngestReport()
        started = time.monotonic()
        now = datetime.now(timezone.utc)
        oldest_allowed = now - timedelta(days=self.settings.retention_days)

        # 1. Poll all feeds in parallel.
        candidates: dict[str, Candidate] = {}
        with ThreadPoolExecutor(self.settings.max_workers) as pool:
            for future in as_completed([pool.submit(self.poll_feed, f, report) for f in feeds]):
                for cand in future.result():
                    report.entries_seen += 1
                    # The same link often appears in several sections of one outlet.
                    candidates.setdefault(cand.id, cand)
        report.feeds_checked = len(feeds)

        # 2. Drop links already in the database and stale entries.
        known = self.store.known_ids(candidates)
        report.already_known = len(known)
        fresh = [
            c for cid, c in candidates.items()
            if cid not in known and (c.published_at is None or c.published_at >= oldest_allowed)
        ]
        log.info("%d new links to fetch (%d already known)", len(fresh), len(known))

        # 3. Scrape: one worker per site, so sites are fetched in parallel while
        #    each site still gets at most one request per `per_domain_delay`.
        #    Results stream back and are indexed in batches as they arrive.
        by_domain: dict[str, list[Candidate]] = defaultdict(list)
        for cand in sorted(fresh, key=lambda c: c.published_at or now, reverse=True):
            by_domain[urlsplit(cand.url).netloc].append(cand)
        results: queue.Queue = queue.Queue()

        def scrape_domain(cands: list[Candidate]) -> None:
            for cand in cands:
                try:
                    result = self.scrape(cand)
                except Exception as exc:  # one bad page must not stop the site's queue
                    log.warning("scrape crashed for %s: %s", cand.url, exc)
                    result = None
                results.put((cand, result))

        batch: list[tuple[Candidate, str, bool]] = []
        workers = max(1, min(len(by_domain), 32))  # I/O-bound; the per-site delay is what limits load
        with ThreadPoolExecutor(workers) as pool:
            for cands in by_domain.values():
                pool.submit(scrape_domain, cands)
            for done in range(1, len(fresh) + 1):
                cand, result = results.get()
                if result:
                    batch.append((cand, *result))
                if len(batch) >= INDEX_BATCH or (done == len(fresh) and batch):
                    self._store_batch(batch, report, now)
                    batch = []
                if done % 100 == 0:
                    log.info("scraped %d/%d pages, %d articles added so far", done, len(fresh), report.added)

        # 4. Retention.
        report.pruned = self.prune(oldest_allowed)
        report.seconds = round(time.monotonic() - started, 2)
        return report

    def _store_batch(self, scraped: list[tuple[Candidate, str, bool]], report: IngestReport, now: datetime) -> None:
        """Dedupe by content, group into stories, chunk, embed and save one batch."""
        articles: list[tuple[Article, list[str]]] = []
        for cand, text, full in sorted(scraped, key=lambda item: item[0].published_at or now):
            chash = content_hash(text)
            if self.store.find_by_content_hash(chash) or any(a.content_hash == chash for a, _ in articles):
                report.duplicate_content += 1
                continue
            published = min(cand.published_at or now, now)
            chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
            if not chunks:
                continue
            articles.append(
                (
                    Article(
                        id=cand.id, url=canonical_url(cand.url), title=cand.title, source=cand.feed.source,
                        category=cand.feed.category, region=cand.feed.region, feed_url=cand.feed.url,
                        published_at=published, fetched_at=now, text=text, full_text=full, content_hash=chash,
                    ),
                    chunks,
                )
            )
        if articles:
            self._index(articles, report)

    def _index(self, articles: list[tuple[Article, list[str]]], report: IngestReport) -> None:
        leads = self.embedder.embed_documents([embedding_text(a.title, chunks[0][:500]) for a, chunks in articles])
        window = timedelta(hours=self.settings.story_window_hours)
        for (article, chunks), lead in zip(articles, leads):
            match = self.index.nearest_lead(lead, since=article.published_at - window)
            if match and match[1] >= self.settings.story_similarity:
                article.story_id = match[2].get("story_id") or match[0]
                report.joined_existing_story += 1
            else:
                article.story_id = article.id
            chunk_ids = self.store.add_article(article, chunks)
            meta = {
                "article_id": article.id,
                "story_id": article.story_id,
                "source_lc": article.source.lower(),
                "category": article.category,
                "published_ts": article.published_at.timestamp(),
            }
            self.index.add_lead(article.id, lead, meta)
            vectors = self.embedder.embed_documents([embedding_text(article.title, c) for c in chunks])
            self.index.add_chunks(chunk_ids, vectors, [meta] * len(chunk_ids))
            report.added += 1
            report.added_summary_only += int(not article.full_text)
            report.chunks_added += len(chunks)

    def prune(self, cutoff: datetime) -> int:
        article_ids, chunk_ids = self.store.prune_older_than(cutoff)
        if article_ids:
            self.index.delete(article_ids, chunk_ids)
        return len(article_ids)

    def reindex(self) -> int:
        """Rebuild the vector index from SQLite (e.g. after changing embedding model)."""
        self.index.reset()
        count = 0
        for row in self.store.iter_articles():
            chunks = self.store.chunks_for_article(row["id"])
            if not chunks:
                continue
            meta = {
                "article_id": row["id"],
                "story_id": row["story_id"],
                "source_lc": row["source"].lower(),
                "category": row["category"],
                "published_ts": row["published_ts"],
            }
            lead = self.embedder.embed_documents([embedding_text(row["title"], chunks[0]["text"][:500])])[0]
            self.index.add_lead(row["id"], lead, meta)
            vecs = self.embedder.embed_documents([embedding_text(row["title"], c["text"]) for c in chunks])
            self.index.add_chunks([c["id"] for c in chunks], vecs, [meta] * len(chunks))
            count += 1
        return count
