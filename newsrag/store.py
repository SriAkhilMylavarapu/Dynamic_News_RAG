"""SQLite metadata store: articles, chunks, a BM25 full-text index and feed state.

SQLite is the source of truth; the Chroma vector index can always be rebuilt
from it (`newsrag reindex`).
"""

from __future__ import annotations

import functools
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .models import Article, Hit
from .text import fts_query

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    source        TEXT NOT NULL,
    category      TEXT NOT NULL,
    region        TEXT NOT NULL,
    feed_url      TEXT NOT NULL,
    published_ts  REAL NOT NULL,
    fetched_ts    REAL NOT NULL,
    text          TEXT NOT NULL,
    full_text     INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    story_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_ts);
CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_story ON articles(story_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    article_id  TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(article_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, title, text, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS feed_state (
    url            TEXT PRIMARY KEY,
    etag           TEXT,
    last_modified  TEXT,
    last_checked   REAL,
    last_status    TEXT
);
"""


def _locked(method):
    """Serialise access: one connection is shared by ingestion threads (and later a web server)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _ts(dt: datetime) -> float:
    return dt.timestamp()


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    @_locked
    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------ articles --

    @_locked
    def known_ids(self, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        if not ids:
            return set()
        found: set[str] = set()
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            rows = self.conn.execute(
                f"SELECT id FROM articles WHERE id IN ({','.join('?' * len(batch))})", batch
            ).fetchall()
            found.update(r["id"] for r in rows)
        return found

    @_locked
    def find_by_content_hash(self, content_hash: str) -> str | None:
        row = self.conn.execute("SELECT id FROM articles WHERE content_hash = ? LIMIT 1", (content_hash,)).fetchone()
        return row["id"] if row else None

    @_locked
    def add_article(self, article: Article, chunks: Sequence[str]) -> list[str]:
        chunk_ids = [f"{article.id}:{i}" for i in range(len(chunks))]
        with self.conn:
            self.conn.execute(
                """INSERT INTO articles (id, url, title, source, category, region, feed_url, published_ts,
                       fetched_ts, text, full_text, content_hash, story_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    article.id, article.url, article.title, article.source, article.category, article.region,
                    article.feed_url, _ts(article.published_at), _ts(article.fetched_at), article.text,
                    int(article.full_text), article.content_hash, article.story_id or article.id,
                ),
            )
            self.conn.executemany(
                "INSERT INTO chunks (id, article_id, idx, text) VALUES (?,?,?,?)",
                [(cid, article.id, i, text) for i, (cid, text) in enumerate(zip(chunk_ids, chunks))],
            )
            self.conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, title, text) VALUES (?,?,?)",
                [(cid, article.title, text) for cid, text in zip(chunk_ids, chunks)],
            )
        return chunk_ids

    @_locked
    def get_article(self, article_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()

    @_locked
    def iter_articles(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM articles ORDER BY published_ts").fetchall()

    @_locked
    def chunks_for_article(self, article_id: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chunks WHERE article_id = ? ORDER BY idx", (article_id,)).fetchall()

    @_locked
    def prune_older_than(self, cutoff: datetime) -> tuple[list[str], list[str]]:
        """Delete articles published before `cutoff`. Returns (article_ids, chunk_ids) removed."""
        cutoff_ts = _ts(cutoff)
        article_ids = [r["id"] for r in self.conn.execute("SELECT id FROM articles WHERE published_ts < ?", (cutoff_ts,))]
        if not article_ids:
            return [], []
        chunk_ids: list[str] = []
        with self.conn:
            for i in range(0, len(article_ids), 500):
                batch = article_ids[i : i + 500]
                marks = ",".join("?" * len(batch))
                ids = [r["id"] for r in self.conn.execute(f"SELECT id FROM chunks WHERE article_id IN ({marks})", batch)]
                chunk_ids.extend(ids)
                for j in range(0, len(ids), 500):
                    sub = ids[j : j + 500]
                    self.conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({','.join('?' * len(sub))})", sub)
                self.conn.execute(f"DELETE FROM chunks WHERE article_id IN ({marks})", batch)
                self.conn.execute(f"DELETE FROM articles WHERE id IN ({marks})", batch)
        return article_ids, chunk_ids

    # -------------------------------------------------------------- search --

    @_locked
    def bm25_search(
        self,
        query: str,
        limit: int,
        since: datetime | None = None,
        categories: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Keyword search over chunks. Returns (chunk_id, score) with higher = better."""
        match = fts_query(query)
        if not match:
            return []
        sql = [
            "SELECT f.chunk_id AS chunk_id, bm25(chunks_fts, 0.0, 2.0, 1.0) AS rank",
            "FROM chunks_fts f JOIN chunks c ON c.id = f.chunk_id JOIN articles a ON a.id = c.article_id",
            "WHERE chunks_fts MATCH ?",
        ]
        params: list = [match]
        sql_filters, filter_params = self._filters(since, categories, sources)
        sql.extend(sql_filters)
        params.extend(filter_params)
        sql.append("ORDER BY rank LIMIT ?")
        params.append(limit)
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        # FTS5's bm25() is lower-is-better (negative); flip the sign.
        return [(r["chunk_id"], -r["rank"]) for r in rows]

    @staticmethod
    def _filters(since, categories, sources) -> tuple[list[str], list]:
        sql: list[str] = []
        params: list = []
        if since is not None:
            sql.append("AND a.published_ts >= ?")
            params.append(_ts(since))
        if categories:
            sql.append(f"AND a.category IN ({','.join('?' * len(categories))})")
            params.extend(categories)
        if sources:
            sql.append(f"AND lower(a.source) IN ({','.join('?' * len(sources))})")
            params.extend(s.lower() for s in sources)
        return sql, params

    @_locked
    def hits_for_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Hit]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"""SELECT c.id AS chunk_id, c.text AS text, a.id AS article_id, a.title, a.source, a.category,
                       a.url, a.published_ts, a.story_id
                FROM chunks c JOIN articles a ON a.id = c.article_id WHERE c.id IN ({marks})""",
            list(chunk_ids),
        ).fetchall()
        return {
            r["chunk_id"]: Hit(
                chunk_id=r["chunk_id"], article_id=r["article_id"], text=r["text"], title=r["title"],
                source=r["source"], category=r["category"], url=r["url"],
                published_at=_dt(r["published_ts"]), story_id=r["story_id"],
            )
            for r in rows
        }

    @_locked
    def top_stories(self, since: datetime, categories: Sequence[str] | None = None, limit: int = 10) -> list[dict]:
        """Stories published since `since`, ranked by how many outlets covered them."""
        sql = [
            """SELECT a.story_id AS story_id, COUNT(*) AS articles, COUNT(DISTINCT a.source) AS outlets,
                      MAX(a.published_ts) AS latest_ts, GROUP_CONCAT(DISTINCT a.source) AS source_list
               FROM articles a WHERE 1=1"""
        ]
        filters, params = self._filters(since, categories, None)
        sql.extend(filters)
        sql.append("GROUP BY a.story_id ORDER BY outlets DESC, articles DESC, latest_ts DESC LIMIT ?")
        params.append(limit)
        stories = []
        for r in self.conn.execute(" ".join(sql), params).fetchall():
            members = self.conn.execute(
                "SELECT id, title, source, url, published_ts, category FROM articles WHERE story_id = ? ORDER BY published_ts DESC",
                (r["story_id"],),
            ).fetchall()
            stories.append(
                {
                    "story_id": r["story_id"],
                    "outlets": r["outlets"],
                    "articles": [dict(m) | {"published_at": _dt(m["published_ts"]).isoformat()} for m in members],
                    "latest_at": _dt(r["latest_ts"]).isoformat(),
                }
            )
        return stories

    # --------------------------------------------------------- feed state --

    @_locked
    def feed_state(self, url: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM feed_state WHERE url = ?", (url,)).fetchone()

    @_locked
    def save_feed_state(self, url: str, etag: str | None, last_modified: str | None, status: str) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO feed_state (url, etag, last_modified, last_checked, last_status) VALUES (?,?,?,?,?)
                   ON CONFLICT(url) DO UPDATE SET etag=excluded.etag, last_modified=excluded.last_modified,
                   last_checked=excluded.last_checked, last_status=excluded.last_status""",
                (url, etag, last_modified, datetime.now(timezone.utc).timestamp(), status),
            )

    # --------------------------------------------------------------- stats --

    @_locked
    def stats(self) -> dict:
        one = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        by_source = self.conn.execute(
            "SELECT source, COUNT(*) AS n FROM articles GROUP BY source ORDER BY n DESC"
        ).fetchall()
        by_category = self.conn.execute(
            "SELECT category, COUNT(*) AS n FROM articles GROUP BY category ORDER BY n DESC"
        ).fetchall()
        oldest, newest = self.conn.execute("SELECT MIN(published_ts), MAX(published_ts) FROM articles").fetchone()
        return {
            "articles": one("SELECT COUNT(*) FROM articles"),
            "full_text_articles": one("SELECT COUNT(*) FROM articles WHERE full_text = 1"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "stories": one("SELECT COUNT(DISTINCT story_id) FROM articles"),
            "multi_outlet_stories": one(
                "SELECT COUNT(*) FROM (SELECT story_id FROM articles GROUP BY story_id HAVING COUNT(DISTINCT source) > 1)"
            ),
            "oldest": _dt(oldest).isoformat() if oldest else None,
            "newest": _dt(newest).isoformat() if newest else None,
            "by_source": {r["source"]: r["n"] for r in by_source},
            "by_category": {r["category"]: r["n"] for r in by_category},
        }
