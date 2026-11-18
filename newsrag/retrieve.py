"""Hybrid retrieval: BM25 + dense vectors, fused, re-weighted for recency and
diversified across articles and stories.

Two fusion methods are available:

* "convex" (default): min-max normalise BM25 and cosine scores over the
  candidate pool and take a weighted sum. Keeps the size of the relevance gap
  between candidates, so a weak match can't overtake a strong one just by being
  newer.
* "rrf": reciprocal rank fusion. Scale-free and simple, but adjacent ranks get
  almost the same score (1/61 vs 1/62), so any recency boost can reorder them.

`newsrag eval` compares the two on your own data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from .config import Settings
from .models import Hit
from .query import asks_for_routine, is_routine
from .store import Store
from .text import TimeWindow, parse_time_window
from .vectors import Embedder, VectorIndex

Mode = Literal["hybrid", "bm25", "dense"]
Fusion = Literal["convex", "rrf"]
RRF_K = 60
ROUTINE_PENALTY = 0.5  # fuel prices, "stocks to watch"... unless the question asks for them
MIN_RELEVANCE = 0.05  # convex scores are 0..1; below this a passage matched nothing useful
# Drop passages scoring under this fraction of the best match. When a question has only
# three good matches, the other five slots would otherwise be filled with unrelated news.
RELATIVE_CUTOFF = 0.4


@dataclass
class SearchResult:
    hits: list[Hit]
    window: TimeWindow
    relaxed: bool  # True if the time filter matched nothing and was dropped


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion: robust to the two retrievers' scores being on different scales."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank + 1)
    return dict(scores)


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def convex(bm25: dict[str, float], cosine: dict[str, float], alpha: float = 0.5) -> dict[str, float]:
    """alpha * dense + (1 - alpha) * keyword, each min-max normalised. Missing scores count as 0."""
    b, c = minmax(bm25), minmax(cosine)
    if not b:
        return c
    if not c:
        return b
    return {k: alpha * c.get(k, 0.0) + (1 - alpha) * b.get(k, 0.0) for k in set(b) | set(c)}


def recency_decay(published_at: datetime, now: datetime, half_life_hours: float) -> float:
    age_hours = max((now - published_at).total_seconds() / 3600, 0.0)
    return 0.5 ** (age_hours / half_life_hours)


class Retriever:
    def __init__(self, settings: Settings, store: Store, index: VectorIndex, embedder: Embedder):
        self.settings = settings
        self.store = store
        self.index = index
        self.embedder = embedder

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        since: datetime | None = None,
        categories: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        mode: Mode = "hybrid",
        fusion: Fusion = "convex",
        use_recency: bool = True,
        use_time_parsing: bool = True,
        max_per_article: int = 2,
        max_per_story: int = 3,
        max_per_source: int | None = 2,
        relative_cutoff: float = RELATIVE_CUTOFF,
        now: datetime | None = None,
    ) -> SearchResult:
        k = k or self.settings.top_k
        now = now or datetime.now(timezone.utc)
        window = parse_time_window(query, now) if use_time_parsing else TimeWindow(None, None, False)
        effective_since = since or window.since

        if sources:
            max_per_source = None  # the user asked for these outlets specifically
        args = (query, k, categories, sources, mode, fusion, use_recency,
                (max_per_article, max_per_story, max_per_source, relative_cutoff), now)
        hits = self._search(*args, since=effective_since, wants_recent=window.wants_recent)
        relaxed = False
        if not hits and effective_since is not None and since is None:
            hits = self._search(*args, since=None, wants_recent=True)
            relaxed = True
        return SearchResult(hits, window, relaxed)

    def _search(self, query, k, categories, sources, mode, fusion, use_recency, caps,
                now, *, since, wants_recent) -> list[Hit]:
        max_per_article, max_per_story, max_per_source, relative_cutoff = caps
        pool = max(k * 6, 40)
        rankings: list[list[str]] = []
        cosine: dict[str, float] = {}
        bm25: dict[str, float] = {}

        if mode in ("hybrid", "bm25"):
            found = self.store.bm25_search(query, pool, since, categories, sources)
            bm25 = dict(found)
            rankings.append([cid for cid, _ in found])
        if mode in ("hybrid", "dense"):
            qvec = self.embedder.embed_query(query)
            found = self.index.search_chunks(qvec, pool, since, categories, sources)
            cosine = dict(found)
            rankings.append([cid for cid, _ in found])
            # Score keyword-only candidates on the dense side too, so fusion compares like with like.
            cosine |= self.index.cosine_for([c for c in bm25 if c not in cosine], qvec)

        if fusion == "rrf":
            relevance = rrf(rankings)
        elif mode == "bm25":
            relevance = minmax(bm25)
        elif mode == "dense":
            relevance = minmax(cosine)
        else:
            relevance = convex(bm25, cosine)
        if not relevance:
            return []
        hits = self.store.hits_for_chunks(list(relevance))

        # "latest"/"today" questions lean harder on freshness.
        half_life = self.settings.recency_half_life_hours / (3 if wants_recent else 1)
        weight = 0.5 if wants_recent else 0.2
        demote_routine = not asks_for_routine(query)
        for cid, hit in hits.items():
            decay = recency_decay(hit.published_at, now, half_life)
            hit.signals = {"relevance": relevance[cid], "recency": decay}
            if cid in cosine:
                hit.signals["cosine"] = cosine[cid]
            if cid in bm25:
                hit.signals["bm25"] = bm25[cid]
            hit.score = relevance[cid] * ((1 - weight) + weight * decay) if use_recency else relevance[cid]
            if demote_routine and is_routine(hit.title):
                hit.score *= ROUTINE_PENALTY
                hit.signals["routine"] = 1.0

        ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)
        floor = relative_cutoff * max(relevance.values()) if fusion == "convex" else 0.0
        per_article: dict[str, int] = defaultdict(int)
        per_story: dict[str, int] = defaultdict(int)
        per_source: dict[str, int] = defaultdict(int)
        out: list[Hit] = []
        source_capped: list[Hit] = []
        for hit in ranked:
            if fusion == "convex" and hit.signals["relevance"] < max(MIN_RELEVANCE, floor):
                continue
            story = hit.story_id or hit.article_id
            if per_article[hit.article_id] >= max_per_article or per_story[story] >= max_per_story:
                continue
            if max_per_source is not None and per_source[hit.source] >= max_per_source:
                source_capped.append(hit)
                continue
            per_article[hit.article_id] += 1
            per_story[story] += 1
            per_source[hit.source] += 1
            out.append(hit)
            if len(out) == k:
                break
        # A niche topic may only be covered by one or two outlets; better to
        # repeat an outlet than to return too few passages.
        for hit in source_capped:
            if len(out) == k:
                break
            story = hit.story_id or hit.article_id
            if per_article[hit.article_id] >= max_per_article or per_story[story] >= max_per_story:
                continue
            per_article[hit.article_id] += 1
            per_story[story] += 1
            out.append(hit)
        return out
