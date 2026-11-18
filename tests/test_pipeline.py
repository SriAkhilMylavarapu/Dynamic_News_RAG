"""End-to-end: fake feeds -> ingest -> search -> ask -> briefing, all offline."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from newsrag.assistant import NewsAssistant
from newsrag.config import Feed, Settings
from newsrag.fetch import FetchResult
from newsrag.ingest import Ingestor
from newsrag.store import Store
from newsrag.vectors import HashEmbedder, VectorIndex

NOW = datetime.now(timezone.utc)

FLOOD = (
    "Heavy rain flooded large parts of Chennai on Tuesday, submerging roads and halting suburban trains. "
    "The Tamil Nadu government declared a holiday for schools and colleges in Chennai, Chengalpattu and Tiruvallur. "
    "The India Meteorological Department issued a red alert for the coastal districts until Thursday. "
    "Rescue teams from the National Disaster Response Force evacuated more than 3,000 residents from low-lying areas. "
)
FLOOD_B = (
    "Chennai was flooded after heavy rain on Tuesday, with roads submerged and suburban trains halted. "
    "Schools and colleges in Chennai, Chengalpattu and Tiruvallur were shut by the Tamil Nadu government. "
    "A red alert for coastal districts remains in place until Thursday, the India Meteorological Department said. "
    "National Disaster Response Force teams evacuated over 3,000 residents from low-lying areas of the city. "
)
RBI = (
    "The Reserve Bank of India kept the repo rate unchanged at 5.5 per cent on Friday. "
    "Governor Sanjay Malhotra said inflation was within the target band and growth remained resilient. "
    "The monetary policy committee voted unanimously and kept a neutral stance. "
    "Bond yields eased slightly after the announcement while the rupee was steady against the dollar. "
)
CRICKET = (
    "India beat Australia by six wickets in the second Test in Adelaide to level the series. "
    "Jasprit Bumrah took nine wickets in the match and was named player of the match. "
    "Shubman Gill scored an unbeaten 84 in the chase of 211 on the fourth day. "
    "The third Test begins in Brisbane next week. "
)


def page(body: str) -> str:
    paras = "".join(f"<p>{s.strip()}.</p>" for s in body.split(". ") if s.strip())
    return f"<html><body><nav>Home | India | World</nav><article><h1>Headline</h1>{paras * 2}</article></body></html>"


def rss(items: list[tuple[str, str, datetime]]) -> str:
    entries = "".join(
        f"<item><title>{t}</title><link>{link}</link><pubDate>{format_datetime(d)}</pubDate>"
        f"<description>Summary of {t} with enough words to be useful as a fallback when scraping fails, "
        f"including several details that a reader might care about.</description></item>"
        for t, link, d in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{entries}</channel></rss>'


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.requests: list[tuple[str, str | None]] = []

    def allowed(self, url: str) -> bool:
        return True

    def get(self, url, etag=None, last_modified=None):
        self.requests.append((url, etag))
        if url.endswith(".xml") and etag == "v1":
            return FetchResult(304, None, etag, last_modified, url)
        if url not in self.pages:
            import requests
            raise requests.HTTPError("404")
        return FetchResult(200, self.pages[url], "v1" if url.endswith(".xml") else None, None, url)


@pytest.fixture
def world(tmp_path):
    settings = Settings(data_dir=tmp_path, embedding_model="hash", google_api_key=None,
                        story_similarity=0.6, retention_days=30)
    feeds = [
        Feed("https://a.test/india.xml", "Outlet A", "india", "india"),
        Feed("https://b.test/news.xml", "Outlet B", "india", "india"),
        Feed("https://c.test/sport.xml", "Outlet C", "sports", "india"),
    ]
    pages = {
        "https://a.test/india.xml": rss([
            ("Chennai floods: schools shut", "https://a.test/india/chennai-floods-123?utm_source=rss", NOW - timedelta(hours=3)),
            ("RBI holds repo rate", "https://a.test/business/rbi-policy-456", NOW - timedelta(days=2)),
            ("India section", "https://a.test/india/", NOW),
            ("Very old story", "https://a.test/india/old-story-1", NOW - timedelta(days=90)),
        ]),
        "https://b.test/news.xml": rss([
            ("Rain batters Chennai", "https://b.test/news/rain-chennai-789", NOW - timedelta(hours=2)),
            ("RBI policy: rates unchanged (wire)", "https://b.test/news/rbi-wire-111", NOW - timedelta(days=2)),
            ("Paywalled scoop", "https://b.test/news/paywalled-222", NOW - timedelta(hours=5)),
        ]),
        "https://c.test/sport.xml": rss([("India level series in Adelaide", "https://c.test/cricket/adelaide-333", NOW - timedelta(hours=10))]),
        "https://a.test/india/chennai-floods-123?utm_source=rss": page(FLOOD),
        "https://a.test/business/rbi-policy-456": page(RBI),
        "https://b.test/news/rain-chennai-789": page(FLOOD_B),
        "https://b.test/news/rbi-wire-111": page(RBI.upper()),  # same wire copy, different formatting
        "https://c.test/cricket/adelaide-333": page(CRICKET),
        # paywalled-222 is missing -> 404 -> falls back to the RSS summary
    }
    store = Store(settings.sqlite_path)
    index = VectorIndex(settings.chroma_path, "hash")
    fetcher = FakeFetcher(pages)
    ingestor = Ingestor(settings, store, index, HashEmbedder(), fetcher)
    return settings, feeds, store, index, fetcher, ingestor


def test_ingest_dedupes_and_clusters(world):
    settings, feeds, store, index, fetcher, ingestor = world
    report = ingestor.run(feeds)

    assert report.feeds_checked == 3 and report.feeds_failed == 0
    assert report.added == 5                # 2 flood + rbi + cricket + paywalled (RSS summary)
    assert report.added_summary_only == 1
    assert report.duplicate_content == 1    # RBI wire copy
    urls = [u for u, _ in fetcher.requests]
    assert "https://a.test/india/" not in urls           # section page skipped
    assert "https://a.test/india/old-story-1" not in urls  # older than retention

    stats = store.stats()
    assert stats["multi_outlet_stories"] == 1  # the two Chennai reports were grouped
    flood_stories = {r["story_id"] for r in store.iter_articles() if "Chennai" in r["title"]}
    assert len(flood_stories) == 1


def test_summary_fallback_when_page_unavailable(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    rows = {r["title"]: r for r in store.iter_articles()}
    assert rows["Paywalled scoop"]["full_text"] == 0
    assert rows["Paywalled scoop"]["text"].startswith("Summary of Paywalled scoop")
    assert all(r["full_text"] == 1 for t, r in rows.items() if t != "Paywalled scoop")


def test_second_run_is_incremental(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    fetcher.requests.clear()
    report = ingestor.run(feeds)
    assert report.added == 0
    assert report.feeds_unchanged == 3  # conditional GET returned 304
    assert all(u.endswith(".xml") for u, _ in fetcher.requests)


def test_prune_removes_from_both_stores(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    before = store.stats()["articles"]
    removed = ingestor.prune(NOW - timedelta(days=1))  # drops the 2-day-old RBI article
    assert removed == 1 and store.stats()["articles"] == before - 1
    assert index.chunks.count() == store.stats()["chunks"]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, system, prompt):
        self.prompts.append(prompt)
        return "Schools were shut in Chennai after flooding [1][2]."

    def stream(self, system, prompt):
        yield from self.generate(system, prompt).split(" ")


def test_search_and_ask(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    llm = FakeLLM()
    assistant = NewsAssistant(settings, store, index, HashEmbedder(), llm)

    for mode in ("bm25", "dense", "hybrid"):
        res = assistant.retriever.search("repo rate decision Reserve Bank", mode=mode)
        assert res.hits and "RBI" in res.hits[0].title, mode

    res = assistant.retriever.search("cricket Adelaide Bumrah", categories=["sports"])
    assert {h.category for h in res.hits} == {"sports"}

    # "today" filters out the 2-day-old RBI story
    res = assistant.retriever.search("repo rate today")
    assert res.window.label == "today" and all("RBI" not in h.title for h in res.hits)

    answer = assistant.ask("What happened with the Chennai floods?")
    assert answer.sources and "[1]" in answer.text
    assert "Today is" in llm.prompts[-1] and "Chennai" in llm.prompts[-1]
    assert answer.to_dict()["sources"][0]["url"].startswith("https://")


def test_briefing_ranks_multi_outlet_story_first(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    assistant = NewsAssistant(settings, store, index, HashEmbedder(), FakeLLM())
    brief = assistant.briefing(hours=24)
    assert brief["stories"][0]["outlets"] == 2
    assert "Chennai" in brief["stories"][0]["articles"][0]["title"] or "Rain" in brief["stories"][0]["articles"][0]["title"]


def test_reindex_rebuilds_vectors(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    n_chunks = index.chunks.count()
    assert ingestor.reindex() == store.stats()["articles"]
    assert index.chunks.count() == n_chunks


def test_rrf_fusion_still_works(world):
    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    assistant = NewsAssistant(settings, store, index, HashEmbedder(), None)
    res = assistant.retriever.search("Reserve Bank repo rate monetary policy", mode="hybrid", fusion="rrf", use_recency=False)
    assert "RBI" in res.hits[0].title


def test_eval_runs_on_headlines(world):
    from newsrag.evaluate import evaluate

    settings, feeds, store, index, fetcher, ingestor = world
    ingestor.run(feeds)
    assistant = NewsAssistant(settings, store, index, HashEmbedder(), None)
    report = evaluate(assistant, n=10, use_llm=False)
    assert report["cases"] == 4 and set(report["results"]) == {"bm25", "dense", "hybrid-rrf", "hybrid-convex"}
    assert report["results"]["hybrid-convex"]["recall@10"] == 1.0
