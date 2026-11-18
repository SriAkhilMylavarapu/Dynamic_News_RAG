import time
from datetime import datetime, timezone

from newsrag.text import (canonical_url, chunk_text, content_hash, entry_datetime, fts_query,
                          parse_time_window, url_id)

NOW = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)


def test_canonical_url_strips_tracking_and_variants():
    a = "https://www.example.com/news/story-123/?utm_source=rss&id=5#top"
    b = "http://example.com/news/story-123?id=5"
    c = "https://amp.example.com/news/story-123/amp/?id=5&fbclid=xyz"
    assert canonical_url(a) == canonical_url(b) == canonical_url(c) == "https://example.com/news/story-123?id=5"
    assert url_id(a) == url_id(c)


def test_content_hash_ignores_formatting():
    assert content_hash("Hello,  World!\n\nPTI") == content_hash("hello world pti")
    assert content_hash("hello world") != content_hash("hello there world")


def test_entry_datetime_uses_utc_struct():
    struct = time.strptime("2026-11-18 09:30:00", "%Y-%m-%d %H:%M:%S")
    assert entry_datetime({"published_parsed": struct}) == datetime(2026, 11, 18, 9, 30, tzinfo=timezone.utc)


def test_entry_datetime_falls_back_to_rfc822_string():
    got = entry_datetime({"published": "Wed, 18 Nov 2026 10:00:00 +0530"})
    assert got == datetime(2026, 11, 18, 4, 30, tzinfo=timezone.utc)
    assert entry_datetime({}) is None


def test_chunking_respects_size_and_overlaps():
    sentences = [f"Sentence number {i} says something about the economy." for i in range(60)]
    chunks = chunk_text(" ".join(sentences), size=300, overlap=80)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share at least one sentence
    for a, b in zip(chunks, chunks[1:]):
        assert a.split(". ")[-1].rstrip(".") in b
    # every sentence survives
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_chunking_hard_wraps_giant_sentence():
    chunks = chunk_text("x" * 2000, size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) >= 4


def test_time_window_parsing():
    assert parse_time_window("what happened today in delhi", NOW).since == datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    assert parse_time_window("cricket news yesterday", NOW).since == datetime(2026, 11, 17, tzinfo=timezone.utc)
    assert parse_time_window("markets over the past 3 days", NOW).since == datetime(2026, 11, 15, 15, tzinfo=timezone.utc)
    assert parse_time_window("AI news this week", NOW).label == "this week"
    w = parse_time_window("latest on the India-US trade deal", NOW)
    assert w.since is None and w.wants_recent
    w = parse_time_window("who is the RBI governor", NOW)
    assert w.since is None and not w.wants_recent


def test_fts_query_is_safe():
    q = fts_query('What\'s the "latest" on AND OR NOT India-US trade? (today)')
    assert q == '"india" OR "us" OR "trade"'
    assert fts_query("the of and") == ""
