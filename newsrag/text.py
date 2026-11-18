"""Pure text helpers: URL canonicalisation, hashing, dates, chunking, query parsing."""

from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that only track where a click came from.
_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid$|gclid$|mc_cid$|mc_eid$|ref$|ref_src$|cmpid$|ito$|at_medium$|at_campaign$|from$|src$)")


def canonical_url(url: str) -> str:
    """Normalise a URL so the same article linked two ways hashes the same.

    Lowercases scheme and host, drops `www.`, fragments, tracking parameters,
    trailing slashes and AMP variants.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("amp."):
        host = host[4:]
    path = re.sub(r"/amp/?$", "", parts.path)
    path = re.sub(r"\.amp(?=\.|$)", "", path)
    path = path.rstrip("/") or "/"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k.lower())))
    return urlunsplit(("https", host, path, query, ""))


def url_id(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:24]


def normalize_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Hash of the body with case, punctuation and whitespace removed.

    Wire copy (PTI, Reuters, AP) is republished by many outlets with small
    formatting differences; this catches those exact-content duplicates.
    """
    squashed = re.sub(r"[^a-z0-9]+", "", text.lower())
    return hashlib.sha256(squashed.encode()).hexdigest()[:24]


def entry_datetime(entry: dict) -> datetime | None:
    """Best publish time for a feedparser entry, in UTC."""
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        struct = entry.get(key)
        if struct:
            # feedparser normalises these structs to UTC, so use timegm, not mktime.
            return datetime.fromtimestamp(calendar.timegm(struct), tz=timezone.utc)
    for key in ("published", "updated", "pubDate"):
        raw = entry.get(key)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                try:
                    dt = datetime.fromisoformat(raw)
                except ValueError:
                    continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    return None


_SENTENCE_END = re.compile(r"(?<=[.!?\"”’])\s+(?=[A-Z0-9\"“‘(])")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            sentences.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())
    return sentences


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into chunks of about `size` characters on sentence boundaries.

    Consecutive chunks share roughly `overlap` characters of trailing sentences
    so a fact that straddles a boundary is still retrievable.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        # A single sentence longer than the chunk size is hard-wrapped.
        while len(sentence) > size:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            chunks.append(sentence[:size])
            sentence = sentence[size - overlap :]
        if length + len(sentence) + 1 > size and current:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        last = " ".join(current)
        if not chunks or last not in chunks[-1]:
            chunks.append(last)
    return chunks


@dataclass(frozen=True)
class TimeWindow:
    since: datetime | None
    label: str | None
    wants_recent: bool


_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10, "a": 1}
_RECENT_WORDS = re.compile(r"\b(latest|recent|recently|newest|now|current|currently|breaking|update|updates|so far)\b", re.I)


def parse_time_window(query: str, now: datetime | None = None) -> TimeWindow:
    """Read a time filter out of a natural-language question.

    "today", "yesterday", "this week", "past 3 days", "last month"... become a
    lower bound on publish time. Words like "latest" don't filter but ask the
    retriever to weight recency more heavily.
    """
    now = now or datetime.now(timezone.utc)
    q = query.lower()
    wants_recent = bool(_RECENT_WORDS.search(q))

    m = re.search(r"\b(?:past|last|previous)\s+(\d+|one|two|three|four|five|six|seven|ten|a)\s+(hour|day|week|month)s?\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUMBERS[m.group(1)]
        unit = {"hour": timedelta(hours=1), "day": timedelta(days=1), "week": timedelta(weeks=1), "month": timedelta(days=30)}[m.group(2)]
        return TimeWindow(now - n * unit, m.group(0), True)

    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rules: list[tuple[str, datetime]] = [
        (r"\b(today|tonight|this morning|this evening)\b", now - timedelta(hours=24)),
        (r"\byesterday\b", start_of_today - timedelta(days=1)),
        (r"\b(this|past|last) week\b|\bweekly\b", now - timedelta(days=7)),
        (r"\b(this|past|last) month\b|\bmonthly\b", now - timedelta(days=30)),
        (r"\b(past|last) (24|twenty[- ]four) hours\b", now - timedelta(hours=24)),
    ]
    for pattern, since in rules:
        m = re.search(pattern, q)
        if m:
            return TimeWindow(since, m.group(0), True)
    return TimeWindow(None, None, wants_recent)


def fts_query(text: str) -> str:
    """Turn free text into a safe SQLite FTS5 MATCH expression (OR of terms)."""
    terms = re.findall(r"[A-Za-z0-9]+", text.lower())
    terms = [t for t in terms if t not in _STOPWORDS and len(t) > 1]
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return " OR ".join(f'"{t}"' for t in seen)


_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with about from into over after before between
    is are was were be been being am do does did doing have has had having it its this that these those
    what which who whom whose when where why how any all some more most other such no not only own same
    so than too very can will just should would could may might must shall i me my we our you your he him
    his she her they them their there here as up down out off again further then once also latest recent
    recently news update updates tell give show summarize summarise happened happening report reported
    today yesterday week month day days past last new""".split()
)
