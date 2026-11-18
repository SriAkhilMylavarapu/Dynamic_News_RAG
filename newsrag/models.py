"""Plain data objects passed between the pipeline stages.

They are deliberately JSON-friendly (see `to_dict`) so a future HTTP API can
return them unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class Article:
    id: str
    url: str
    title: str
    source: str
    category: str
    region: str
    feed_url: str
    published_at: datetime
    fetched_at: datetime
    text: str
    full_text: bool  # False when only the RSS summary could be used
    content_hash: str
    story_id: str | None = None


@dataclass
class Hit:
    """One retrieved chunk plus the article it came from."""

    chunk_id: str
    article_id: str
    text: str
    title: str
    source: str
    category: str
    url: str
    published_at: datetime
    story_id: str | None
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        return d


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]
    time_window: str | None
    timings_ms: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "time_window": self.time_window,
            "timings_ms": self.timings_ms,
            "sources": [s.to_dict() for s in self.sources],
        }


@dataclass
class IngestReport:
    feeds_checked: int = 0
    feeds_unchanged: int = 0
    feeds_failed: int = 0
    entries_seen: int = 0
    already_known: int = 0
    duplicate_content: int = 0
    added: int = 0
    added_summary_only: int = 0
    joined_existing_story: int = 0
    chunks_added: int = 0
    pruned: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
