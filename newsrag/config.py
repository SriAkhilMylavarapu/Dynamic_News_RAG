"""Runtime settings, read from environment variables (and a .env file)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FEEDS_FILE = PACKAGE_DIR.parent / "feeds.toml"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Feed:
    url: str
    source: str
    category: str
    region: str


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("NEWSRAG_DATA_DIR", "./data")))
    feeds_file: Path = field(default_factory=lambda: Path(_env("NEWSRAG_FEEDS_FILE", str(DEFAULT_FEEDS_FILE))))

    # Embeddings (fastembed / ONNX, runs on CPU without torch)
    embedding_model: str = field(default_factory=lambda: _env("NEWSRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # LLM
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("NEWSRAG_LLM_MODEL", "gemini-2.5-flash"))
    llm_temperature: float = field(default_factory=lambda: _env_float("NEWSRAG_LLM_TEMPERATURE", 0.2))

    # Ingestion
    user_agent: str = field(
        default_factory=lambda: _env(
            "NEWSRAG_USER_AGENT",
            "DynamicNewsRAG/2.0 (+https://github.com/SriAkhilMylavarapu/Dynamic_News_RAG)",
        )
    )
    max_items_per_feed: int = field(default_factory=lambda: _env_int("NEWSRAG_MAX_ITEMS_PER_FEED", 25))
    request_timeout: float = field(default_factory=lambda: _env_float("NEWSRAG_REQUEST_TIMEOUT", 15))
    per_domain_delay: float = field(default_factory=lambda: _env_float("NEWSRAG_PER_DOMAIN_DELAY", 1.0))
    max_workers: int = field(default_factory=lambda: _env_int("NEWSRAG_MAX_WORKERS", 8))
    respect_robots: bool = field(default_factory=lambda: _env("NEWSRAG_RESPECT_ROBOTS", "1") == "1")
    retention_days: int = field(default_factory=lambda: _env_int("NEWSRAG_RETENTION_DAYS", 30))

    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("NEWSRAG_CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("NEWSRAG_CHUNK_OVERLAP", 150))

    # Story clustering: two articles from the last N hours whose title+lead
    # embeddings are at least this similar are treated as the same story.
    story_similarity: float = field(default_factory=lambda: _env_float("NEWSRAG_STORY_SIMILARITY", 0.86))
    story_window_hours: int = field(default_factory=lambda: _env_int("NEWSRAG_STORY_WINDOW_HOURS", 48))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("NEWSRAG_TOP_K", 8))
    recency_half_life_hours: float = field(default_factory=lambda: _env_float("NEWSRAG_RECENCY_HALF_LIFE_HOURS", 72))

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "news.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


def load_feeds(path: Path) -> list[Feed]:
    """Load the feed list from a TOML file (see feeds.toml)."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    feeds: list[Feed] = []
    for source in data.get("source", []):
        if not source.get("enabled", True):
            continue
        for feed in source.get("feeds", []):
            feeds.append(
                Feed(
                    url=feed["url"],
                    source=source["name"],
                    category=feed.get("category", "general"),
                    region=source.get("region", "global"),
                )
            )
    return feeds
