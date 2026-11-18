"""Pull the article body out of a news page.

Order of preference:
1. schema.org `articleBody` in JSON-LD (most news sites publish it for search engines)
2. trafilatura's boilerplate-removal extractor (works on almost any layout)
"""

from __future__ import annotations

import html
import json
import re

import trafilatura
from bs4 import BeautifulSoup

from .text import normalize_text

MIN_BODY_CHARS = 300


def _json_ld_bodies(soup: BeautifulSoup) -> list[str]:
    bodies: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            body = node.get("articleBody")
            if isinstance(body, str):
                bodies.append(body)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            walk(json.loads(tag.string or ""))
        except (json.JSONDecodeError, TypeError):
            continue
    return bodies


def extract_article(page_html: str, url: str | None = None) -> str | None:
    """Return the article's body text, or None if nothing article-like was found."""
    soup = BeautifulSoup(page_html, "html.parser")
    candidates = [html.unescape(re.sub(r"<[^>]+>", " ", b)) for b in _json_ld_bodies(soup)]

    extracted = trafilatura.extract(
        page_html, url=url, include_comments=False, include_tables=False, favor_precision=True
    )
    if extracted:
        candidates.append(extracted)

    candidates = [normalize_text(c) for c in candidates]
    candidates = [c for c in candidates if len(c) >= MIN_BODY_CHARS]
    if not candidates:
        return None
    # Pages sometimes carry a truncated articleBody; keep the longest version.
    return max(candidates, key=len)


def clean_summary(summary_html: str | None) -> str:
    """Strip HTML from an RSS <description>, used when the page can't be scraped."""
    if not summary_html:
        return ""
    text = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
    return normalize_text(text)
