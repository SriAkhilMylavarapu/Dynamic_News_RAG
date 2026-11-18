import json

from newsrag.extract import clean_summary, extract_article
from newsrag.ingest import looks_like_article

BODY = " ".join(f"Paragraph sentence {i} about the monsoon session of Parliament." for i in range(20))


def test_json_ld_article_body_is_used():
    ld = json.dumps({"@context": "https://schema.org", "@graph": [{"@type": "NewsArticle", "articleBody": BODY}]})
    page = f'<html><head><script type="application/ld+json">{ld}</script></head><body><p>short</p></body></html>'
    assert extract_article(page) == BODY


def test_falls_back_to_html_extraction():
    paras = "".join(f"<p>{BODY[i:i+200]}</p>" for i in range(0, len(BODY), 200))
    page = f"<html><body><nav>Home | World | Sports</nav><article><h1>T</h1>{paras}</article><footer>(c) 2026</footer></body></html>"
    text = extract_article(page)
    assert text and "monsoon session" in text and "Home | World" not in text


def test_returns_none_for_non_articles():
    assert extract_article("<html><body><p>Subscribe now</p></body></html>") is None


def test_clean_summary_strips_html():
    assert clean_summary('<p>Hello <a href="x">world</a></p><img src="y"/>') == "Hello world"


def test_section_pages_are_not_articles():
    assert not looks_like_article("https://timesofindia.indiatimes.com/india/")
    assert looks_like_article("https://timesofindia.indiatimes.com/india/some-story/articleshow/120794248.cms")
    assert looks_like_article("https://www.bbc.co.uk/news/articles/c0lz3q1p2x4o")
