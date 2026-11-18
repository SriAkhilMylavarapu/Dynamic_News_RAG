import pytest

from newsrag.ingest import looks_like_article
from newsrag.query import asks_for_routine, is_routine, parse_intent

OVERVIEW = [
    ("What are the top stories in India today?", "India"),
    ("top news", None),
    ("headlines", None),
    ("What's happening in tech?", "Tech"),
    ("catch me up on world news", "World"),
    ("biggest business stories this week", "Business"),
    ("Any news in cricket today?", "Sports"),
    ("today's news", None),
]

TOPICAL = [
    "latest news on the India-US trade deal",
    "What did the RBI decide this week?",
    "What happened in Chennai?",
    "Top stories about Nvidia's CEO",
    "who won the match in Adelaide",
]


@pytest.mark.parametrize("question,label", OVERVIEW)
def test_overview_questions(question, label):
    intent = parse_intent(question)
    assert intent.overview, question
    assert intent.scope.label == label


@pytest.mark.parametrize("question", TOPICAL)
def test_topical_questions(question):
    assert not parse_intent(question).overview, question


def test_india_scope_includes_indian_front_pages():
    scope = parse_intent("top stories in India").scope
    assert scope.categories == ("india",) and scope.top_region == "india"


@pytest.mark.parametrize("title", [
    "Petrol and diesel prices today, November 18: How much does fuel cost in Delhi, Mumbai, Bengaluru?",
    "Top stocks to watch today: Ashoka Buildcon, IIFL Finance, Prestige Estates, HCLTech and 5 others in focus",
    "Reliance, UltraTech, Hyundai Motor shares: Top 3 stocks to buy; check ratings, target price",
    "Gold rate today: yellow metal rises in Chennai",
    "Bhangwant Mann's Punjab Da Crorepati quiz competition: Step-by-step guide on how to register for the quiz",
    "Horoscope today, November 18, 2026",
])
def test_routine_titles(title):
    assert is_routine(title)


@pytest.mark.parametrize("title", [
    "'Lateef killed': Forces celebrate as Kashmir's last known active terrorist falls",
    "Delhi court orders cops to file action taken report on complaint against Sonia Gandhi",
    "RBI holds repo rate steady, keeps neutral stance",
    "Nvidia beats Q3 expectations in first report post-Jensen Huang",
])
def test_news_titles_are_not_routine(title):
    assert not is_routine(title)


def test_questions_about_prices_keep_routine_items():
    assert asks_for_routine("what are petrol prices in Delhi today")
    assert asks_for_routine("which stocks to buy this week")
    assert not asks_for_routine("what are the top stories in India today")


def test_media_pages_are_skipped():
    assert not looks_like_article("https://www.bbc.co.uk/news/videos/ckjwjl6ywgn9o?at_medium=RSS")
    assert not looks_like_article("https://timesofindia.indiatimes.com/videos/news/some-clip/videoshow/1234.cms")
    assert not looks_like_article("https://indianexpress.com/photos/india-news/some-gallery-123/")
    assert looks_like_article("https://www.bbc.co.uk/news/articles/c0lz3q1p2x4o")
