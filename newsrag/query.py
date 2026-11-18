"""Understanding what kind of question was asked.

Two kinds of question need different retrieval:

* Topical ("What did the RBI decide?", "latest on the Nvidia CEO saga"):
  find passages that match the topic. Handled by `retrieve.Retriever`.
* Overview ("What are the top stories in India today?", "tech headlines",
  "catch me up"): no topic to match, the user wants the most important
  stories. Matching the words "top", "stories" and "today" finds things like
  "Top stocks to watch today", so these are answered from story rankings
  (how many outlets covered each story, and how recently) instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import _STOPWORDS


@dataclass(frozen=True)
class Scope:
    """Which part of the news a question is about."""

    categories: tuple[str, ...] = ()
    # Also include general "top stories" feeds from outlets in this region.
    # "India" means India-section feeds from any outlet plus Indian outlets' front pages.
    top_region: str | None = None
    label: str | None = None


# Phrase -> (categories, top_region, label). First match wins.
_SCOPES: list[tuple[str, Scope]] = [
    (r"\b(india|indian|national|nation|desh)\b", Scope(("india",), "india", "India")),
    (r"\b(world|international|global|foreign|abroad)\b", Scope(("world",), None, "World")),
    (r"\b(business|economy|economic|markets?|stocks?|finance|financial|sensex|nifty)\b", Scope(("business",), None, "Business")),
    (r"\b(tech|technology|gadgets?|startups?|software|internet)\b", Scope(("tech",), None, "Tech")),
    (r"\b(science|space|climate|environment|research)\b", Scope(("science",), None, "Science")),
    (r"\b(sports?|cricket|football|soccer|ipl|tennis|hockey|olympics?)\b", Scope(("sports",), None, "Sports")),
]

_OVERVIEW = re.compile(
    r"\b(top|biggest|big|major|main|important|key|latest|recent|breaking|trending)\s+([a-z]+\s+)?"
    r"(news|stories|story|headlines|developments|events|updates)\b"
    r"|\bheadlines?\b"
    r"|\bwhat'?s\s+(happening|going on|new)\b|\bwhat is (happening|going on)\b"
    r"|\bcatch me up\b|\bkeep me posted\b"
    r"|\b(today'?s|this week'?s|daily) (news|stories|update|briefing)\b"
    r"|\bnews (today|this week|now)\b|\bany news\b"
    r"|\bbriefing\b|\bround-?up\b|\bsummar(y|ise|ize) (of )?(the )?news\b",
    re.I,
)

# Words that can appear in an overview question without making it topical.
_FILLER = frozenset(
    """top biggest big major main important key latest recent breaking trending news stories story headlines
    headline developments events updates happening going on whats what's catch me up keep posted today todays
    tonight morning evening this week weeks month daily today's week's briefing roundup round brief summary summarise summarize
    any give tell show list please could would you can the in on of for from about across around right now so far
    s around hours hour past last day days""".split()
)


@dataclass(frozen=True)
class QueryIntent:
    overview: bool
    scope: Scope


def parse_scope(question: str) -> Scope:
    q = question.lower()
    for pattern, scope in _SCOPES:
        if re.search(pattern, q):
            return scope
    return Scope()


def parse_intent(question: str) -> QueryIntent:
    """Classify a question as overview or topical, and read its scope."""
    q = question.lower()
    scope = parse_scope(q)
    if not _OVERVIEW.search(q):
        return QueryIntent(False, scope)
    # "latest news on the India-US trade deal" mentions "latest news" but has a
    # topic. Treat it as an overview only if nothing topical is left over.
    leftover = q
    for pattern, _ in _SCOPES:
        leftover = re.sub(pattern, " ", leftover)
    terms = [t for t in re.findall(r"[a-z0-9']+", leftover) if t not in _FILLER and t not in _STOPWORDS and not t.isdigit()]
    return QueryIntent(not terms, scope)


ROUTINE = re.compile(
    r"\b(petrol|diesel|fuel|cng|lpg)\b.{0,30}\b(price|prices|rate|rates)\b"
    r"|\bgold\b.{0,20}\b(silver\b.{0,10})?(price|prices|rate|rates)\b"
    r"|\b(stocks?|shares?)\s+to\s+(watch|buy|sell)\b|\b(top|best)\s+\d+\s+(stocks?|shares?)\b|\bstock picks?\b"
    r"|\bhoroscope\b|\brashifal\b|\btarot\b|\bzodiac\b|\blottery\b|\blucky (number|colou?r)\b"
    r"|\bhow to (register|apply|check|download)\b|\bstep[- ]by[- ]step guide\b"
    r"|\bweather (today|update|forecast)\b|\bcrossword\b|\bwordle\b|\bquiz\b|\brecipe\b"
    r"|\bbank holidays?\b|\bmarket holidays?\b",
    re.I,
)


def is_routine(title: str) -> bool:
    """Daily service pieces (fuel prices, stocks to watch, horoscopes...)
    that are rarely what someone means by "the news"."""
    return bool(ROUTINE.search(title))


def asks_for_routine(question: str) -> bool:
    """True if the question itself is about prices, stock picks, etc."""
    return bool(ROUTINE.search(question)) or bool(
        re.search(r"\b(price|prices|rate|rates|stocks? to|shares? to|horoscope|register|weather|lottery|quiz)\b", question, re.I)
    )
