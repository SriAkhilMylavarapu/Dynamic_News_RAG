"""Offline retrieval evaluation.

For a sample of stored articles we build one question each, then check whether
each retrieval mode (BM25 only, dense only, hybrid) brings that article back.

Questions come from the LLM (one question per article, written from a passage
in the middle of the article so it doesn't just echo the headline) and are
cached in data/eval_questions.json so runs are comparable over time. Without an
API key, headlines are used as queries instead; that's a much easier test, so
the report labels it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .assistant import NewsAssistant

QUESTION_SYSTEM = """Write ONE question a news reader might ask that this passage answers.
Be specific (names, places, numbers) but paraphrase; do not copy long phrases. Do not mention
"the article" or "the passage". Reply with the question only."""

MODES = (("bm25", "convex"), ("dense", "convex"), ("hybrid", "rrf"), ("hybrid", "convex"))
KS = (1, 3, 5, 10)


@dataclass
class EvalCase:
    article_id: str
    question: str


def build_cases(assistant: NewsAssistant, n: int, use_llm: bool, seed: int = 7) -> list[EvalCase]:
    cache_path = assistant.settings.data_dir / "eval_questions.json"
    cache: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    rows = [r for r in assistant.store.iter_articles() if r["full_text"]]
    random.Random(seed).shuffle(rows)
    cases: list[EvalCase] = []
    for row in rows:
        if len(cases) >= n:
            break
        if not use_llm:
            cases.append(EvalCase(row["id"], row["title"]))
            continue
        if row["id"] not in cache:
            chunks = assistant.store.chunks_for_article(row["id"])
            passage = chunks[len(chunks) // 2]["text"]
            cache[row["id"]] = assistant.llm.generate(QUESTION_SYSTEM, passage).strip()
        cases.append(EvalCase(row["id"], cache[row["id"]]))
    if use_llm:
        cache_path.write_text(json.dumps(cache, indent=2))
    return cases


def evaluate(assistant: NewsAssistant, n: int = 50, use_llm: bool | None = None) -> dict:
    use_llm = bool(assistant.llm) if use_llm is None else use_llm
    cases = build_cases(assistant, n, use_llm)
    results: dict[str, dict[str, float]] = {}
    for mode, fusion in MODES:
        hits_at = {k: 0 for k in KS}
        rr_total = 0.0
        for case in cases:
            res = assistant.retriever.search(
                case.question, k=max(KS), mode=mode, fusion=fusion, use_recency=False, use_time_parsing=False,
                max_per_article=1, max_per_story=10,
            )
            ranked = [h.article_id for h in res.hits]
            rank = ranked.index(case.article_id) + 1 if case.article_id in ranked else None
            for k in KS:
                hits_at[k] += int(rank is not None and rank <= k)
            rr_total += 1.0 / rank if rank else 0.0
        total = max(len(cases), 1)
        name = mode if mode != "hybrid" else f"hybrid-{fusion}"
        results[name] = {f"recall@{k}": round(v / total, 3) for k, v in hits_at.items()} | {"mrr@10": round(rr_total / total, 3)}
    return {"questions": "llm" if use_llm else "headline", "cases": len(cases), "results": results}
