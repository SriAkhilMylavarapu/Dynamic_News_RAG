"""Command-line interface: `newsrag <command>` or `python -m newsrag <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from .assistant import NewsAssistant, Turn, cited_indices
from .config import Settings, load_feeds
from .models import Hit


def _filters(args) -> dict:
    f = {}
    if getattr(args, "days", None):
        f["since"] = datetime.now(timezone.utc) - timedelta(days=args.days)
    if getattr(args, "category", None):
        f["categories"] = args.category
    if getattr(args, "source", None):
        f["sources"] = args.source
    if getattr(args, "k", None):
        f["k"] = args.k
    return f


def _ago(dt: datetime) -> str:
    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    if hours < 1:
        return f"{int(hours * 60)}m ago"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


def _print_sources(hits: list[Hit], answer_text: str | None = None) -> None:
    cited = cited_indices(answer_text, len(hits)) if answer_text else []
    show = cited or list(range(1, len(hits) + 1))
    print("\nSources:")
    for i in show:
        h = hits[i - 1]
        print(f"  [{i}] {h.title}\n      {h.source} · {_ago(h.published_at)} · {h.url}")


def cmd_ingest(args) -> None:
    assistant = NewsAssistant.from_settings(need_llm=False)
    feeds = load_feeds(assistant.settings.feeds_file)
    while True:
        print(f"Polling {len(feeds)} feeds from {len({f.source for f in feeds})} outlets...")
        report = assistant.ingestor().run(feeds)
        summary = {k: v for k, v in asdict(report).items() if k != "errors"}
        print(json.dumps(summary, indent=2))
        for err in report.errors[:10]:
            print(f"  ! {err}")
        if len(report.errors) > 10:
            print(f"  ! ...and {len(report.errors) - 10} more errors")
        if not args.every:
            break
        print(f"Next run in {args.every} minutes (Ctrl+C to stop).")
        time.sleep(args.every * 60)


def cmd_ask(args) -> None:
    assistant = NewsAssistant.from_settings()
    question = " ".join(args.question)
    answer = assistant.ask(question, **_filters(args))
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return
    if answer.time_window:
        print(f"({answer.time_window})")
    print("\n" + answer.text)
    _print_sources(answer.sources, answer.text if assistant.llm else None)
    if args.verbose:
        print(f"\nTimings (ms): {answer.timings_ms}")


def cmd_chat(args) -> None:
    assistant = NewsAssistant.from_settings()
    if not assistant.llm:
        sys.exit("Chat needs ANTHROPIC_API_KEY in your environment or .env file.")
    history: list[Turn] = []
    print("Ask about the news. Commands: /new to reset context, /quit to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "quit", "exit"):
            break
        if q == "/new":
            history.clear()
            print("(context cleared)")
            continue
        result, stream = assistant.ask_stream(q, history, **_filters(args))
        print()
        parts = []
        for piece in stream:
            parts.append(piece)
            print(piece, end="", flush=True)
        text = "".join(parts)
        print()
        _print_sources(result.hits, text)
        history.append(Turn(q, text))


def cmd_search(args) -> None:
    assistant = NewsAssistant.from_settings(need_llm=False)
    result = assistant.retriever.search(" ".join(args.query), mode=args.mode, fusion=args.fusion, **_filters(args))
    if result.window.label:
        print(f"(time filter: {result.window.label}{', relaxed - nothing in range' if result.relaxed else ''})")
    for i, h in enumerate(result.hits, 1):
        sig = " ".join(f"{k}={v:.3f}" for k, v in h.signals.items())
        print(f"\n[{i}] {h.title}  ({h.source}, {_ago(h.published_at)})\n    score={h.score:.4f} {sig}")
        print(textwrap.indent(textwrap.shorten(h.text, 300), "    "))


def cmd_briefing(args) -> None:
    assistant = NewsAssistant.from_settings()
    brief = assistant.briefing(hours=args.hours, categories=args.category, stories=args.stories)
    if args.json:
        print(json.dumps(brief, indent=2))
        return
    scope = f" · {', '.join(args.category)}" if args.category else ""
    print(f"# Briefing: last {args.hours}h{scope}\n")
    print(brief["text"])
    if brief["sources"]:
        print("\nSources:")
        for i, s in enumerate(brief["sources"], 1):
            print(f"  [{i}] {s['title']} ({s['source']}) {s['url']}")


def cmd_stats(args) -> None:
    store = NewsAssistant.from_settings(need_llm=False).store
    print(json.dumps(store.stats(), indent=2))


def cmd_feeds(args) -> None:
    settings = Settings()
    assistant = NewsAssistant.from_settings(settings, need_llm=False)
    for feed in load_feeds(settings.feeds_file):
        state = assistant.store.feed_state(feed.url)
        status = state["last_status"] if state else "never polled"
        print(f"{feed.source:<22} {feed.category:<9} {status:<28} {feed.url}")


def cmd_prune(args) -> None:
    assistant = NewsAssistant.from_settings(need_llm=False)
    days = args.days or assistant.settings.retention_days
    removed = assistant.ingestor().prune(datetime.now(timezone.utc) - timedelta(days=days))
    print(f"Removed {removed} articles older than {days} days.")


def cmd_reindex(args) -> None:
    assistant = NewsAssistant.from_settings(need_llm=False)
    print(f"Re-embedded {assistant.ingestor().reindex()} articles with {assistant.settings.embedding_model}.")


def cmd_eval(args) -> None:
    from .evaluate import evaluate

    assistant = NewsAssistant.from_settings(need_llm=not args.headlines)
    report = evaluate(assistant, n=args.n, use_llm=False if args.headlines else None)
    print(f"{report['cases']} questions ({report['questions']}-generated)\n")
    cols = list(next(iter(report["results"].values())))
    print(f"{'mode':<15}" + "".join(f"{c:>11}" for c in cols))
    for mode, row in report["results"].items():
        print(f"{mode:<15}" + "".join(f"{row[c]:>11.3f}" for c in cols))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="show logs and timings")
    p = argparse.ArgumentParser(prog="newsrag", description="Ask questions about today's news, with citations.")
    sub = p.add_subparsers(dest="command", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *a, **kw: _add(*a, parents=[common], **kw)  # every command accepts -v

    def add_filters(sp):
        sp.add_argument("--days", type=float, help="only articles from the last N days")
        sp.add_argument("--category", action="append", help="e.g. india, world, business, tech, sports (repeatable)")
        sp.add_argument("--source", action="append", help='outlet name, e.g. "BBC News" (repeatable)')
        sp.add_argument("-k", type=int, help="number of passages to retrieve")

    sp = sub.add_parser("ingest", help="poll feeds and index new articles")
    sp.add_argument("--every", type=float, metavar="MINUTES", help="keep running, polling every N minutes")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("ask", help="answer one question")
    sp.add_argument("question", nargs="+")
    sp.add_argument("--json", action="store_true")
    add_filters(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("chat", help="interactive chat with follow-up questions")
    add_filters(sp)
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("search", help="show retrieved passages and their scores (no LLM)")
    sp.add_argument("query", nargs="+")
    sp.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    sp.add_argument("--fusion", choices=["convex", "rrf"], default="convex")
    add_filters(sp)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("briefing", help="summary of the most widely covered stories")
    sp.add_argument("--hours", type=int, default=24)
    sp.add_argument("--category", action="append")
    sp.add_argument("--stories", type=int, default=8)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_briefing)

    sub.add_parser("stats", help="database statistics").set_defaults(func=cmd_stats)
    sub.add_parser("feeds", help="list feeds and their last poll status").set_defaults(func=cmd_feeds)

    sp = sub.add_parser("prune", help="delete old articles")
    sp.add_argument("--days", type=int)
    sp.set_defaults(func=cmd_prune)

    sub.add_parser("reindex", help="rebuild the vector index from the database").set_defaults(func=cmd_reindex)

    sp = sub.add_parser("eval", help="measure retrieval quality (recall@k, MRR)")
    sp.add_argument("-n", type=int, default=50, help="number of test questions")
    sp.add_argument("--headlines", action="store_true", help="use headlines as queries instead of LLM questions")
    sp.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
