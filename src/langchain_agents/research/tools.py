"""The two tools, in two modes with the same names so prompts don't change:

    search(query)            web: OpenRouter web search   corpus: BM25 over fixtures/research/corpus
    read_source(source_id)   web: httpx fetch of the URL  corpus: read the document

read_source is the only way text gets into state["sources"], and state["sources"]
is the only text the verifier accepts quotes from. Budgets (searches, sources per
run) are enforced here in code, from state, so a looping model can't run up a bill.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from .corpus import Corpus
from .fetch import Fetcher, normalize_url
from .websearch import OpenRouterWebSearch, WebSearchError

MAX_CHARS_TO_MODEL = 15_000   # what the reader sees per source
MAX_CHARS_STORED = 300_000    # what the verifier can match against


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reply(update: dict, content: str, tool_call_id: str) -> Command:
    return Command(update={**update, "messages": [ToolMessage(content, tool_call_id=tool_call_id)]})


def make_tools(mode: str, *, corpus: Corpus | None = None, searcher: OpenRouterWebSearch | None = None,
               fetcher: Fetcher | None = None, max_searches: int = 4, max_sources: int = 8) -> list[BaseTool]:
    if mode not in ("web", "corpus"):
        raise ValueError(f"mode must be 'web' or 'corpus', got {mode!r}")
    if mode == "corpus":
        corpus = corpus or Corpus()
    else:
        searcher = searcher or OpenRouterWebSearch()
        fetcher = fetcher or Fetcher()

    @tool
    def search(query: str, runtime: ToolRuntime) -> Command:
        """Search for sources. Returns source_ids (URLs or corpus ids), titles and short snippets.
        Snippets are NOT quotable: a source must be read with read_source before it can be cited."""
        tool_call_id = runtime.tool_call_id
        done = (runtime.state or {}).get("searches") or {}
        if len(done) >= max_searches:
            return _reply({}, f"Search budget used up ({max_searches} searches this run). Work with the sources "
                              "already found.", tool_call_id)
        t0 = time.monotonic()
        if mode == "corpus":
            results = corpus.search(query)  # type: ignore[union-attr]
            cost = 0.0
            payload = [{"source_id": r["source_id"], "title": r["title"], "publisher": r["publisher"],
                        "date": r["date"], "snippet": r["snippet"]} for r in results]
        else:
            try:
                res = searcher.search(query)  # type: ignore[union-attr]
            except WebSearchError as exc:
                return _reply({}, f"Search failed: {exc}", tool_call_id)
            cost = res["cost_usd"]
            payload = [{"source_id": r["url"], "title": r["title"], "snippet": r["snippet"]} for r in res["results"]]
        sid = f"s{len(done) + 1}-{uuid.uuid4().hex[:6]}"
        record = {"query": query, "mode": mode, "n_results": len(payload), "cost_usd": cost,
                  "result_ids": [p["source_id"] for p in payload], "elapsed_s": round(time.monotonic() - t0, 3)}
        body = json.dumps({"query": query, "results": payload,
                           "note": "Snippets are not quotable. Read a source before citing it."}, indent=1)
        return _reply({"searches": {sid: record}}, body if payload else f"No results for {query!r}.", tool_call_id)

    @tool
    def read_source(source_id: str, runtime: ToolRuntime) -> Command:
        """Read one source in full (fetch a URL, or open a corpus document by its source_id).
        Only text returned by this tool can be quoted in the brief."""
        tool_call_id = runtime.tool_call_id
        known = (runtime.state or {}).get("sources") or {}
        key = source_id.strip() if mode == "corpus" else normalize_url(source_id)
        if mode == "corpus" and corpus is not None:
            doc = corpus.get(source_id)
            key = doc.source_id if doc else key
        cached = known.get(key)
        if cached and cached.get("text"):
            return _reply({}, _render(key, cached), tool_call_id)
        if sum(1 for s in known.values() if s.get("text")) >= max_sources:
            return _reply({}, f"Source budget used up ({max_sources} sources this run).", tool_call_id)
        if mode == "corpus":
            if doc is None:
                return _reply({}, f"No document with source_id {source_id!r}. Use a source_id from search results.",
                              tool_call_id)
            rec = {"source_id": doc.source_id, "kind": "corpus", "title": doc.title, "text": doc.text,
                   "final_url": "", "fetched_at": _now(), "chars": len(doc.text), "truncated": False, "error": None}
        else:
            res = fetcher.fetch(source_id.strip())  # type: ignore[union-attr]
            rec = {"source_id": key, "kind": "web", "title": res.title, "text": res.text[:MAX_CHARS_STORED],
                   "final_url": res.final_url, "fetched_at": _now(), "chars": len(res.text),
                   "truncated": res.truncated or len(res.text) > MAX_CHARS_STORED, "error": res.error,
                   "status": res.status, "elapsed_s": res.elapsed_s}
            if not res.ok:
                rec["text"] = ""
                return _reply({"sources": {key: rec}}, f"Could not read {source_id}: {res.error}. Try another source.",
                              tool_call_id)
        return _reply({"sources": {key: rec}}, _render(key, rec), tool_call_id)

    return [search, read_source]


def _render(key: str, rec: dict) -> str:
    text = rec["text"]
    shown = text[:MAX_CHARS_TO_MODEL]
    tail = f"\n\n[... {len(text) - len(shown)} more characters not shown ...]" if len(text) > len(shown) else ""
    return f"SOURCE source_id={key}\nTITLE {rec.get('title') or '(none)'}\n----- TEXT -----\n{shown}{tail}"
