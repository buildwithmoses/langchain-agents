"""Live web search through OpenRouter's `openrouter:web_search` server tool.

Verified against https://openrouter.ai/docs/guides/features/server-tools/web-search.md
(2026-10-28) and a live probe: the plugin (`plugins: [{"id": "web"}]`) and the
`:online` suffix are deprecated; the server tool goes in `tools`, OpenRouter runs
the search, and results come back on the assistant message as
`annotations: [{"type": "url_citation", "url_citation": {url, title, content}}]`.
`usage.cost` includes the search fee (Exa: $0.007/request), and
`usage.server_tool_use_details.web_search_requests` counts searches.

We use the results only as a list of URLs to fetch. The `content` excerpt is
shown to the searcher as a SNIPPET and is never stored as source text, so it can
never satisfy the citation verifier.
"""

from __future__ import annotations

import os

import httpx
from langsmith import traceable

from langchain_agents.core.config import APP_TITLE, APP_URL, model_id

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class WebSearchError(RuntimeError):
    pass


class OpenRouterWebSearch:
    def __init__(self, *, api_key: str | None = None, model: str | None = None, engine: str = "exa",
                 max_results: int = 5, max_characters: int = 500, transport: httpx.BaseTransport | None = None,
                 timeout_s: float = 60.0) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = (model or model_id()).removeprefix("openrouter:")
        self.engine, self.max_results, self.max_characters = engine, max_results, max_characters
        self.transport, self.timeout_s = transport, timeout_s

    @traceable(run_type="tool", name="openrouter_web_search")
    def search(self, query: str) -> dict:
        if not self.api_key:
            raise WebSearchError("OPENROUTER_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": (
                "Run exactly one web search for the query below. Then list each result on its own line as "
                f"'URL - title'. Nothing else.\n\nQuery: {query}")}],
            "tools": [{"type": "openrouter:web_search", "parameters": {
                "engine": self.engine, "max_results": self.max_results, "max_uses": 1,
                "max_characters": self.max_characters}}],
            "max_tokens": 800,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": APP_URL, "X-Title": APP_TITLE}
        with httpx.Client(transport=self.transport, timeout=self.timeout_s) as client:
            r = client.post(ENDPOINT, json=payload, headers=headers)
        if r.status_code >= 400:
            raise WebSearchError(f"OpenRouter web search HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage") or {}
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        results, seen = [], set()
        for ann in msg.get("annotations") or []:
            if ann.get("type") != "url_citation":
                continue
            cit = ann.get("url_citation") or {}
            url = cit.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({"url": url, "title": cit.get("title", ""), "snippet": (cit.get("content") or "")[:300]})
        return {
            "query": query,
            "results": results,
            "cost_usd": float(usage.get("cost") or 0.0),
            "web_search_requests": int((usage.get("server_tool_use_details") or {}).get("web_search_requests") or 0),
        }
