"""Run one question end to end and write runs/research-<ts>/."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from langchain_agents.core.config import PROJECT_ROOT, model_id
from langchain_agents.core.cost import CostTracker

from .agent import build_graph
from .render import write_outputs


def stats(state: dict) -> dict:
    rounds = state.get("rounds") or []
    first, last = (rounds[0], rounds[-1]) if rounds else ({}, {})
    sources = state.get("sources") or {}
    final = state.get("final") or {}
    return {
        "raw_claims": first.get("n_claims", 0),
        "raw_verified": first.get("n_verified", 0),
        "raw_failed_by_reason": _reasons(first),
        "revisions": max(0, len(rounds) - 1),
        "final_claims": len(final.get("claims") or []),
        "final_dropped": len(final.get("dropped") or []),
        "final_failed_by_reason": _reasons(last),
        "sources_read": sum(1 for s in sources.values() if s.get("text")),
        "sources_failed": sum(1 for s in sources.values() if not s.get("text")),
        "searches": len(state.get("searches") or {}),
        "error": state.get("error"),
    }


def _reasons(rnd: dict) -> dict:
    out: dict[str, int] = {}
    for c in rnd.get("checks") or []:
        if not c["verified"]:
            out[c["reason"]] = out.get(c["reason"], 0) + 1
    return out


def run_question(question: str, *, mode: str = "web", out_dir: Path | str | None = None, model=None,
                 graph=None, write: bool = True, **graph_kwargs) -> dict:
    graph = graph or build_graph(model, mode=mode, **graph_kwargs)  # type: ignore[arg-type]
    tracker = CostTracker()
    t0 = time.monotonic()
    state = graph.invoke({"question": question},
                         {"callbacks": [tracker], "recursion_limit": 60, "run_name": "research",
                          "metadata": {"agent": "research", "mode": mode}, "tags": ["research", mode]})
    latency = time.monotonic() - t0
    search_cost = sum(float(s.get("cost_usd") or 0) for s in (state.get("searches") or {}).values())
    llm = tracker.summary()
    meta = {
        "question": question, "mode": mode,
        "model": model if isinstance(model, str) else (model_id() if model is None else type(model).__name__),
        "latency_s": round(latency, 2),
        "cost_usd": round(llm["cost_usd"] + search_cost, 6),
        "llm_cost_usd": llm["cost_usd"], "search_cost_usd": round(search_cost, 6),
        "model_calls": llm["model_calls"], "input_tokens": llm["input_tokens"], "output_tokens": llm["output_tokens"],
        "stats": stats(state),
    }
    paths = {}
    if write:
        if out_dir is None:
            out_dir = PROJECT_ROOT / "runs" / f"research-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        paths = {k: str(v) for k, v in write_outputs(
            Path(out_dir), question, state, meta).items()}
    return {"state": state, "meta": meta, "paths": paths}
