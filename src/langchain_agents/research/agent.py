"""Assemble the research agent.

    build_research_agent(...)  the Deep Agent: a lead that plans (write_todos), keeps
                               notes in the virtual filesystem, and delegates to a
                               `searcher` and a `reader` subagent; returns a Brief.
    build_graph(...)           the outer LangGraph: research -> verify ->
                               (revise -> research, once) -> finalize.

The verify/finalize nodes are plain code. The model can't talk its way past them.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from deepagents import SubAgent, create_deep_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from langchain_agents.core.config import get_model

from .corpus import Corpus
from .fetch import Fetcher
from .middleware import MustCallToolMiddleware, guardrails
from .prompts import READER_PROMPT, REVISE_PROMPT, SEARCHER_PROMPT, lead_prompt, question_message
from .schemas import Brief, ReaderNotes
from .state import ResearchAgentState, ResearchState
from .tools import make_tools
from .verify import ClaimCheck, finalize, verify_claims
from .websearch import OpenRouterWebSearch

Mode = Literal["web", "corpus"]


def build_research_agent(model: BaseChatModel | str | None = None, *, mode: Mode = "web", corpus: Corpus | None = None,
                         searcher: OpenRouterWebSearch | None = None, fetcher: Fetcher | None = None,
                         max_searches: int = 4, max_sources: int = 6, checkpointer: Any = None):
    llm = model if isinstance(model, BaseChatModel) else get_model(model)
    search, read_source = make_tools(mode, corpus=corpus, searcher=searcher, fetcher=fetcher,
                                     max_searches=max_searches, max_sources=max_sources)
    subagents = [
        SubAgent(name="searcher", model=llm, tools=[search], middleware=guardrails(model_calls=8, tool_calls=6),
                 description="Finds candidate sources for a question. Give it the question and sub-questions. "
                             "Returns source_ids with titles. It does not read sources.",
                 system_prompt=SEARCHER_PROMPT.format(n_search=max_searches)),
        SubAgent(name="reader", model=llm, tools=[read_source],
                 middleware=[*guardrails(model_calls=5, tool_calls=2), MustCallToolMiddleware("read_source")],
                 description="Reads ONE source (give it the exact source_id and the question) and returns verbatim "
                             "quotes relevant to the question.",
                 system_prompt=READER_PROMPT, response_format=ProviderStrategy(ReaderNotes)),
    ]
    return create_deep_agent(
        model=llm,
        tools=[],
        system_prompt=lead_prompt(max_sources),
        subagents=subagents,
        middleware=guardrails(model_calls=25, tool_calls=30),
        response_format=ProviderStrategy(Brief),
        state_schema=ResearchAgentState,
        checkpointer=checkpointer,
        name="research-lead",
    )


def _question(state: ResearchState) -> str:
    if state.get("question"):
        return state["question"]
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.text if hasattr(m, "text") else str(m.content)
    return ""


def _dump(obj: Any) -> dict | None:
    if obj is None:
        return None
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


def _feedback(checks: list[ClaimCheck]) -> str:
    lines = []
    for c in checks:
        if c.verified:
            continue
        line = f"- Claim [{c.index}] ({c.reason}): {c.detail}.\n  claim: {c.claim}\n  source_id: {c.source_id}\n  quote: {c.quote!r}"
        if c.closest_passage and (c.closest_ratio or 0) >= 0.6:
            line += f"\n  closest passage in that source: {c.closest_passage!r}"
        lines.append(line)
    return "\n".join(lines)


def build_graph(model: BaseChatModel | str | None = None, *, mode: Mode = "web", corpus: Corpus | None = None,
                searcher: OpenRouterWebSearch | None = None, fetcher: Fetcher | None = None,
                max_searches: int = 4, max_sources: int = 6, max_revisions: int = 1, recursion_limit: int = 120,
                checkpointer: Any = None):
    agent = build_research_agent(model, mode=mode, corpus=corpus, searcher=searcher, fetcher=fetcher,
                                 max_searches=max_searches, max_sources=max_sources)

    def research(state: ResearchState, config: RunnableConfig) -> dict:
        attempts = state.get("attempts", 0)
        new_msgs = []
        if attempts == 0:
            new_msgs = [HumanMessage(question_message(_question(state), mode), id=str(uuid.uuid4()))]
        messages = list(state.get("messages") or []) + new_msgs
        try:
            out = agent.invoke({"messages": messages, "sources": state.get("sources") or {},
                                "searches": state.get("searches") or {}, "files": state.get("files") or {}},
                               {**config, "recursion_limit": recursion_limit})
        except Exception as exc:  # fail closed: no brief, the finalize node reports it
            return {"messages": new_msgs, "brief": None, "attempts": attempts + 1,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        brief = _dump(out.get("structured_response"))
        update = {"messages": out.get("messages", []), "sources": out.get("sources") or {},
                  "searches": out.get("searches") or {}, "files": out.get("files") or {}, "brief": brief, "attempts": attempts + 1,
                  "error": None if brief else "the agent finished without returning a structured brief"}
        if attempts == 0:
            update["raw_brief"] = brief
        return update

    def verify(state: ResearchState) -> dict:
        brief = state.get("brief")
        checks = verify_claims((brief or {}).get("claims") or [], state.get("sources") or {})
        return {"rounds": [{
            "attempt": state.get("attempts", 0),
            "n_claims": len(checks),
            "n_verified": sum(c.verified for c in checks),
            "checks": [c.to_dict() for c in checks],
            "error": state.get("error"),
        }]}

    def route(state: ResearchState) -> str:
        last = state["rounds"][-1]
        failed = last["n_claims"] - last["n_verified"]
        if state.get("brief") and failed and state.get("attempts", 0) <= max_revisions:
            return "revise"
        return "finalize"

    def revise(state: ResearchState) -> dict:
        last = state["rounds"][-1]
        checks = [ClaimCheck(**c) for c in last["checks"]]
        msg = REVISE_PROMPT.format(n_bad=last["n_claims"] - last["n_verified"], n_total=last["n_claims"],
                                   details=_feedback(checks))
        return {"messages": [HumanMessage(msg, name="citation_check", id=str(uuid.uuid4()))]}

    def finalize_node(state: ResearchState) -> dict:
        last = state["rounds"][-1]
        checks = [ClaimCheck(**c) for c in last["checks"]]
        final = finalize(state.get("brief"), checks)
        if state.get("error") and not state.get("brief"):
            final["gaps"].append(f"Run error: {state['error']}")
        summary = (f"Brief ready: {len(final['claims'])} verified claims, {len(final['dropped'])} dropped.\n\n"
                   f"{final['answer']}")
        return {"final": final, "messages": [AIMessage(summary, id=str(uuid.uuid4()))]}

    g = StateGraph(ResearchState)
    g.add_node("research", research)
    g.add_node("verify", verify)
    g.add_node("revise", revise)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "research")
    g.add_edge("research", "verify")
    g.add_conditional_edges("verify", route, {"revise": "revise", "finalize": "finalize"})
    g.add_edge("revise", "research")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name="research")
