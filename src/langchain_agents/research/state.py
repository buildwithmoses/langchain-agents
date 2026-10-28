"""Graph state. Fetched sources live in state, not in a module global, so each
run (and each `langgraph dev` thread) has its own record of what was actually
read, and it survives the subagent boundary.

Deep Agents passes parent state into a subagent and writes the subagent's full
values back, so an accumulating reducer would double count. These reducers are
idempotent dict merges keyed by source id / search id."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired

from deepagents.graph import DeepAgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class ResearchAgentState(DeepAgentState):
    """State of the deep agent (lead + subagents inherit it)."""
    sources: NotRequired[Annotated[dict[str, dict[str, Any]], merge_dicts]]
    searches: NotRequired[Annotated[dict[str, dict[str, Any]], merge_dicts]]


class ResearchState(TypedDict, total=False):
    """Outer graph: research -> verify -> (revise -> research) -> finalize."""
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[dict[str, dict[str, Any]], merge_dicts]
    searches: Annotated[dict[str, dict[str, Any]], merge_dicts]
    files: Annotated[dict[str, Any], merge_dicts]  # the agent's virtual filesystem (notes), kept across revisions
    brief: dict | None
    raw_brief: dict | None
    rounds: Annotated[list[dict], operator.add]
    attempts: int
    error: str | None
    final: dict
