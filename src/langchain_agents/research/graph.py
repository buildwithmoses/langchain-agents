"""Entry point for `langgraph dev` / LangSmith Deployment (langgraph.json -> research).

Input: {"question": "..."} (or a chat message). Live web mode by default; set
LCA_RESEARCH_MODE=corpus to serve the offline fictional corpus instead.
No checkpointer here: the LangGraph server injects its own persistence."""

from __future__ import annotations

import os

from langchain_agents.core.config import load_env

from langchain_agents.research.agent import build_graph

load_env()
graph = build_graph(mode="corpus" if os.getenv("LCA_RESEARCH_MODE", "web") == "corpus" else "web")
