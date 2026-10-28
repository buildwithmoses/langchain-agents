"""Module-level compiled graph for `langgraph dev` / LangSmith Deployment (see langgraph.json).

No checkpointer here: the platform supplies persistence (and the approval interrupt
uses it). The fixture DB is built on first import if it is missing.
"""

from langchain_agents.core.config import load_env

from langchain_agents.analyst.agent import build_graph

load_env()
graph = build_graph()
