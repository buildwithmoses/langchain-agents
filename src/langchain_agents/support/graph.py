"""Entry point for `langgraph dev` (registered in langgraph.json as support).

No checkpointer or store here: the LangGraph server provides persistence (threads)
and the long-term Store. Tests and scripts call agent.build_graph() instead.
"""

from langchain_agents.core.config import load_env
from langchain_agents.support.agent import build_graph

load_env()
graph = build_graph()
