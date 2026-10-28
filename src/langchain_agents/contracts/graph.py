"""Module-level compiled graph for `langgraph dev` (langgraph.json -> contracts).

No checkpointer here: the LangGraph platform supplies persistence. The model is
created lazily on first use, so importing this module needs no API key.
Input: {"contract_path": "02-iron-horse-transport.md"} (a fixture name or a path)
or {"contract_text": "...", "contract_name": "..."}.
"""

from langchain_agents.contracts.pipeline import build_graph

graph = build_graph()
