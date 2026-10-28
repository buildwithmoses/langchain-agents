"""Named variants for eval kit demos, as `module:factory` targets.

    langchain_agents.support.agent:build_graph                         the shipped agent
    langchain_agents.evalkit.targets:support_detention_unapproved      regression: detention reviews skip approval
"""

from __future__ import annotations


def support_detention_unapproved(**kwargs):
    from langchain_agents.support.agent import CONSEQUENTIAL_TOOLS, build_graph

    tools = tuple(t for t in CONSEQUENTIAL_TOOLS if t != "request_detention_review")
    return build_graph(approval_tools=tools, **kwargs)
