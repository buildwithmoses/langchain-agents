"""LangChain Agents: five proof-of-concept shapes for one customer, built on LangChain, LangGraph and Deep Agents.

    langchain-agents support ...    Support agent with memory and a human approval node
    langchain-agents research ...   Deep Agents research brief with verified citations
    langchain-agents contracts ...  Contract review against a playbook, lawyer approves
    langchain-agents analyst ...    SQL analyst with read-only enforcement and query approval
    langchain-agents evalkit ...    Mine traces, build a dataset, set up LangSmith evaluators and a gate
"""

from __future__ import annotations

import sys

USAGE = __doc__
COMMANDS = {"support", "research", "contracts", "analyst", "evalkit"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"Unknown command {cmd!r}.\n{USAGE}")
        return 2
    from importlib import import_module

    from .core.config import load_env

    load_env()
    return import_module(f".{cmd}.cli", __package__).cli(rest) or 0
