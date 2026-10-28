"""Research agent: a Deep Agents research brief where every claim is checked
against its source text in code before it reaches the customer.

LangChain already ships open-source research agents (open_deep_research, the
deepagents examples). This is not a new research architecture. It is what a
deployed engineer adds on top for an enterprise customer: a brief scoped to the
customer's question, a deterministic citation verifier that drops anything it
can't match to text we actually fetched, and an eval with a gate.
"""
