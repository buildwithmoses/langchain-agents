"""Build the support agent graph.

Why `create_agent` (a LangGraph graph under the hood) and not a hand-built StateGraph
or a deep agent: this is one conversational loop over a handful of tools. Everything
custom (account binding, memory injection, the reply guard, approvals) is middleware
on that loop, and the prebuilt HumanInTheLoopMiddleware emits the interrupt format
that LangChain's agent-chat-ui already renders (approve / edit / reject). A planner
plus subagents would add latency and tokens for no gain here.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

from langchain_agents.core.config import get_model
from langchain_agents.support import prompts
from langchain_agents.support.middleware import (
    AccountBindingMiddleware,
    ReplyGuardMiddleware,
    ShipperContextMiddleware,
    SupportState,
    is_transient,
)
from langchain_agents.support.tools import ALL_TOOLS, CONSEQUENTIAL_TOOLS, READ_ONLY_TOOLS, prior_escalation

DECISIONS = ["approve", "edit", "reject"]  # no free-text "respond": an action is run, corrected, or declined


def describe_action(tool_call: dict, state: Any, runtime: Any) -> str:
    name, args = tool_call["name"], tool_call["args"]
    acct = (state or {}).get("account") or "unbound"
    title = {"file_claim_intake": "File a cargo claim intake",
             "request_detention_review": "Send detention to billing for review",
             "escalate_to_ops": "Escalate to Ridgeline operations"}.get(name, name)
    return f"{title} (shipper {acct}). Approve, edit the arguments, or reject.\n{json.dumps(args, indent=1)}"


def needs_review(request: Any) -> bool:
    """HITL `when` predicate. Every consequential call is reviewed, except a duplicate
    escalate_to_ops for a load already escalated in this thread: the tool files nothing
    for it (same check, tools.prior_escalation), so there is nothing to approve.
    Deterministic across interrupt and resume: the first escalation only appears in
    state after it was approved and ran."""
    tc = request.tool_call
    if tc["name"] != "escalate_to_ops":
        return True
    return prior_escalation(request.state, str(tc["args"].get("load_id", ""))) is None


INTERRUPT_ON = {name: {"allowed_decisions": DECISIONS, "description": describe_action, "when": needs_review}
                for name in CONSEQUENTIAL_TOOLS}


def build_middleware(*, guard: bool = True, approval_tools: tuple[str, ...] = CONSEQUENTIAL_TOOLS) -> list:
    return [
        AccountBindingMiddleware(),
        ShipperContextMiddleware(),
        ReplyGuardMiddleware(enabled=guard),
        # Fail closed: an outage is a failed run, never an error string dressed up as a reply.
        ModelRetryMiddleware(max_retries=3, retry_on=is_transient, on_failure="error", initial_delay=1.0),
        ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=12, exit_behavior="continue"),
        # Only read-only tools are retried. A consequential action is never re-run automatically.
        ToolRetryMiddleware(max_retries=2, tools=list(READ_ONLY_TOOLS), retry_on=is_transient, on_failure="continue"),
        HumanInTheLoopMiddleware(interrupt_on={k: v for k, v in INTERRUPT_ON.items() if k in approval_tools},
                                 description_prefix="Ridgeline support action"),
    ]


def build_graph(model: BaseChatModel | str | None = None, checkpointer: Any = None, store: Any = None, *,
                guard: bool = True, prompt_variant: str = "default",
                approval_tools: tuple[str, ...] = CONSEQUENTIAL_TOOLS):
    """Compile the support agent.

    Under `langgraph dev` / LangSmith Deployment leave checkpointer and store as None:
    the platform supplies both. Scripts, tests and evals pass InMemorySaver /
    InMemoryStore (interrupts need a checkpointer; memory needs a store).
    `guard`, `prompt_variant` and `approval_tools` exist only so the eval suite can
    build deliberately broken variants and prove the gate blocks them.
    """
    llm = model if isinstance(model, BaseChatModel) else get_model(model)
    system = prompts.SYSTEM_PROMPT
    if prompt_variant == "weak":
        if prompts.RULE_1 not in system:
            raise ValueError("weak prompt variant: rule 1 not found in SYSTEM_PROMPT")
        system = system.replace(prompts.RULE_1, prompts.WEAK_RULE_1)
    return create_agent(
        llm,
        tools=ALL_TOOLS,
        system_prompt=system,
        middleware=build_middleware(guard=guard, approval_tools=approval_tools),
        state_schema=SupportState,
        checkpointer=checkpointer,
        store=store,
        name="ridgeline-support",
    )
