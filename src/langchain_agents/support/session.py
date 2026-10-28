"""One conversation thread with the support graph, including the approval loop.

Used by the terminal chat, the scripted demo and the evals, so all three exercise
exactly the same interrupt / resume path that agent-chat-ui uses.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langchain_agents.support.middleware import message_text
from langchain_agents.support.tools import ACTIONS_NS

# decide(action_request, review_config) -> decision dict ({"type": "approve"} etc.)
Decider = Callable[[dict, dict], dict]


def approve_all(_req: dict, _cfg: dict) -> dict:
    return {"type": "approve"}


def reject_all(_req: dict, _cfg: dict) -> dict:
    return {"type": "reject", "message": "Reviewer declined this action."}


@dataclass
class TurnResult:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)       # every tool call the model proposed this turn
    interrupts: list[dict] = field(default_factory=list)       # action requests shown to the reviewer
    decisions: list[dict] = field(default_factory=list)        # what the reviewer answered (aligned with interrupts)
    guard_events: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class SupportSession:
    def __init__(self, graph: Any, *, account: str | None = None, thread_id: str | None = None,
                 callbacks: list | None = None, metadata: dict | None = None) -> None:
        self.graph = graph
        self.account = account
        self.thread_id = thread_id or f"support-{uuid.uuid4().hex[:8]}"
        self.callbacks = callbacks or []
        self.metadata = metadata or {}
        self._sent_account = False

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "callbacks": self.callbacks,
                "metadata": {**self.metadata, "thread_id": self.thread_id}, "recursion_limit": 40}

    def state(self) -> dict:
        return self.graph.get_state(self.config).values

    def send(self, text: str, decide: Decider = approve_all, max_resumes: int = 6) -> TurnResult:
        t0 = time.perf_counter()
        before = len(self.state().get("messages", [])) if self._sent_account or not self.account else 0
        payload: dict = {"messages": [HumanMessage(content=text)]}
        if self.account and not self._sent_account:
            payload["account"] = self.account
            self._sent_account = True
        guard_before = len(self.state().get("guard_events", []) or [])
        out = self.graph.invoke(payload, self.config)
        interrupts, decisions = [], []
        for _ in range(max_resumes):
            pending = out.get("__interrupt__") or []
            if not pending:
                break
            value = pending[0].value
            reqs, cfgs = value.get("action_requests", []), value.get("review_configs", [])
            by_name = {c["action_name"]: c for c in cfgs}
            batch = []
            for req in reqs:
                d = decide(req, by_name.get(req["name"], {}))
                interrupts.append(req)
                decisions.append(d)
                batch.append(d)
            out = self.graph.invoke(Command(resume={"decisions": batch}), self.config)
        else:
            if out.get("__interrupt__"):
                raise RuntimeError("Too many approval rounds in one turn")
        state = self.state()
        new = state.get("messages", [])[before:]
        tool_calls = [tc for m in new if isinstance(m, AIMessage) for tc in m.tool_calls]
        ais = [m for m in new if isinstance(m, AIMessage) and message_text(m).strip()]
        reply = message_text(ais[-1]) if ais else ""
        return TurnResult(reply=reply, tool_calls=tool_calls, interrupts=interrupts, decisions=decisions,
                          guard_events=(state.get("guard_events") or [])[guard_before:],
                          seconds=round(time.perf_counter() - t0, 2))

    def tool_messages(self) -> list[ToolMessage]:
        return [m for m in self.state().get("messages", []) if isinstance(m, ToolMessage)]

    def executed_actions(self, store: Any) -> list[dict]:
        """Actions that actually ran (written by the tools) in this thread."""
        if store is None:
            return []
        return [i.value for i in store.search(ACTIONS_NS, filter={"thread_id": self.thread_id}, limit=100)]
