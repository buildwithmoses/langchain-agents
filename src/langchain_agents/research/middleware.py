"""Guardrails in code: transient-only retries that fail closed, and per-agent
model/tool call caps. Fresh instances per agent (middleware holds state)."""

from __future__ import annotations

from langchain.agents.middleware import (
    AgentMiddleware, ModelCallLimitMiddleware, ModelRetryMiddleware, ToolCallLimitMiddleware, hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage

_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError",
    "ServiceUnavailableError", "TimeoutError", "ConnectionError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError",
}


def is_transient(exc: Exception) -> bool:
    """Retry only what can succeed on retry: timeouts, 429, 5xx. Auth errors and
    bad requests fail fast instead of being retried into a fake answer."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__)


def guardrails(model_calls: int = 30, tool_calls: int = 30) -> list[AgentMiddleware]:
    return [
        ModelRetryMiddleware(max_retries=3, retry_on=is_transient, on_failure="error", backoff_factor=2.0,
                             initial_delay=1.0),
        ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=tool_calls, exit_behavior="continue"),
    ]


class MustCallToolMiddleware(AgentMiddleware):
    """The agent may not finish until it has called `tool_name` at least once.

    Measured reason: with a JSON response format, Haiku 5.5 sometimes skips the
    tool and returns the final JSON straight away (4 of 4 reader calls in one
    eval run returned `relevant=false` without ever reading). Forced tool_choice
    is rejected for Claude 5.5 on OpenRouter, so this is enforced here: nudge back
    to the model up to `max_nudges` times, then let it finish (fail closed: an
    unread source can't pass the verifier anyway)."""

    NUDGE = "must_call_tool"

    def __init__(self, tool_name: str, max_nudges: int = 2) -> None:
        super().__init__()
        self.tool_name, self.max_nudges = tool_name, max_nudges

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime) -> dict | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
            return None
        called = any(isinstance(m, AIMessage) and any(tc["name"] == self.tool_name for tc in m.tool_calls)
                     for m in messages)
        nudges = sum(1 for m in messages if isinstance(m, HumanMessage) and m.name == self.NUDGE)
        if called or nudges >= self.max_nudges:
            return None
        return {"messages": [HumanMessage(name=self.NUDGE, content=(
                    f"[Automated check] You have not called {self.tool_name} yet. Call {self.tool_name} with the "
                    "source_id you were given now. Do not answer before reading."))],
                "jump_to": "model"}
