"""Offline test helpers: a scripted tool-calling chat model and a throwaway fixture DB."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .db import ReadOnlyDB
from .fixture_db import build_db

_DB_PATH: Path | None = None


def shared_db() -> ReadOnlyDB:
    """Build the fixture once per test process in a temp dir (never touches fixtures/)."""
    global _DB_PATH
    if _DB_PATH is None or not _DB_PATH.exists():
        _DB_PATH = build_db(Path(tempfile.mkdtemp(prefix="analyst-test-")) / "ridgeline.db")
    return ReadOnlyDB(_DB_PATH)


def sql_call(sql: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "run_query", "args": {"sql": sql}, "id": call_id}])


class ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order; records what it was sent."""

    script: list[Any]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ARG002
        self.seen.append(list(messages))
        if not self.script:
            raise AssertionError("ScriptedModel ran out of responses")
        nxt = self.script.pop(0)
        msg = nxt(messages) if callable(nxt) else nxt
        if isinstance(msg, str):
            msg = AIMessage(content=msg)
        return ChatResult(generations=[ChatGeneration(message=msg)])
