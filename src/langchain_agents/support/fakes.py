"""A scripted chat model for offline tests and the no-key smoke run."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Plays back a fixed list of AIMessages and records every prompt it was shown."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):  # the script decides which tools get called
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        if self.cursor >= len(self.responses):
            raise AssertionError(f"Script exhausted after {self.cursor} calls")
        msg = self.responses[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def ai_tool(name: str, args: dict, call_id: str, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)
