"""One structured-output call with the retry policy every node shares.

- Native JSON-schema structured output (Claude 5.5 via OpenRouter rejects forced
  tool_choice and non-default temperature).
- Transient errors (429, 5xx, timeouts, connection) retry with backoff; auth and
  bad-request errors fail immediately. An outage never becomes a fake answer.
- A response that fails schema parsing is retried once, then raises.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError",
    "ServiceUnavailableError", "TimeoutError", "ConnectionError", "ReadTimeout", "ConnectTimeout",
    "RemoteProtocolError",
}


def is_transient(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__)


class StructuredParseError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    def structured(self, schema: type[BaseModel], system: str, user: str, *, config: Any = None,
                   name: str = "") -> BaseModel: ...


class ChatStructuredLLM:
    """Wraps a LangChain chat model."""

    def __init__(self, model: BaseChatModel | None = None, *, max_transient_retries: int = 3,
                 initial_delay: float = 2.0) -> None:
        self._model = model
        self.max_transient_retries = max_transient_retries
        self.initial_delay = initial_delay

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:  # lazy, so importing the graph needs no API key
            from ..core.config import get_model
            self._model = get_model()
        return self._model

    def structured(self, schema, system, user, *, config=None, name=""):
        runnable = self.model.with_structured_output(schema, method="json_schema", include_raw=True)
        if name:
            runnable = runnable.with_config(run_name=name)
        messages = [SystemMessage(system), HumanMessage(user)]
        parse_failures = 0
        attempt = 0
        while True:
            try:
                out = runnable.invoke(messages, config=config)
            except Exception as exc:  # noqa: BLE001
                if is_transient(exc) and attempt < self.max_transient_retries:
                    time.sleep(self.initial_delay * (2 ** attempt))
                    attempt += 1
                    continue
                raise
            parsed = out.get("parsed") if isinstance(out, dict) else out
            if parsed is not None:
                return parsed
            parse_failures += 1
            if parse_failures > 1:
                raise StructuredParseError(f"{name or schema.__name__}: model output failed schema validation twice: "
                                           f"{out.get('parsing_error') if isinstance(out, dict) else ''}")
