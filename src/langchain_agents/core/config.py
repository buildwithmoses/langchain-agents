"""Runtime configuration. Everything is overridable by environment variable.

Models go through OpenRouter (langchain-openrouter): one OPENROUTER_API_KEY and
any id from openrouter.ai/models. Lessons carried over from earlier builds:
- Claude 5.5 rejects forced tool_choice and non-default temperature through
  OpenRouter, so structured output uses native JSON schema, never function calling.
- `require_parameters` keeps a request off hosts that would silently drop
  response_format or tools.
- ChatOpenRouter's `timeout` is in MILLISECONDS. 180 means 180 ms.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "openrouter:anthropic/claude-haiku-5.5"
# Judges should not share a family with the model they grade.
DEFAULT_JUDGE_MODEL = "openrouter:openai/gpt-6.1-terra"
APP_URL = "https://buildwithmoses.vercel.app"
APP_TITLE = "LangChain Agents (buildwithmoses)"
OPENROUTER_PROVIDER = {"require_parameters": True, "allow_fallbacks": True}
TIMEOUT_MS = 180_000


def load_env() -> None:
    """Load this project's .env, then fill unset vars from .env files further up
    (buildwithmoses-automations/.env holds OPENROUTER_API_KEY). Real env vars win."""
    from dotenv import dotenv_values

    for folder in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        env_file = folder / ".env"
        if env_file.is_file():
            for key, value in dotenv_values(env_file).items():
                if value and not os.environ.get(key):
                    os.environ[key] = value


def model_id() -> str:
    return os.getenv("LCA_MODEL", DEFAULT_MODEL)


def judge_model_id() -> str:
    return os.getenv("LCA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def get_model(mid: str | None = None, **kwargs) -> BaseChatModel:
    """A chat model from an init_chat_model id. OpenRouter ids get attribution,
    provider routing, a real timeout and no SDK-level retries (one retry layer only:
    the caller's, which retries transient errors)."""
    mid = mid or model_id()
    if mid.startswith("openrouter:"):
        kwargs.setdefault("timeout", TIMEOUT_MS)
        kwargs.setdefault("max_retries", 0)
        return init_chat_model(mid, app_url=APP_URL, app_title=APP_TITLE,
                               openrouter_provider=kwargs.pop("openrouter_provider", OPENROUTER_PROVIDER), **kwargs)
    return init_chat_model(mid, **kwargs)


def data_dir(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath("fixtures", *parts)
