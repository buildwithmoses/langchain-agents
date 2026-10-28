"""Close the loop in production with LangSmith's own automation rules: every
trace that errors or gets a low user score is copied into the eval dataset, so
next week's eval set contains this week's failures. Dry run by default."""

from __future__ import annotations

import json
import os

import httpx

DEFAULT_FILTER = 'or(eq(status, "error"), and(eq(feedback_key, "user_score"), lt(feedback_score, 0.5)))'


def rule_body(project_id: str, dataset_id: str, name: str, filter: str = DEFAULT_FILTER,
              sampling_rate: float = 1.0) -> dict:
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    return {"display_name": name, "session_id": project_id, "is_enabled": True, "sampling_rate": sampling_rate,
            "filter": filter, "add_to_dataset_id": dataset_id, "add_to_dataset_prefer_correction": True}


def endpoint() -> str:
    return os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/") + "/api/v1/runs/rules"


def create_rule(body: dict) -> dict:
    r = httpx.post(endpoint(), json=body, headers={"x-api-key": os.environ["LANGSMITH_API_KEY"]}, timeout=30)
    r.raise_for_status()
    return r.json()


def plan(project: str, dataset: str, filter: str = DEFAULT_FILTER, client=None) -> dict:
    from langsmith import Client

    client = client or Client()
    proj = client.read_project(project_name=project)
    ds = client.read_dataset(dataset_name=dataset)
    body = rule_body(str(proj.id), str(ds.id), f"evalkit: failures from {project} -> {dataset}", filter)
    return {"method": "POST", "url": endpoint(), "body": body}


def describe(p: dict) -> str:
    return f"{p['method']} {p['url']}\n{json.dumps(p['body'], indent=2)}"
