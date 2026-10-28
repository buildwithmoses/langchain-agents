"""Run one review end to end: graph -> lawyer interrupt -> resume -> run folder.

Output: runs/contracts-<ts>/memo.md, findings.json, redlines.md (cost + latency
are in findings.json["run"])."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ..core.config import PROJECT_ROOT, model_id
from ..core.cost import CostTracker
from .pipeline import build_graph

Decider = Callable[[dict], dict]


def auto_approve_all(payload: dict) -> dict:
    """EVAL ONLY: approve every redline unchanged. Never use for a real review."""
    return {"decisions": [{"id": r["id"], "action": "approve"} for r in payload.get("redlines", [])]}


def review(contract: str | Path, *, decide: Decider, variant: str = "baseline", llm: Any = None,
           graph: Any = None, write: bool = True, out_root: Path | None = None,
           tags: list[str] | None = None) -> dict:
    graph = graph or build_graph(llm, checkpointer=InMemorySaver())
    tracker = CostTracker()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [tracker],
           "run_name": "contracts-review", "tags": ["contracts", variant, *(tags or [])],
           "metadata": {"agent": "contracts", "variant": variant, "model": model_id(), "contract": str(contract)}}
    t0 = time.perf_counter()
    res = graph.invoke({"contract_path": str(contract), "variant": variant}, cfg)
    t_review = time.perf_counter() - t0
    interrupts = res.get("__interrupt__") or []
    if not interrupts:
        raise RuntimeError("review finished without the lawyer-review interrupt; refusing to produce redlines")
    payload = interrupts[0].value
    answer = decide(payload)
    res = graph.invoke(Command(resume=answer), cfg)
    total = time.perf_counter() - t0
    state = graph.get_state(cfg).values
    run = {"model": model_id(), "variant": variant, "latency_s": round(t_review, 2),
           "latency_incl_review_s": round(total, 2), **tracker.summary()}
    result = {"state": state, "run": run}
    if write:
        result["out_dir"] = str(write_run(state, run, out_root))
    return result


def write_run(state: dict, run: dict, out_root: Path | None = None) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (out_root or PROJECT_ROOT / "runs") / f"contracts-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "memo.md").write_text(state["memo_md"])
    (out / "redlines.md").write_text(state["redlines_md"])
    findings = {
        "contract": state.get("contract_name"), "carrier": state.get("carrier"), "run": run,
        "findings": state.get("findings", []), "redlines": state.get("redlines", []),
        "decisions": state.get("decisions", {}), "extraction": state.get("extraction", {}),
        "verification": state.get("verification", {}), "first_pass_problems": state.get("first_pass_problems", {}),
        "errors": state.get("errors", []),
    }
    (out / "findings.json").write_text(json.dumps(findings, indent=2))
    return out
