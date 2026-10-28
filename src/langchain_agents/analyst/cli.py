"""`langchain-agents analyst ...`

    langchain-agents analyst ask "Which lanes lost money last month?"
    langchain-agents analyst ask "How many carriers have a contact email on file?" --yes
    langchain-agents analyst chat
    langchain-agents analyst build-db

Queries that touch PII or full-scan a big table stop for approval: [a]pprove, [e]dit, [r]eject.
--yes approves automatically; without a terminal (and without --yes) they are rejected.
Every run writes runs/analyst-<timestamp>/ (answer.md, run.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_agents.core.config import PROJECT_ROOT, model_id
from langchain_agents.core.cost import CostTracker

from .db import ReadOnlyDB, default_db_path


def _prompt_approval(payload: dict, auto: str | None) -> dict:
    print("\n--- Query needs approval -------------------------------------------")
    for r in payload.get("reasons", []):
        print(f"  why: {r}")
    print(payload.get("sql", ""))
    print("--------------------------------------------------------------------")
    if auto:
        print(f"(auto-{auto})")
        return {"decision": auto}
    if not sys.stdin.isatty():
        print("(no terminal: rejected; pass --yes to auto-approve)")
        return {"decision": "reject", "reason": "no reviewer available"}
    while True:
        choice = input("[a]pprove / [e]dit / [r]eject? ").strip().lower()[:1]
        if choice == "a":
            return {"decision": "approve"}
        if choice == "r":
            return {"decision": "reject", "reason": input("Reason (optional): ").strip() or "rejected by reviewer"}
        if choice == "e":
            print("Enter SQL; finish with an empty line:")
            lines = []
            while (line := input()).strip():
                lines.append(line)
            if lines:
                return {"decision": "edit", "sql": "\n".join(lines)}


def run_turn(graph, question: str, thread_id: str, tracker: CostTracker, auto: str | None) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [tracker],
           "run_name": "analyst", "metadata": {"agent": "analyst", "model": model_id()}}
    t0 = time.perf_counter()
    result = graph.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
    while result.get("__interrupt__"):
        intr = result["__interrupt__"][0]
        result = graph.invoke(Command(resume=_prompt_approval(intr.value, auto)), cfg)
    final = dict(result.get("final") or {})
    final["latency_s"] = round(time.perf_counter() - t0, 2)
    return final


def _run_dir() -> Path:
    d = PROJECT_ROOT / "runs" / f"analyst-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _footer(final: dict, tracker: CostTracker) -> str:
    s = tracker.summary()
    return (f"_{final.get('latency_s', 0)}s, {s['model_calls']} model calls, ${s['cost_usd']:.4f} "
            f"({s['input_tokens']} in / {s['output_tokens']} out tokens), model {model_id()}_")


def cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="langchain-agents analyst", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    c = sub.add_parser("chat", help="multi-turn session")
    for sp in (a, c):
        sp.add_argument("--yes", action="store_true", help="auto-approve queries that need approval")
        sp.add_argument("--reject", action="store_true", help="auto-reject queries that need approval")
        sp.add_argument("--db", type=Path, default=None)
        sp.add_argument("--json", action="store_true", help="print run JSON instead of markdown")
    sub.add_parser("build-db", help="(re)build the fixture database")
    args = p.parse_args(argv)

    if args.cmd == "build-db":
        from .fixture_db import build_db

        path = build_db(default_db_path())
        print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return 0

    from .agent import build_graph

    db = ReadOnlyDB(args.db) if args.db else ReadOnlyDB()
    graph = build_graph(db=db, checkpointer=InMemorySaver())
    auto = "approve" if args.yes else ("reject" if args.reject else None)
    run_dir = _run_dir()
    thread = str(uuid.uuid4())

    if args.cmd == "ask":
        tracker = CostTracker()
        final = run_turn(graph, args.question, thread, tracker, auto)
        final["cost"] = tracker.summary()
        md = final.get("markdown", "") + "\n\n" + _footer(final, tracker)
        (run_dir / "answer.md").write_text(f"# {args.question}\n\n{md}\n")
        (run_dir / "run.json").write_text(json.dumps({"question": args.question, "model": model_id(), **final},
                                                     indent=2, default=str))
        print(json.dumps(final, indent=2, default=str) if args.json else md)
        print(f"\n(saved to {run_dir.relative_to(PROJECT_ROOT)})", file=sys.stderr)
        return 0 if final.get("mode") == "answer" else 1

    print("Ridgeline SQL analyst. Ask about loads, lanes, carriers, shippers. Ctrl-D or 'exit' to quit.")
    turns = []
    total = CostTracker()
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        tracker = CostTracker()
        final = run_turn(graph, q, thread, tracker, auto)
        final["cost"] = tracker.summary()
        for k in ("cost_usd",):
            total.cost_usd += final["cost"][k]
        total.calls += final["cost"]["model_calls"]
        print("\n" + final.get("markdown", "") + "\n\n" + _footer(final, tracker))
        turns.append({"question": q, **final})
        (run_dir / "run.json").write_text(json.dumps({"model": model_id(), "turns": turns}, indent=2, default=str))
        (run_dir / "answer.md").write_text("\n\n---\n\n".join(f"# {t['question']}\n\n{t.get('markdown', '')}"
                                                             for t in turns) + "\n")
    print(f"\n{len(turns)} turn(s), ${total.cost_usd:.4f} total. Saved to {run_dir.relative_to(PROJECT_ROOT)}")
    return 0
