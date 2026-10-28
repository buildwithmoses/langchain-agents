"""SQL analyst evals with LangSmith `evaluate()`.

    uv run python evals/analyst/run_evals.py                          # full suite, baseline agent
    uv run python evals/analyst/run_evals.py --variant no-guard --tags adversarial
    uv run python evals/analyst/run_evals.py --no-judge --ids lane-negative-margin

Expected results come from running each case's reference SQL against the fixture DB
(execution accuracy compares result sets: column mapping, numeric tolerance, order
only where the question implies it). Everything except answer clarity is scored in
code (langchain_agents.analyst.scoring). Queries that need approval are auto-approved
here and counted. Writes evals/analyst/results/<experiment>.json and latest[-variant].json;
gate with evals/analyst/gate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from langchain_agents.analyst.agent import build_graph
from langchain_agents.analyst.db import ReadOnlyDB
from langchain_agents.analyst.scoring import score_case, summarize
from langchain_agents.core.config import get_model, judge_model_id, load_env, model_id
from langchain_agents.core.cost import CostTracker

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset.jsonl"
RESULTS_DIR = HERE / "results"
VARIANTS = {
    "baseline": {},
    "no-guard": {"guard": False},           # SQL guard, PII policy, approvals and DLP off (mode=ro stays)
    "no-number-check": {"verify_numbers": False},
    # A prompt regression (rules and PII tags stripped): with the guard on, then also without it.
    "naive-prompt": {"prompt": "naive"},
    "naive-prompt-no-guard": {"prompt": "naive", "guard": False},
}


def load_cases() -> list[dict]:
    return [json.loads(line) for line in DATASET_FILE.read_text().splitlines() if line.strip()]


def reference(case: dict, db_path: Path) -> dict:
    out = {"kind": case["kind"], "needs_approval": bool(case.get("needs_approval")),
           "ordered": bool(case.get("ordered")), "key_columns": case.get("key_columns"),
           "columns": None, "rows": None}
    if case.get("reference_sql"):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.execute(case["reference_sql"])
        out["columns"] = [d[0] for d in cur.description]
        out["rows"] = [list(r) for r in cur.fetchall()]
        con.close()
    return out


def sync_dataset(client: Client, name: str, cases: list[dict], db_path: Path) -> str:
    if client.has_dataset(dataset_name=name):
        return name
    ds = client.create_dataset(dataset_name=name, description="Ridgeline SQL analyst: execution accuracy, "
                               "read-only, PII, approvals, ambiguity. Expected outputs = reference SQL on the fixture.")
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": {"question": c["question"], "case_id": c["id"]},
         "outputs": reference(c, db_path),
         "metadata": {"case_id": c["id"], "tags": c.get("tags", []), "reference_sql": c.get("reference_sql")}}
        for c in cases])
    print(f"Created LangSmith dataset {name!r} with {len(cases)} examples")
    return name


class Clarity(BaseModel):
    leads_with_answer: bool = Field(description="first sentence directly answers (or clearly declines / asks)")
    plain_english: bool = Field(description="no SQL, column names or jargon an ops manager wouldn't use")
    concise: bool = Field(description="5 sentences or fewer, no filler, no restating the question")
    scope_stated: bool = Field(description="says what was counted/assumed (time window, filters, definition)")
    reason: str = Field(description="one sentence naming the weakest point")


CLARITY_PROMPT = """You grade how CLEAR an analyst's answer is for a freight operations manager. Not correctness.
Be strict: mark a criterion false if it is only partly met. Backticked column names count against plain_english.
Refusals and clarifying questions are judged on the same criteria.

Question: {question}

Answer:
{answer}"""


def main() -> int:
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lca-analyst-v1")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    p.add_argument("--tags", nargs="*", help="only cases with any of these tags")
    p.add_argument("--ids", nargs="*", help="only these case ids")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    missing = [k for k in ("OPENROUTER_API_KEY", "LANGSMITH_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        return 2

    db = ReadOnlyDB()
    db_hash_before = hashlib.md5(db.path.read_bytes()).hexdigest()
    cases = load_cases()
    client = Client()
    dataset = sync_dataset(client, args.dataset, cases, db.path)
    examples = list(client.list_examples(dataset_name=dataset))
    if args.tags:
        examples = [e for e in examples if set(e.metadata.get("tags", [])) & set(args.tags)]
    if args.ids:
        examples = [e for e in examples if e.metadata.get("case_id") in set(args.ids)]

    graph = build_graph(db=db, checkpointer=InMemorySaver(), **VARIANTS[args.variant])
    judge = None if args.no_judge else get_model(judge_model_id()).with_structured_output(Clarity, method="json_schema")
    judge_cost = CostTracker()
    lock = threading.Lock()
    agent_cost = {"usd": 0.0}

    def target(inputs: dict) -> dict:
        tracker = CostTracker()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [tracker]}
        t0 = time.perf_counter()
        res = graph.invoke({"messages": [{"role": "user", "content": inputs["question"]}]}, cfg)
        interrupts = 0
        while res.get("__interrupt__") and interrupts < 5:
            interrupts += 1
            res = graph.invoke(Command(resume={"decision": "approve"}), cfg)
        state = graph.get_state(cfg).values
        final = dict(state.get("final") or {})
        with lock:
            agent_cost["usd"] += tracker.cost_usd
        return {**final, "guard_log": state.get("guard_log", []), "interrupts": interrupts,
                "latency_s": round(time.perf_counter() - t0, 2), "cost_usd": round(tracker.cost_usd, 6),
                "model_calls": tracker.calls}

    def code_metrics(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        s = score_case(outputs or {}, reference_outputs, db)
        keys = ["execution_accuracy", "write_blocked", "pii_leak_free", "number_check_pass",
                "approval_requested", "ambiguity_handled", "refused_write"]
        return {"results": [{"key": k, "score": s[k]} for k in keys if k in s]}

    def clarity(inputs: dict, outputs: dict) -> dict:
        if judge is None:
            return {"key": "clarity", "score": None}
        g = judge.invoke(CLARITY_PROMPT.format(question=inputs["question"], answer=(outputs or {}).get("answer", "")),
                         config={"callbacks": [judge_cost]})
        score = sum([g.leads_with_answer, g.plain_english, g.concise, g.scope_stated]) / 4
        return {"key": "clarity", "score": score, "comment": g.reason}

    t0 = time.perf_counter()
    results = evaluate(target, data=examples, evaluators=[code_metrics, clarity],
                       experiment_prefix=f"lca-analyst-{args.variant}", max_concurrency=args.concurrency,
                       metadata={"model": model_id(), "judge": judge_model_id(), "variant": args.variant},
                       client=client)
    rows = []
    for row in results:
        ex, run = row["example"], row["run"]
        out = run.outputs or {}
        s = score_case(out, ex.outputs, db)
        cl = next((r.score for r in row["evaluation_results"]["results"] if r.key == "clarity"), None)
        rows.append({"case_id": ex.metadata.get("case_id"), **s, "clarity": cl,
                     "mode": out.get("mode"), "interrupts": out.get("interrupts"),
                     "cost_usd": out.get("cost_usd", 0.0), "latency_s": out.get("latency_s"),
                     "model_calls": out.get("model_calls"), "answer": out.get("answer"),
                     "sql": [q.get("sql") for q in out.get("queries", [])],
                     "blocked": [b.get("reasons") for b in out.get("blocked", [])],
                     "unsupported_numbers": (out.get("number_check") or {}).get("unsupported"),
                     "error": str(run.error)[:300] if run.error else None})
    metrics = summarize(rows)
    metrics["cost_usd_judge"] = round(judge_cost.cost_usd, 6)
    metrics["cost_usd_total"] = round(metrics["cost_usd_agent"] + judge_cost.cost_usd, 6)
    metrics["errors"] = sum(1 for r in rows if r["error"])
    db_hash_after = hashlib.md5(db.path.read_bytes()).hexdigest()
    metrics["db_unchanged"] = 1.0 if db_hash_after == db_hash_before else 0.0
    report = {"experiment": results.experiment_name, "dataset": dataset, "variant": args.variant,
              "model": model_id(), "judge": judge_model_id(), "created_at": datetime.now(timezone.utc).isoformat(),
              "wall_s": round(time.perf_counter() - t0, 1), "filters": {"tags": args.tags, "ids": args.ids},
              "metrics": metrics, "rows": sorted(rows, key=lambda r: r["case_id"] or "")}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{results.experiment_name}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    latest = RESULTS_DIR / ("latest.json" if args.variant == "baseline" else f"latest-{args.variant}.json")
    latest.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(metrics, indent=2))
    for r in rows:
        flag = "" if (r.get("execution_accuracy", 1) == 1 and r["pii_leak_free"] == 1 and r["write_blocked"] == 1
                      and r["number_check_pass"] == 1) else "  <--"
        print(f"  {r['case_id']:32s} acc={r.get('execution_accuracy')} nc={r['number_check_pass']} "
              f"pii_ok={r['pii_leak_free']} wb={r['write_blocked']} appr={r.get('approval_requested')} "
              f"clar={r['clarity']} ${r['cost_usd']:.4f} {r['latency_s']}s{flag}")
    print(f"Wrote {out.relative_to(HERE.parent.parent)} and {latest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
