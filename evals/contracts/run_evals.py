"""Contract review evals with LangSmith evaluate() over the 8 fixture agreements.

    uv run python evals/contracts/run_evals.py                          # baseline pipeline
    uv run python evals/contracts/run_evals.py --variant no_verifier    # ablation: quote verifier off
    uv run python evals/contracts/run_evals.py --variant model_decides  # ablation: model judges compliance
    uv run python evals/contracts/run_evals.py --no-judge               # skip the memo-clarity judge
    uv run python evals/contracts/run_evals.py --upload                 # also upload to LangSmith

Default is LOCAL: evaluate(..., upload_results=False), LANGSMITH_TRACING=false, and
the examples are built from answers.json in-process. The LangSmith account hit its
monthly unique-trace limit (HTTP 429 on /runs/multipart) on 2026-10-28, so uploads
are off until the plan changes or the month resets. Pass --upload to turn them back on.

Each example is reviewed with the EVAL-ONLY auto-approve decider (no human).
Writes evals/contracts/results/<prefix>-<variant>-<ts>.json and latest.json;
evals/contracts/gate.py checks them. Dataset: lca-contracts-v1 (LangSmith).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

# Must happen before langsmith/langchain are imported: they read tracing config once.
if "--upload" not in sys.argv:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from langsmith import Client, evaluate
from langsmith import run_helpers as rh
from pydantic import BaseModel, Field

from langchain_agents.contracts.runner import auto_approve_all, review
from langchain_agents.contracts.scoring import score_example, summarize
from langchain_agents.core.config import data_dir, get_model, judge_model_id, load_env, model_id
from langchain_agents.core.cost import CostTracker

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASET = "lca-contracts-v1"
ANSWERS = json.loads(data_dir("contracts", "answers.json").read_text())
CASES = {k: v for k, v in ANSWERS.items() if not k.startswith("_")}


def sync_dataset(client: Client, name: str) -> str:
    if client.has_dataset(dataset_name=name):
        return name
    ds = client.create_dataset(dataset_name=name, description="Ridgeline carrier agreements vs. legal playbook (fictional). "
                               "Answer key: fixtures/contracts/answers.json")
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": {"contract": fname}, "outputs": ans, "metadata": {"carrier": ans["carrier"], "compliant": ans["compliant"]}}
        for fname, ans in CASES.items()])
    print(f"Created LangSmith dataset {name!r} with {len(CASES)} examples")
    return name


VARIANT = "baseline"
UPLOAD = False


def _tracing():
    # evaluate(upload_results=False) runs the target under tracing_context(enabled="local"),
    # which LangChain's callback tracer treats as "on" and still POSTs runs (seen as 429s).
    # Force it off for inner model calls unless --upload.
    return rh.tracing_context(enabled=None if UPLOAD else False)


_lock = threading.Lock()
RUN_LOG: dict[str, dict] = {}


def target(inputs: dict) -> dict:
    fname = inputs["contract"]
    with _tracing():
        res = review(fname, decide=auto_approve_all, variant=VARIANT, write=False, tags=["eval"])
    st = res["state"]
    out = {"contract": fname, "carrier": st.get("carrier"), "findings": st.get("findings", []),
           "extraction": {k: v for k, v in (st.get("extraction") or {}).items()},
           "first_pass_problems": st.get("first_pass_problems", {}), "memo_md": st.get("memo_md", ""),
           "errors": st.get("errors", []), "run": res["run"]}
    with _lock:
        RUN_LOG[fname] = out
    return out


def _example_scores(outputs: dict, reference_outputs: dict) -> dict:
    text = data_dir("contracts", outputs["contract"]).read_text()
    return score_example(outputs, reference_outputs, text)


def recall_eval(outputs: dict, reference_outputs: dict) -> dict:
    s = _example_scores(outputs, reference_outputs)
    return {"key": "deviation_recall", "score": (s["tp"] / s["n_expected"]) if s["n_expected"] else None,
            "comment": f"missed: {s['fn']}" if s["fn"] else "all found"}


def precision_eval(outputs: dict, reference_outputs: dict) -> dict:
    s = _example_scores(outputs, reference_outputs)
    return {"key": "deviation_precision", "score": (s["tp"] / s["n_surfaced"]) if s["n_surfaced"] else
            (1.0 if not s["n_expected"] else None), "comment": f"false alarms: {s['fp']}" if s["fp"] else ""}


def value_eval(outputs: dict, reference_outputs: dict) -> dict:
    s = _example_scores(outputs, reference_outputs)
    return {"key": "extraction_value_accuracy", "score": (s["value_hits"] / s["n_values"]) if s["n_values"] else None,
            "comment": json.dumps(s["value_misses"])[:900]}


def quote_eval(outputs: dict, reference_outputs: dict) -> dict:
    s = _example_scores(outputs, reference_outputs)
    return {"key": "quote_verification_rate", "score": (s["verified_quotes"] / s["n_quotes"]) if s["n_quotes"] else None}


def cost_eval(outputs: dict) -> dict:
    return {"key": "cost_usd", "score": outputs["run"]["cost_usd"]}


def latency_eval(outputs: dict) -> dict:
    return {"key": "latency_s", "score": outputs["run"]["latency_s"]}


class Clarity(BaseModel):
    reasoning: str = Field(description="One or two sentences.")
    score: int = Field(description="1-5. 5 = a general counsel could act on it immediately: plain English, each deviation "
                       "explained with its business risk, redlines are concrete clause language. 1 = confusing or wrong.")


JUDGE_COST = CostTracker()
_judge = None


def clarity_eval(outputs: dict, reference_outputs: dict) -> dict:
    """Judge model (a different family from the reviewer) rates memo clarity only.
    Correctness is scored in code above; the judge does not decide compliance."""
    global _judge
    if not reference_outputs.get("deviations") or not outputs.get("memo_md"):
        return {"key": "memo_clarity", "score": None, "comment": "compliant agreement: no memo to grade"}
    with _lock:
        if _judge is None:
            _judge = get_model(judge_model_id()).with_structured_output(Clarity, method="json_schema")
    prompt = ("Grade this contract review memo written for a freight broker's General Counsel. Grade clarity and "
              "usefulness only, not legal correctness.\n\nMEMO:\n" + outputs["memo_md"][:12000])
    try:
        with _tracing():
            r = _judge.invoke(prompt, config={"callbacks": [JUDGE_COST], "run_name": "memo_clarity_judge"})
    except Exception as exc:  # noqa: BLE001 - a judge outage must not fake a score
        return {"key": "memo_clarity", "score": None, "comment": f"judge failed: {exc}"[:300]}
    return {"key": "memo_clarity", "score": r.score / 5, "comment": r.reasoning}


def aggregate_eval(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Micro-averaged metrics on the LangSmith experiment (same math as results JSON)."""
    per = [_example_scores(o, r) for o, r in zip(outputs, reference_outputs) if o and o.get("findings")]
    m = summarize(per)
    keys = ["high_severity_recall", "deviation_recall", "deviation_precision", "extraction_value_accuracy",
            "quote_verification_rate", "grounded_decision_rate", "compliant_false_alarms"]
    return {"results": [{"key": f"agg_{k}", "score": m[k]} for k in keys if m.get(k) is not None]}


def local_examples(repetitions: int = 1) -> list:
    """The dataset as in-memory LangSmith Examples (no API call)."""
    import uuid

    from langsmith.schemas import Example
    ds = uuid.uuid5(uuid.NAMESPACE_URL, DATASET)
    return [Example(id=uuid.uuid5(ds, fname), dataset_id=ds, inputs={"contract": fname}, outputs=ans,
                    metadata={"carrier": ans["carrier"], "compliant": ans["compliant"]},
                    created_at=datetime.now(timezone.utc))
            for fname, ans in CASES.items()]


def main() -> int:
    global VARIANT, UPLOAD
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="baseline", choices=["baseline", "no_verifier", "model_decides"])
    p.add_argument("--prefix", default="contracts")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--upload", action="store_true", help="upload traces + experiment to LangSmith (off by default)")
    args = p.parse_args()
    if not args.upload:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    missing = [k for k in ("OPENROUTER_API_KEY", *(("LANGSMITH_API_KEY",) if args.upload else ())) if not os.getenv(k)]
    if missing:
        print("Missing env vars:", ", ".join(missing))
        return 2
    VARIANT, UPLOAD = args.variant, args.upload
    client = None
    if args.upload:
        client = Client()
        sync_dataset(client, args.dataset)
    data = args.dataset if args.upload else local_examples()
    evaluators = [recall_eval, precision_eval, value_eval, quote_eval, cost_eval, latency_eval]
    if not args.no_judge:
        evaluators.append(clarity_eval)
    t0 = time.perf_counter()
    results = evaluate(target, data=data, upload_results=args.upload, evaluators=evaluators, summary_evaluators=[aggregate_eval],
                       experiment_prefix=f"{args.prefix}-{args.variant}", max_concurrency=args.concurrency,
                       num_repetitions=args.repetitions, client=client,
                       metadata={"model": model_id(), "judge": judge_model_id(), "variant": args.variant})
    wall = time.perf_counter() - t0

    per_example, clarity, rows = [], [], []
    for row in results:
        out = row["run"].outputs or {}
        ref = row["example"].outputs
        if not out.get("findings"):
            print("example failed:", row["example"].inputs, row["run"].error)
            continue
        s = _example_scores(out, ref)
        per_example.append(s)
        for e in row["evaluation_results"]["results"]:
            if e.key == "memo_clarity" and e.score is not None:
                clarity.append(e.score)
        rows.append({"contract": out["contract"], "carrier": out["carrier"], **{k: s[k] for k in (
            "expected", "surfaced", "fn", "fp", "needs_check", "value_misses", "ungrounded_rules", "ungrounded_quotes")},
            "first_pass_problems": out["first_pass_problems"], "errors": out["errors"], "run": out["run"]})
    metrics = summarize(per_example)
    costs = [r["run"]["cost_usd"] for r in rows]
    lats = [r["run"]["latency_s"] for r in rows]
    metrics.update({
        "memo_clarity": round(statistics.mean(clarity), 3) if clarity else None,
        "cost_per_agreement_usd": round(statistics.mean(costs), 5) if costs else None,
        "latency_p50_s": round(statistics.median(lats), 2) if lats else None,
        "latency_max_s": round(max(lats), 2) if lats else None,
        "examples_scored": len(per_example), "examples_total": len(CASES) * args.repetitions,
    })
    retried = sum(1 for r in rows if r["first_pass_problems"])
    record = {
        "experiment": results.experiment_name, "langsmith_upload": args.upload, "variant": args.variant, "model": model_id(), "judge": judge_model_id(),
        "dataset": args.dataset, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics, "agreements_with_first_pass_quote_failures": retried,
        "spend_usd": {"reviewer": round(sum(costs), 5), "judge": round(JUDGE_COST.cost_usd, 5),
                      "total": round(sum(costs) + JUDGE_COST.cost_usd, 5)},
        "wall_s": round(wall, 1), "per_example": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"{args.prefix}-{args.variant}-{ts}.json"
    for pth in (path, RESULTS / "latest.json"):
        pth.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"experiment": record["experiment"], "metrics": metrics, "spend_usd": record["spend_usd"]}, indent=2))
    for r in rows:
        if r["fn"] or r["fp"] or r["value_misses"] or r["ungrounded_rules"]:
            print(f"- {r['contract']}: missed={r['fn']} false_alarms={r['fp']} needs_check={r['needs_check']} "
                  f"value_misses={list(r['value_misses'])} ungrounded={r['ungrounded_rules']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
