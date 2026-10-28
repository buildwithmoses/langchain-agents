"""Research agent evals, corpus mode. LangSmith `evaluate()` when LCA_RESEARCH_UPLOAD=1, else a local loop
with the same target and evaluators (default since the 2026-10-28 trace-quota 429s).

    uv run python evals/research/run_evals.py                 # all 10 questions
    uv run python evals/research/run_evals.py --limit 3       # quick check
    uv run python evals/research/run_evals.py --prefix haiku-v1

Writes evals/research/results/<prefix>-<ts>.json and latest.json; gate.py reads
latest.json. Needs OPENROUTER_API_KEY and LANGSMITH_API_KEY.
"""

from __future__ import annotations

import os

# 2026-10-28: the LangSmith workspace hit its monthly unique-trace limit (429 on /runs/multipart), so tracing
# and experiment uploads are off unless LCA_RESEARCH_UPLOAD=1. Set before langsmith is imported (it caches it).
UPLOAD = os.getenv("LCA_RESEARCH_UPLOAD", "0") == "1"
if not UPLOAD:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from langsmith import Client, evaluate

from langchain_agents.core.config import judge_model_id, load_env, model_id
from langchain_agents.core.cost import CostTracker
from langchain_agents.research.agent import build_graph
from langchain_agents.research.run import run_question
from langchain_agents.research.scoring import dumps, judge_correctness, judge_support, summarize
from langchain_agents.research.verify import verify_claims

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset.jsonl"
RESULTS = HERE / "results"


def load_cases() -> list[dict]:
    return [json.loads(l) for l in DATASET_FILE.read_text().splitlines() if l.strip()]


def sync_dataset(client: Client, name: str, cases: list[dict]) -> None:
    if client.has_dataset(dataset_name=name):
        return
    ds = client.create_dataset(dataset_name=name, description="Research agent, corpus mode: 8 answerable + 2 unanswerable")
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": c["inputs"], "outputs": c["outputs"], "metadata": {"case_id": c["id"]}} for c in cases])
    print(f"Created LangSmith dataset {name!r} ({len(cases)} examples)")


def main(argv: list[str] | None = None) -> int:
    load_env()
    upload = UPLOAD
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lca-research-corpus-v1")
    p.add_argument("--prefix", default="research-corpus")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args(argv)
    assert args.dataset.startswith("lca-research-"), "LangSmith dataset names must start with lca-research-"

    cases = load_cases()
    client = Client() if UPLOAD else None
    if upload:
        sync_dataset(client, args.dataset, cases)
    graph = build_graph(mode="corpus")  # stateless across invocations: safe to share
    judge_cost = CostTracker()
    rows: list[dict] = []

    def target(inputs: dict) -> dict:
        res = run_question(inputs["question"], mode="corpus", graph=graph, write=False)
        st, meta = res["state"], res["meta"]
        rounds = st.get("rounds") or []
        final = st["final"]
        recheck = verify_claims(final["claims"], st.get("sources") or {})
        first_failed = {c["index"] for c in (rounds[0]["checks"] if rounds else []) if not c["verified"]}
        return {
            "final": final,
            "raw_claims": meta["stats"]["raw_claims"], "raw_verified": meta["stats"]["raw_verified"],
            "raw_failed_by_reason": meta["stats"]["raw_failed_by_reason"],
            "recovered_on_revision": max(0, len(final["claims"]) - meta["stats"]["raw_verified"]) if first_failed else 0,
            "final_claims": len(final["claims"]), "final_reverified": sum(c.verified for c in recheck),
            "final_dropped": len(final["dropped"]), "revisions": meta["stats"]["revisions"],
            "sources_read": sorted(k for k, v in (st.get("sources") or {}).items() if v.get("text")),
            "cost_usd": meta["cost_usd"], "latency_s": meta["latency_s"], "error": st.get("error"),
        }

    def code_metrics(inputs, outputs, reference_outputs):
        n = outputs["final_claims"]
        return [
            {"key": "citation_precision_final", "score": outputs["final_reverified"] / n if n else 1.0},
            {"key": "citation_precision_raw",
             "score": outputs["raw_verified"] / outputs["raw_claims"] if outputs["raw_claims"] else 1.0},
            {"key": "claims_dropped", "score": outputs["final_dropped"]},
        ]

    def judged(inputs, outputs, reference_outputs, example):
        final = outputs["final"]
        judge_error = None
        try:
            support = judge_support(final, callbacks=[judge_cost])
        except Exception as exc:  # fail closed: an unjudged claim counts as unsupported
            support, judge_error = [False] * len(final["claims"]), f"support: {exc!r}"[:300]
        row = {
            "id": (example.metadata or {}).get("case_id"), "question": inputs["question"], "answerable": reference_outputs["answerable"],
            **{k: outputs[k] for k in ("raw_claims", "raw_verified", "raw_failed_by_reason", "recovered_on_revision",
                                       "final_claims", "final_reverified", "final_dropped", "cost_usd", "latency_s",
                                       "error", "sources_read", "revisions")},
            "support": support, "answer": final["answer"],
            "claims": [{**c, "judge_supported": ok} for c, ok in zip(final["claims"], support)],
            "dropped": final["dropped"], "gaps": final["gaps"],
            "expected_source_read": all(s in outputs["sources_read"] for s in reference_outputs["expected_sources"]),
        }
        fb = [{"key": "unsupported_claims", "score": sum(1 for s in support if not s)}]
        if reference_outputs["answerable"]:
            try:
                verdict = judge_correctness(inputs["question"], reference_outputs["reference"], final,
                                            callbacks=[judge_cost])
                score, reason = verdict.score, verdict.reason
            except Exception as exc:  # fail closed: scored 0
                score, reason = 0.0, "judge error"
                judge_error = (judge_error or "") + f" correctness: {exc!r}"[:300]
            row.update(correctness=score, correctness_reason=reason, refused=None)
            fb.append({"key": "answer_correctness", "score": score, "comment": reason})
        else:
            row.update(correctness=None, refused=not final["answerable"])
            fb.append({"key": "unanswerable_handled", "score": float(row["refused"])})
        row["judge_error"] = judge_error
        rows.append(row)
        return fb

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = None
    if upload:
        data = args.dataset if not args.limit else list(client.list_examples(dataset_name=args.dataset, limit=args.limit))
        results = evaluate(target, data=data, evaluators=[code_metrics, judged], experiment_prefix=args.prefix,
                           max_concurrency=args.concurrency,
                           metadata={"agent": "research", "mode": "corpus", "model": model_id(), "judge": judge_model_id()})
    else:  # same target + evaluators, run locally with nothing sent to LangSmith
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        def one(case):
            out = target(case["inputs"])
            code_metrics(case["inputs"], out, case["outputs"])
            judged(case["inputs"], out, case["outputs"], SimpleNamespace(metadata={"case_id": case["id"]}))

        with ThreadPoolExecutor(args.concurrency) as pool:
            list(pool.map(one, cases[: args.limit] if args.limit else cases))
    metrics = summarize(rows)
    metrics["judge_cost_usd"] = judge_cost.summary()["cost_usd"]
    metrics["total_spend_usd"] = round(metrics["cost_usd_total"] + metrics["judge_cost_usd"], 4)
    blob = {"metrics": metrics, "model": model_id(), "judge": judge_model_id(), "dataset": args.dataset,
            "experiment": getattr(results, "experiment_name", None) if upload else None,
            "langsmith_upload": upload, "run_at": ts,
            "rows": sorted(rows, key=lambda r: r["question"])}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.prefix}-{ts}.json").write_text(dumps(blob))
    (RESULTS / "latest.json").write_text(dumps(blob))
    print(dumps(metrics))
    print(f"\nWrote {RESULTS / f'{args.prefix}-{ts}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
