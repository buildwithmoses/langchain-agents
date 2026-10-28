from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..core.config import data_dir


def _out(arg: str | None) -> Path:
    p = Path(arg) if arg else Path("runs") / f"evalkit-{datetime.now():%Y%m%d-%H%M%S}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="langchain-agents evalkit",
                                description="Evals before the POC ships: mine traces, review, run, gate, monitor")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="Pull real traces, flag the interesting ones, label them for review")
    m.add_argument("--project", default="langchain-agents")
    m.add_argument("--name", help="Only root runs with this name")
    m.add_argument("--filter", help="LangSmith trace filter, e.g. 'eq(status, \"error\")'")
    m.add_argument("--days", type=float, default=7)
    m.add_argument("-n", type=int, default=30, help="How many candidates to keep")
    m.add_argument("--context", default=str(data_dir("ridgeline", "COMPANY.md")))
    m.add_argument("--out")

    b = sub.add_parser("build", help="Review candidates and write accepted ones to a LangSmith dataset")
    b.add_argument("candidates")
    b.add_argument("--dataset", required=True)
    b.add_argument("--accept-all", action="store_true", help="Non-interactive (CI/demo). A person should still read the set.")

    r = sub.add_parser("run", help="Run an agent against a dataset as a LangSmith experiment")
    r.add_argument("--target", required=True, help="module:factory returning a compiled graph")
    r.add_argument("--dataset", required=True)
    r.add_argument("--prefix", default="evalkit")
    r.add_argument("--local", action="store_true", help="Score locally, upload nothing (e.g. when over the LangSmith trace quota)")
    r.add_argument("--save-baseline", action="store_true")
    r.add_argument("--baseline", default="evals/evalkit/baseline.json")
    r.add_argument("--out")

    mo = sub.add_parser("monitor", help="Create a LangSmith rule: failing production traces flow into the dataset")
    mo.add_argument("--project", default="langchain-agents")
    mo.add_argument("--dataset", required=True)
    mo.add_argument("--apply", action="store_true")

    a = p.parse_args(argv)

    if a.cmd == "mine":
        from .mine import mine, write_jsonl
        from .triage import triage_all

        cands, stats = mine(a.project, n=a.n, days=a.days, name=a.name, filter=a.filter)
        print(f"pulled {stats['runs_pulled']} root runs, {stats['after_dedupe']} unique, sampled {stats['sampled']}; "
              f"signals {stats['signals']}")
        out = _out(a.out)
        write_jsonl(cands, out / "raw.jsonl")
        items, cost = triage_all(cands, Path(a.context).read_text() if Path(a.context).exists() else "")
        with open(out / "candidates.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it, default=str) + "\n")
        (out / "stats.json").write_text(json.dumps({**stats, "triage": cost}, indent=2))
        flagged = sum(1 for it in items if (it.get("triage") or {}).get("looks_wrong"))
        print(f"triaged {len(items)} (model flagged {flagged} as looking wrong) for ${cost['cost_usd']:.4f}")
        print(f"next: langchain-agents evalkit build {out / 'candidates.jsonl'} --dataset lca-evalkit-<agent>")
        return 0

    if a.cmd == "build":
        from .dataset import upsert
        from .review import accept_all, review

        items = [json.loads(line) for line in Path(a.candidates).read_text().splitlines() if line.strip()]
        examples = accept_all(items) if a.accept_all else review(items)
        if not examples:
            print("Nothing accepted. No dataset written.")
            return 1
        print(json.dumps(upsert(a.dataset, examples), indent=2))
        return 0

    if a.cmd == "run":
        from ..core.gate import check, load_baseline, report
        from .experiment import run

        res = run(a.target, a.dataset, a.prefix, local=a.local)
        out = _out(a.out)
        (out / "experiment.json").write_text(json.dumps(res, indent=2))
        print(f"{res['experiment']}: {res['examples']} examples -> {json.dumps(res['metrics'])}")
        for f in res["failures"][:8]:
            print(f"  FAIL: {f['input'][:90]!r}\n        {f['comment'][:200]}")
        if a.save_baseline:
            # A set mined from this agent's own traces should score well on this agent.
            # If it doesn't, the labels are suspect, not the agent (v1 scored 0.2 on
            # rubrics copied from one reply's wording). Don't bless that as a baseline.
            if res["metrics"].get("rubric", 1.0) < 0.8:
                print(f"WARNING: rubric {res['metrics']['rubric']} on the agent the set was mined from. "
                      "Re-check the examples before saving this as a baseline. Not saved.")
                return 1
            Path(a.baseline).parent.mkdir(parents=True, exist_ok=True)
            Path(a.baseline).write_text(json.dumps(res["metrics"], indent=2))
            print(f"saved baseline -> {a.baseline}")
            return 0
        base = load_baseline(a.baseline)
        floors = {"no_error": 1.0}
        if "paused_when_required" in res["metrics"]:
            floors["paused_when_required"] = 1.0
        fails = check(res["metrics"], floors=floors,
                      baseline={k: v for k, v in (base or {}).items() if k in ("rubric", "answered")}, max_drop=0.05)
        return report(fails, res["experiment"])

    if a.cmd == "monitor":
        from .monitor import create_rule, describe, plan

        pl = plan(a.project, a.dataset)
        print(describe(pl))
        if not a.apply:
            print("\nDry run. Re-run with --apply to create this rule in LangSmith.")
            return 0
        if input("Create this rule? Type yes: ").strip() != "yes":
            print("Not created.")
            return 1
        print(json.dumps(create_rule(pl["body"]), indent=2)[:600])
        return 0
    return 2
