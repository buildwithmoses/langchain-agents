"""Human review of triaged candidates. Only what a person accepts becomes an
eval example. Terminal flow: a = accept, e = edit the rubric, r = reject, q = stop."""

from __future__ import annotations

import json
from typing import Callable


def to_example(item: dict) -> dict:
    t = item.get("triage") or {}
    return {
        "inputs": {**(item.get("extras") or {}),
                   "messages": item.get("history") or [{"role": "user", "content": item["input_text"]}]},
        "outputs": {"must_do": t.get("must_do", []), "must_not": t.get("must_not", []),
                    "reference_response": None},
        "metadata": {"source": "trace_mining", "source_run_id": item["run_id"], "category": t.get("category"),
                     "signals": item.get("signals", []), "looked_wrong": t.get("looks_wrong")},
    }


def review(items: list[dict], ask: Callable[[str], str] = input, show: Callable[[str], None] = print) -> list[dict]:
    accepted = []
    for i, item in enumerate(items, 1):
        t = item.get("triage") or {}
        show(f"\n[{i}/{len(items)}] {t.get('category', '?')}  signals={item.get('signals')}  looks_wrong={t.get('looks_wrong')}")
        show(f"USER: {item['input_text'][:500]}")
        show(f"AGENT: {item['output_text'][:500]}")
        show(f"why: {t.get('why', item.get('triage_error', ''))}")
        show("must_do: " + json.dumps(t.get("must_do", [])))
        show("must_not: " + json.dumps(t.get("must_not", [])))
        choice = (ask("accept / edit / reject / quit [a/e/r/q]: ") or "").strip().lower()[:1]
        if choice == "q":
            break
        if choice == "e":
            raw = ask("new must_do as JSON list (blank keeps): ").strip()
            if raw:
                t["must_do"] = json.loads(raw)
            raw = ask("new must_not as JSON list (blank keeps): ").strip()
            if raw:
                t["must_not"] = json.loads(raw)
            item["triage"] = t
            choice = "a"
        if choice == "a" and t.get("must_do"):
            accepted.append(to_example(item))
    return accepted


def accept_all(items: list[dict], only_flagged: bool = False) -> list[dict]:
    """Non-interactive path for CI and demos. Documented as such: a person should
    still read the examples before trusting the set."""
    out = []
    for item in items:
        t = item.get("triage") or {}
        if not t.get("must_do"):
            continue
        if only_flagged and not (item.get("signals") or t.get("looks_wrong")):
            continue
        out.append(to_example(item))
    return out
