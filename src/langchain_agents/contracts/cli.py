"""langchain-agents contracts ...

    contracts review <file>                     Review an agreement; you approve/edit/reject each redline
    contracts review <file> --auto-approve-for-eval
                                                EVAL ONLY: approves every redline without a human.
                                                Never use for a real review.
    contracts demo                              Review a fixture with a hidden deviation, interactively

<file> is a path, or a fixture name in fixtures/contracts/ (e.g. 07-tristar-express.md).
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from .runner import auto_approve_all, review

DEMO_CONTRACT = "07-tristar-express.md"


def _input(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None


def terminal_decider(payload: dict) -> dict:
    """Shows the memo, then asks for a decision on every redline. Fail closed:
    end of input or anything unrecognised is a reject."""
    print("\n" + "=" * 78)
    print(payload["memo_md"])
    print("=" * 78)
    redlines = payload.get("redlines", [])
    if not redlines:
        print("No redlines proposed. Press Enter to acknowledge the memo.")
        _input("> ")
        return {"decisions": []}
    print(f"\n{len(redlines)} suggested redline(s). For each: [a]pprove, [e]dit, [r]eject.\n")
    decisions = []
    for r in redlines:
        print(f"--- {r['id']} {r['title']} ({r['severity']}) - section {', '.join(r['section_refs']) or '?'}")
        print(textwrap.fill(r["suggested_text"], 100, initial_indent="    ", subsequent_indent="    "))
        while True:
            ans = _input(f"{r['id']} [a/e/r]: ")
            if ans is None:
                print("(no input: rejecting this and any remaining redlines)")
                return {"decisions": decisions}
            ans = ans.strip().lower()
            if ans in ("a", "approve"):
                decisions.append({"id": r["id"], "action": "approve"})
                break
            if ans in ("r", "reject"):
                note = _input("  reason (optional): ") or ""
                decisions.append({"id": r["id"], "action": "reject", "note": note})
                break
            if ans in ("e", "edit"):
                print("  Replacement text; finish with an empty line:")
                lines = []
                while (line := _input("  | ")) not in (None, ""):
                    lines.append(line)
                decisions.append({"id": r["id"], "action": "edit", "text": "\n".join(lines)})
                break
            print("  please type a, e or r")
    return {"decisions": decisions}


def _print_result(result: dict) -> None:
    run, state = result["run"], result["state"]
    devs = [f for f in state["findings"] if f["status"] == "deviation"]
    checks = [f for f in state["findings"] if f["status"] == "needs_check"]
    approved = [k for k, d in state["decisions"].items() if d["action"] in ("approve", "edit")]
    print(f"\n{state.get('carrier')}: {len(devs)} deviation(s), {len(checks)} to check, "
          f"{len(approved)}/{len(state['redlines'])} redline(s) approved.")
    print(f"cost ${run['cost_usd']:.4f} over {run['model_calls']} model call(s), latency {run['latency_s']}s ({run['model']})")
    print(f"wrote {result['out_dir']}/memo.md, findings.json, redlines.md")
    for e in state.get("errors", []):
        print("warning:", e)


def cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="langchain-agents contracts", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="review one carrier agreement")
    r.add_argument("file")
    r.add_argument("--auto-approve-for-eval", action="store_true",
                   help="EVAL ONLY: approve every redline without a human. Never use for a real review.")
    r.add_argument("--variant", default="baseline", choices=["baseline", "no_verifier", "model_decides"],
                   help=argparse.SUPPRESS)
    sub.add_parser("demo", help=f"interactive review of {DEMO_CONTRACT}")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        print(f"Demo: reviewing {DEMO_CONTRACT} (its Section 7 looks mutual; a clause in 16.4 reverses it).")
        result = review(DEMO_CONTRACT, decide=terminal_decider)
    else:
        if args.auto_approve_for_eval:
            print("WARNING: --auto-approve-for-eval approves every redline without a lawyer. Eval use only.",
                  file=sys.stderr)
        decide = auto_approve_all if args.auto_approve_for_eval else terminal_decider
        result = review(args.file, decide=decide, variant=args.variant)
    _print_result(result)
    return 0
