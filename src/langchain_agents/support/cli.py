"""langchain-agents support chat [--account ACCT] | demo

chat: interactive terminal chat. Consequential actions stop for YOUR decision
      (approve / edit / reject), exactly like the approval card in agent-chat-ui.
demo: scripted multi-turn run with the real model: a load lookup, a policy answer,
      an action approved by the (scripted) reviewer, a preference saved, and that
      preference recalled in a second, separate thread.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_agents.core.config import model_id
from langchain_agents.core.cost import CostTracker
from langchain_agents.support.agent import build_graph
from langchain_agents.support.middleware import shipper_memories
from langchain_agents.support.session import SupportSession, TurnResult


def _print_turn(res: TurnResult) -> None:
    for tc in res.tool_calls:
        print(f"   [tool] {tc['name']}({json.dumps(tc['args'])})")
    for ev in res.guard_events:
        print(f"   [guard] {ev['action']}: {', '.join(ev['problems'])}")
    print(f"agent> {res.reply}\n   ({res.seconds}s)")


def _parse(value: str, old):
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def terminal_decider(req: dict, cfg: dict) -> dict:
    allowed = cfg.get("allowed_decisions", ["approve", "edit", "reject"])
    print("\n" + "=" * 60 + "\nAPPROVAL NEEDED\n" + req.get("description", req["name"]) + "\n" + "=" * 60)
    while True:
        choice = input(f"[{'/'.join(a[0] for a in allowed)}] {' / '.join(allowed)}? ").strip().lower()
        if choice in ("a", "approve") and "approve" in allowed:
            return {"type": "approve"}
        if choice in ("r", "reject") and "reject" in allowed:
            reason = input("reason (optional): ").strip()
            return {"type": "reject", **({"message": reason} if reason else {})}
        if choice in ("e", "edit") and "edit" in allowed:
            args = dict(req["args"])
            for k, v in args.items():
                new = input(f"  {k} [{v}]: ").strip()
                if new:
                    args[k] = _parse(new, v)
            return {"type": "edit", "edited_action": {"name": req["name"], "args": args}}
        print("Please answer approve, edit or reject.")


def chat(account: str | None) -> int:
    cost = CostTracker()
    store = InMemoryStore()
    graph = build_graph(checkpointer=InMemorySaver(), store=store)
    session = SupportSession(graph, account=account, callbacks=[cost], metadata={"entry": "cli-chat"})
    print(f"Ridgeline support chat ({model_id()}), thread {session.thread_id}"
          f"{', account ' + account if account else ''}.\nCommands: /new (new thread, same memory store), /memory, /quit\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/new":
            acct = session.state().get("account") or account
            session = SupportSession(graph, account=acct, callbacks=[cost], metadata={"entry": "cli-chat"})
            print(f"(new thread {session.thread_id}; long-term memory kept)")
            continue
        if text == "/memory":
            acct = session.state().get("account") or account
            print(json.dumps(shipper_memories(store, acct), indent=1))
            continue
        try:
            _print_turn(session.send(text, decide=terminal_decider))
        except Exception as exc:  # fail closed and visibly
            print(f"[error] {type(exc).__name__}: {exc}\nNothing was sent to the shipper.")
    print(f"\nSpend this session: {cost.summary()}")
    return 0


DEMO_THREAD_A = [
    "Hi, this is Jordan at Tupelo Home Goods, account ACCT-1001. Where is load RL-48213 right now?",
    "How does your detention policy work?",
    "On RL-48251 the truck sat at the receiver for 4.25 hours on Oct 24. Please send that detention to billing for review.",
    "Also, for the future: our dock closes at 3pm and we always need a liftgate at delivery.",
]
DEMO_THREAD_B = ["Hi again. We might push the pickup on RL-48410 to late Friday afternoon. Anything I should know?"]


def _scripted_reviewer(req: dict, cfg: dict) -> dict:
    print(f"\n   [approval node] {req['description'].splitlines()[0]}\n   [scripted reviewer] approve")
    return {"type": "approve"}


def demo() -> int:
    cost = CostTracker()
    store = InMemoryStore()
    graph = build_graph(checkpointer=InMemorySaver(), store=store)
    a = SupportSession(graph, account=None, callbacks=[cost], metadata={"entry": "cli-demo", "demo_thread": "A"})
    print(f"Model: {model_id()}\n\n=== Thread A ({a.thread_id}) ===")
    for text in DEMO_THREAD_A:
        print(f"\nshipper> {text}")
        _print_turn(a.send(text, decide=_scripted_reviewer))
    acct = a.state().get("account")
    print(f"\nActions recorded in thread A: {[x['ref'] + ' ' + x['kind'] for x in a.executed_actions(store)]}")
    print(f"Long-term memory for {acct}: {shipper_memories(store, acct)}")

    b = SupportSession(graph, account=acct, callbacks=[cost], metadata={"entry": "cli-demo", "demo_thread": "B"})
    print(f"\n=== Thread B ({b.thread_id}): new thread, no shared messages, same store ===")
    print(f"Recalled into thread B's system prompt at start: {shipper_memories(store, acct)}")
    for text in DEMO_THREAD_B:
        print(f"\nshipper> {text}")
        res = b.send(text, decide=_scripted_reviewer)
        _print_turn(res)
    # Only the saved memory says the dock closes at 3pm (it is not in the TMS), so mentioning it proves recall.
    recalled = bool(re.search(r"\b3\s?(?::00\s?)?(?:pm|p\.m\.)|\b15:00\b", res.reply, re.I))
    print(f"\nMemory recalled in thread B reply: {'yes' if recalled else 'NO'}")
    print(f"Spend: {cost.summary()}")
    return 0 if recalled else 1


def cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="langchain-agents support", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="interactive terminal chat with approvals")
    c.add_argument("--account", help="bind the conversation to a shipper account, e.g. ACCT-1001")
    sub.add_parser("demo", help="scripted multi-turn demo (real model)")
    args = p.parse_args(argv)
    if args.cmd == "chat":
        return chat(args.account.upper() if args.account else None)
    return demo()


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
