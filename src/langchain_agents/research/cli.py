"""langchain-agents research ask "<question>" [--corpus] [--out DIR] [--model ID]"""

from __future__ import annotations

import argparse
import json


def cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="langchain-agents research", description="Research brief with verified citations")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="research one question")
    ask.add_argument("question")
    ask.add_argument("--corpus", action="store_true", help="use the offline fictional corpus instead of the web")
    ask.add_argument("--out", help="output folder (default runs/research-<ts>/)")
    ask.add_argument("--model", help="init_chat_model id, e.g. openrouter:anthropic/claude-haiku-5.5")
    ask.add_argument("--max-searches", type=int, default=4)
    ask.add_argument("--max-sources", type=int, default=6)
    args = p.parse_args(argv)

    from .run import run_question

    res = run_question(args.question, mode="corpus" if args.corpus else "web", out_dir=args.out, model=args.model,
                       max_searches=args.max_searches, max_sources=args.max_sources)
    final, meta = res["state"]["final"], res["meta"]
    print(final["answer"])
    print()
    print(json.dumps({"stats": meta["stats"], "cost_usd": meta["cost_usd"], "latency_s": meta["latency_s"]}, indent=1))
    print(f"\nBrief: {res['paths'].get('brief')}")
    return 0 if final["claims"] or not final["answerable"] else 1
