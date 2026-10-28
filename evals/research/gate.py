"""Research eval gate (core/gate.py shape: hard floors + no drop below baseline).

    uv run python evals/research/gate.py                       # latest.json vs baseline.json
    uv run python evals/research/gate.py --candidate results/X.json
    uv run python evals/research/gate.py --promote             # on pass, latest becomes baseline

Floors that can't average away:
  citation_precision_final = 1.0   every quote in a shipped brief is in a source we read
  unanswerable_handled     = 1.0   the agent says "not in the sources" instead of inventing
  answer_correctness      >= 0.75
Lower-is-better metrics are gated as ceilings (unsupported_claim_rate <= 0.30, judge-based).
Exit 0 pass, 1 fail, 2 missing files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_agents.core.gate import check, load_baseline, report

HERE = Path(__file__).resolve().parent
FLOORS = {"citation_precision_final": 1.0, "unanswerable_handled": 1.0, "answer_correctness": 0.75}
TRACKED = ["answer_correctness", "expected_source_read"]
# Judge-based and noisy (0.136 vs 0.254 on two runs of near-identical output), so a loose ceiling, not a floor.
CEILINGS = {"unsupported_claim_rate": 0.30}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, default=HERE / "results" / "latest.json")
    p.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    p.add_argument("--max-drop", type=float, default=0.10)
    p.add_argument("--promote", action="store_true")
    args = p.parse_args(argv)
    if not args.candidate.exists():
        print(f"No results at {args.candidate}. Run evals/research/run_evals.py first.")
        return 2
    metrics = json.loads(args.candidate.read_text())["metrics"]
    base_blob = load_baseline(args.baseline)
    baseline = {k: base_blob["metrics"][k] for k in TRACKED if base_blob and base_blob["metrics"].get(k) is not None}
    fails = check({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, FLOORS, baseline, args.max_drop)
    for name, ceiling in CEILINGS.items():
        if metrics.get(name) is not None and metrics[name] > ceiling:
            fails.append(f"{name}: {metrics[name]:.3f} > ceiling {ceiling:.3f}")
    for k in [*FLOORS, *CEILINGS, "citation_precision_raw", "final_dropped", "cost_usd_per_question", "latency_p50_s"]:
        print(f"  {k:28s} {metrics.get(k)}")
    code = report(fails, "research")
    if code == 0 and args.promote:
        args.baseline.write_text(args.candidate.read_text())
        print(f"Promoted {args.candidate.name} -> {args.baseline.name}")
    return code


if __name__ == "__main__":
    sys.exit(main())
