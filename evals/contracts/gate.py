"""Gate for the contract review agent (uses core/gate.py).

    uv run python evals/contracts/gate.py                     # check results/latest.json
    uv run python evals/contracts/gate.py results/X.json      # check a specific run
    uv run python evals/contracts/gate.py --promote           # make latest the baseline (only if it passes)

Hard floors can't average away: every high-severity deviation found, zero false
alarms on the compliant agreement, and every compliant/deviation decision backed by a
quote that is really in the contract. Other metrics may not drop more than 0.02
below the baseline. Exit 1 blocks the change.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from langchain_agents.core.gate import check, load_baseline, report

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.json"
FLOORS = {
    "high_severity_recall": 1.0,
    "compliant_clean": 1.0,
    "grounded_decision_rate": 1.0,
    "deviation_recall": 0.95,
    "deviation_precision": 0.85,
    "extraction_value_accuracy": 0.90,
}
TRACKED = ["deviation_recall", "deviation_precision", "extraction_value_accuracy", "quote_verification_rate", "memo_clarity"]
MAX_COST_PER_AGREEMENT = 0.02
MAX_DROP = 0.02  # one false alarm or one missed deviation (~0.05 on 19 items) must fail


def gate(record: dict, baseline: dict | None) -> list[str]:
    m = {k: v for k, v in record["metrics"].items() if isinstance(v, (int, float))}
    base = {k: v for k, v in ((baseline or {}).get("metrics") or {}).items() if k in TRACKED and isinstance(v, (int, float))}
    fails = check(m, FLOORS, base, MAX_DROP)
    if record["metrics"].get("examples_scored") != record["metrics"].get("examples_total"):
        fails.append(f"only {record['metrics'].get('examples_scored')}/{record['metrics'].get('examples_total')} examples scored")
    cost = record["metrics"].get("cost_per_agreement_usd")
    if cost is not None and cost > MAX_COST_PER_AGREEMENT:
        fails.append(f"cost_per_agreement_usd: {cost:.4f} > ceiling {MAX_COST_PER_AGREEMENT:.4f}")
    return fails


def main(argv: list[str]) -> int:
    promote = "--promote" in argv
    args = [a for a in argv if not a.startswith("--")]
    path = Path(args[0]) if args else HERE / "results" / "latest.json"
    if not path.is_absolute() and not path.exists():
        path = HERE / path
    record = json.loads(path.read_text())
    fails = gate(record, load_baseline(BASELINE))
    print(f"{record['experiment']} ({record['variant']}, {record['model']})")
    for k in ["high_severity_recall", "deviation_recall", "deviation_precision", "extraction_value_accuracy",
              "quote_verification_rate", "grounded_decision_rate", "compliant_false_alarms", "memo_clarity",
              "cost_per_agreement_usd", "latency_p50_s"]:
        print(f"  {k:28s} {record['metrics'].get(k)}")
    code = report(fails, label=f"contracts/{record['variant']}")
    if promote:
        if code:
            print("not promoting a failing run")
        else:
            shutil.copy(path, BASELINE)
            print(f"promoted {path.name} -> baseline.json")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
