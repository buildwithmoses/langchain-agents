"""Eval scoring and the gate, offline."""

import importlib.util
import json
from pathlib import Path

from langchain_agents.contracts.fakes import compliant_extraction
from langchain_agents.contracts.playbook import evaluate_rules
from langchain_agents.contracts.schemas import TERM_MODELS
from langchain_agents.contracts.scoring import flatten_values, score_example, summarize

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = json.loads((ROOT / "fixtures" / "contracts" / "answers.json").read_text())
TEXT01 = (ROOT / "fixtures" / "contracts" / "01-bluegrass-haulers.md").read_text()
VERIFIED = {t: {"verified": True, "problems": [], "section_refs": []} for t in TERM_MODELS}


def _outputs(extraction):
    return {"findings": [f.to_dict() for f in evaluate_rules(extraction, VERIFIED, TEXT01)], "extraction": extraction}


def test_answer_key_is_consistent():
    from langchain_agents.contracts.playbook import RULES
    cases = {k: v for k, v in ANSWERS.items() if not k.startswith("_")}
    assert len(cases) == 8
    assert sum(v["compliant"] for v in cases.values()) == 1
    for name, v in cases.items():
        assert (ROOT / "fixtures" / "contracts" / name).is_file()
        assert set(v["deviations"]) <= set(RULES), name
        assert v["compliant"] == (not v["deviations"])


def test_fake_compliant_extraction_matches_answer_key_values():
    low = lambda d: {k: v.lower() if isinstance(v, str) else v for k, v in d.items()}  # noqa: E731
    assert low(flatten_values(compliant_extraction())) == low(ANSWERS["01-bluegrass-haulers.md"]["values"])


def test_perfect_compliant_review_scores_clean():
    s = score_example(_outputs(compliant_extraction()), ANSWERS["01-bluegrass-haulers.md"], TEXT01)
    assert s["false_alarms_on_compliant"] == 0 and s["value_hits"] == s["n_values"] == 15
    assert s["verified_quotes"] == s["n_quotes"] > 0
    m = summarize([s])
    assert m["compliant_clean"] == 1.0 and m["deviation_recall"] is None


def test_false_alarm_on_compliant_and_ungrounded_quote_are_counted():
    ext = compliant_extraction(payment_terms={"payment_days": 15, "quotes": ["Ridgeline pays in fifteen (15) days."]})
    s = score_example(_outputs(ext), ANSWERS["01-bluegrass-haulers.md"], TEXT01)
    assert s["fp"] == ["payment_terms"] and s["false_alarms_on_compliant"] == 1
    assert s["ungrounded_rules"] == ["payment_terms"]
    m = summarize([s])
    assert m["compliant_clean"] == 0.0 and m["grounded_decision_rate"] < 1.0


def test_recall_counts_needs_check_as_surfaced_but_strict_does_not():
    ans = {"deviations": ["payment_terms"], "values": {}, "compliant": False}
    out = {"findings": [{"rule_id": "payment_terms", "status": "needs_check", "quotes": []}], "extraction": {}}
    m = summarize([score_example(out, ans, TEXT01)])
    assert m["deviation_recall"] == 1.0 and m["deviation_recall_strict"] == 0.0


def _gate_module():
    spec = importlib.util.spec_from_file_location("contracts_gate", ROOT / "evals" / "contracts" / "gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = {"high_severity_recall": 1.0, "compliant_clean": 1.0, "grounded_decision_rate": 1.0, "deviation_recall": 1.0,
        "deviation_precision": 1.0, "extraction_value_accuracy": 1.0, "quote_verification_rate": 1.0,
        "memo_clarity": 0.8, "cost_per_agreement_usd": 0.004, "examples_scored": 8, "examples_total": 8}


def test_gate_passes_good_and_blocks_regressions():
    gate = _gate_module().gate
    base = {"metrics": GOOD}
    assert gate({"metrics": GOOD}, base) == []
    assert any("high_severity_recall" in f for f in gate({"metrics": {**GOOD, "high_severity_recall": 5 / 6}}, base))
    assert any("compliant_clean" in f for f in gate({"metrics": {**GOOD, "compliant_clean": 0.0}}, base))
    assert any("deviation_precision" in f for f in gate({"metrics": {**GOOD, "deviation_precision": 0.95}}, base))
    assert any("cost" in f for f in gate({"metrics": {**GOOD, "cost_per_agreement_usd": 0.05}}, base))
    assert any("examples scored" in f for f in gate({"metrics": {**GOOD, "examples_scored": 7}}, base))
