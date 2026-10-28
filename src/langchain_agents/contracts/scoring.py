"""Pure eval scoring against fixtures/contracts/answers.json (unit tested, no keys)."""

from __future__ import annotations

from .playbook import HIGH_SEVERITY, normalize_county, normalize_state
from .verify import quote_in_text

SURFACED = {"deviation", "needs_check"}


def flatten_values(extraction: dict) -> dict:
    """Extraction -> the flat, normalized value keys used by the answer key."""
    if not extraction:
        return {}
    g = lambda term, field: (extraction.get(term) or {}).get(field)  # noqa: E731
    venue_county, _ = normalize_county(g("governing_law", "venue"))
    renew = bool(g("auto_renewal", "auto_renewal")) and bool(g("auto_renewal", "found"))
    return {
        "indemnity_direction": g("indemnity", "direction") if g("indemnity", "found") else "none",
        "cargo_limit_usd": g("cargo_liability", "cargo_limit_usd"),
        "released_rates": bool(g("cargo_liability", "released_rates")),
        "payment_days": g("payment_terms", "payment_days"),
        "non_solicit_months": g("non_solicit", "non_solicit_months"),
        "governing_law_state": normalize_state(g("governing_law", "governing_law_state")),
        "venue_county": venue_county,
        "termination_notice_days": g("termination", "termination_notice_days"),
        "auto_renewal": renew,
        "renewal_cancel_notice_days": g("auto_renewal", "renewal_cancel_notice_days") if renew else None,
        "auto_liability_usd": g("insurance", "auto_liability_usd"),
        "cargo_insurance_usd": g("insurance", "cargo_insurance_usd"),
        "general_liability_usd": g("insurance", "general_liability_usd"),
        "certificate_holder_named": bool(g("insurance", "certificate_holder_named")),
        "cap_covers_carrier_negligence": bool(g("liability_cap", "cap_covers_carrier_negligence")),
    }


def _eq(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def score_example(outputs: dict, answer: dict, contract_text: str) -> dict:
    """Per-agreement counts. `outputs` = {"findings": [...], "extraction": {...}}."""
    findings = outputs.get("findings") or []
    status = {f["rule_id"]: f["status"] for f in findings}
    expected = set(answer.get("deviations") or [])
    surfaced = {r for r, s in status.items() if s in SURFACED}
    deviations = {r for r, s in status.items() if s == "deviation"}
    exp_high = expected & HIGH_SEVERITY

    values = flatten_values(outputs.get("extraction") or {})
    key = answer.get("values") or {}
    value_hits = [k for k in key if values and _eq(values.get(k), key[k])]
    value_misses = {k: {"expected": key[k], "got": values.get(k)} for k in key if values and k not in value_hits}

    # Independent quote check: every quote backing a compliant/deviation decision must be in the text.
    decided = [f for f in findings if f["status"] in ("compliant", "deviation")]
    quotes = [(f["rule_id"], q) for f in decided for q in f.get("quotes") or []]
    bad_quotes = [(r, q) for r, q in quotes if not quote_in_text(q, contract_text)]
    ungrounded = sorted({r for r, _ in bad_quotes})

    return {
        "expected": sorted(expected), "surfaced": sorted(surfaced),
        "tp": len(expected & surfaced), "tp_strict": len(expected & deviations),
        "fn": sorted(expected - surfaced), "fp": sorted(surfaced - expected),
        "n_expected": len(expected), "n_surfaced": len(surfaced),
        "high_tp": len(exp_high & surfaced), "n_high": len(exp_high),
        "needs_check": sorted(r for r, s in status.items() if s == "needs_check"),
        "value_hits": len(value_hits), "n_values": len(key) if values else 0, "value_misses": value_misses,
        "n_quotes": len(quotes), "verified_quotes": len(quotes) - len(bad_quotes), "ungrounded_rules": ungrounded,
        "ungrounded_quotes": [{"rule_id": r, "quote": q[:300]} for r, q in bad_quotes],
        "n_decided": len(decided),
        "false_alarms_on_compliant": len(surfaced) if answer.get("compliant") else 0,
        "is_compliant_doc": bool(answer.get("compliant")),
    }


def summarize(per_example: list[dict]) -> dict:
    s = lambda k: sum(e[k] for e in per_example)  # noqa: E731
    ratio = lambda a, b: (a / b) if b else None  # noqa: E731
    compliant = [e for e in per_example if e["is_compliant_doc"]]
    return {
        "deviation_recall": ratio(s("tp"), s("n_expected")),
        "deviation_recall_strict": ratio(s("tp_strict"), s("n_expected")),
        "high_severity_recall": ratio(s("high_tp"), s("n_high")),
        "deviation_precision": ratio(s("tp"), s("n_surfaced")),
        "extraction_value_accuracy": ratio(s("value_hits"), s("n_values")),
        "quote_verification_rate": ratio(s("verified_quotes"), s("n_quotes")),
        "grounded_decision_rate": ratio(s("n_decided") - sum(len(e["ungrounded_rules"]) for e in per_example), s("n_decided")),
        "compliant_false_alarms": sum(e["false_alarms_on_compliant"] for e in compliant),
        "compliant_clean": 1.0 if compliant and all(e["false_alarms_on_compliant"] == 0 for e in compliant) else 0.0,
        "needs_check_total": sum(len(e["needs_check"]) for e in per_example),
    }
