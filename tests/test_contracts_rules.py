"""The playbook rules engine: every rule, pass and fail, with edge cases. Offline."""

import pytest

from langchain_agents.contracts.fakes import compliant_extraction
from langchain_agents.contracts.playbook import (
    HIGH_SEVERITY, RULES, evaluate_rules, normalize_county, normalize_state,
)
from langchain_agents.contracts.schemas import TERM_MODELS

VERIFIED = {t: {"verified": True, "problems": [], "section_refs": ["1"]} for t in TERM_MODELS}


def run(text: str = "", **overrides) -> dict:
    findings = evaluate_rules(compliant_extraction(**overrides), VERIFIED, text)
    return {f.rule_id: f for f in findings}


def status(rule_id: str, text: str = "", **overrides) -> str:
    return run(text, **overrides)[rule_id].status


def test_compliant_baseline_has_no_issues():
    findings = run()
    assert len(findings) == len(RULES) == 13
    assert {f.status for f in findings.values()} == {"compliant"}


def test_high_severity_set():
    assert HIGH_SEVERITY == {"indemnity", "cargo_liability", "liability_cap_own_negligence"}


@pytest.mark.parametrize("direction,expected", [
    ("mutual", "compliant"), ("carrier_indemnifies_ridgeline", "compliant"),
    ("ridgeline_indemnifies_carrier", "deviation"), ("none", "deviation"),
])
def test_indemnity(direction, expected):
    assert status("indemnity", indemnity={"direction": direction}) == expected


def test_indemnity_missing_is_deviation():
    assert status("indemnity", indemnity={"found": False, "quotes": []}) == "deviation"


@pytest.mark.parametrize("limit,released,expected", [
    (100_000, False, "compliant"),      # exactly $100,000 is enough
    (99_999, False, "deviation"),
    (25_000, True, "deviation"),
    (150_000, True, "compliant"),       # released rates are fine if the cap is >= $100k
    (None, False, "compliant"),         # full value, no cap
    (None, True, "needs_check"),        # released rates but no value extracted
])
def test_cargo_liability(limit, released, expected):
    assert status("cargo_liability", cargo_liability={"cargo_limit_usd": limit, "released_rates": released}) == expected


def test_cargo_liability_missing_needs_check():
    assert status("cargo_liability", cargo_liability={"found": False}) == "needs_check"


def test_liability_cap_on_own_negligence():
    assert status("liability_cap_own_negligence", liability_cap={"cap_present": True, "cap_covers_carrier_negligence": True}) == "deviation"
    assert status("liability_cap_own_negligence", liability_cap={"cap_present": True, "cap_covers_carrier_negligence": False}) == "compliant"


def test_liability_cap_absent_uses_keyword_guard():
    absent = {"found": False, "quotes": [], "cap_present": False, "cap_covers_carrier_negligence": False}
    assert status("liability_cap_own_negligence", "Section 1. Services.", liability_cap=absent) == "compliant"
    text = "IN NO EVENT SHALL CARRIER'S AGGREGATE LIABILITY EXCEED THE CHARGES."
    assert status("liability_cap_own_negligence", text, liability_cap=absent) == "needs_check"


@pytest.mark.parametrize("days,expected", [(30, "compliant"), (45, "compliant"), (29, "deviation"), (15, "deviation"),
                                           (None, "needs_check")])
def test_payment_terms(days, expected):
    assert status("payment_terms", payment_terms={"payment_days": days}) == expected


@pytest.mark.parametrize("months,expected", [(12, "compliant"), (24, "compliant"), (11, "deviation"), (6, "deviation"),
                                             (None, "needs_check")])
def test_non_solicit(months, expected):
    assert status("non_solicit", non_solicit={"non_solicit_months": months}) == expected


def test_non_solicit_missing_is_deviation():
    assert status("non_solicit", non_solicit={"found": False, "quotes": [], "non_solicit_months": None}) == "deviation"


@pytest.mark.parametrize("raw,expected", [
    ("Tennessee", "compliant"), ("TN", "compliant"), ("the State of Tennessee", "compliant"), ("Tenn.", "compliant"),
    ("tennessee", "compliant"), ("Delaware", "deviation"), ("DE", "deviation"), ("State of Texas", "deviation"),
    ("Narnia", "needs_check"), (None, "deviation"),
])
def test_governing_law(raw, expected):
    assert status("governing_law", governing_law={"governing_law_state": raw}) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Shelby County, Tennessee", "compliant"), ("Shelby County, TN", "compliant"),
    ("the state or federal courts located in Shelby County, Tennessee", "compliant"),
    ("Davidson County, Tennessee", "deviation"), ("New Castle County, Delaware", "deviation"),
    ("Shelby County, Alabama", "deviation"),  # same county name, wrong state
    ("Memphis", "needs_check"), (None, "deviation"),
])
def test_venue(raw, expected):
    assert status("venue", governing_law={"venue": raw}) == expected


@pytest.mark.parametrize("days,expected,severity", [
    (30, "compliant", "medium"), (31, "deviation", "low"), (45, "deviation", "low"), (60, "deviation", "low"),
    (61, "deviation", "medium"), (90, "deviation", "medium"), (15, "compliant", "medium"),
])
def test_termination_notice(days, expected, severity):
    f = run(termination={"termination_notice_days": days})["termination_notice"]
    assert (f.status, f.severity) == (expected, severity)


def test_termination_one_sided_is_deviation():
    assert status("termination_notice", termination={"either_party": False}) == "deviation"


@pytest.mark.parametrize("auto,days,expected", [
    (True, 30, "compliant"), (True, 31, "deviation"), (True, 60, "deviation"), (True, 90, "deviation"),
    (True, None, "needs_check"), (False, None, "compliant"),
])
def test_auto_renewal(auto, days, expected):
    assert status("auto_renewal", auto_renewal={"auto_renewal": auto, "renewal_cancel_notice_days": days}) == expected


def test_auto_renewal_absent_uses_keyword_guard():
    absent = {"found": False, "quotes": [], "auto_renewal": False, "renewal_cancel_notice_days": None}
    assert status("auto_renewal", "This Agreement remains in effect until terminated.", auto_renewal=absent) == "compliant"
    assert status("auto_renewal", "It shall automatically renew each year.", auto_renewal=absent) == "needs_check"


@pytest.mark.parametrize("rule,field,ok,bad", [
    ("insurance_auto", "auto_liability_usd", 1_000_000, 750_000),
    ("insurance_cargo", "cargo_insurance_usd", 100_000, 50_000),
    ("insurance_gl", "general_liability_usd", 1_000_000, 500_000),
])
def test_insurance_minimums(rule, field, ok, bad):
    assert status(rule, insurance={field: ok}) == "compliant"
    assert status(rule, insurance={field: ok * 2}) == "compliant"
    assert status(rule, insurance={field: bad}) == "deviation"
    assert status(rule, insurance={field: ok - 1}) == "deviation"
    assert status(rule, insurance={field: None}) == "needs_check"


def test_certificate_holder():
    assert status("certificate_holder", insurance={"certificate_holder_named": False}) == "deviation"


def test_unverified_term_is_never_compliant():
    ver = dict(VERIFIED, termination={"verified": False, "problems": ["quote not found"], "section_refs": []})
    f = {x.rule_id: x for x in evaluate_rules(compliant_extraction(), ver, "")}
    assert f["termination_notice"].status == "needs_check"
    assert "lawyer must check" in f["termination_notice"].reason


def test_missing_verification_record_fails_closed():
    f = {x.rule_id: x for x in evaluate_rules(compliant_extraction(), {}, "")}
    assert {x.status for x in f.values()} == {"needs_check"}


def test_state_and_county_normalizers():
    assert normalize_state("TN") == normalize_state("Tennessee") == normalize_state("State of Tennessee") == "TN"
    assert normalize_state("West Virginia") == "WV"
    assert normalize_county("Shelby County, Tennessee") == ("shelby", ["TN"])
    assert normalize_county("courts sitting in Shelby County, TN")[0] == "shelby"
