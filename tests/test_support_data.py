"""Fake TMS, policy search and tool scoping. No model, no keys."""

from __future__ import annotations

import json

import pytest

from langchain_agents.core.config import data_dir
from langchain_agents.support import data


@pytest.mark.parametrize("query,expected", [
    ("how much do you charge for detention", "Detention"),
    ("driver waited 5 hours at the dock", "Detention"),
    ("deadline to file a damage claim", "Claims"),
    ("my cargo was lost", "Claims"),
    ("can you move hazardous chemicals", "Hazmat"),
    ("can I get a refund", "Credits and refunds"),
    ("quick pay fee for carriers", "Payment terms"),
    ("how long is my quote good for", "Quotes"),
    ("truck held overnight", "Layover"),
    ("how often do you do check calls", "Tracking"),
    ("what insurance do carriers need", "Carrier requirements"),
])
def test_policy_search_top_hit(query, expected):
    hits = data.search_policies(query)
    assert hits and hits[0][0].title == expected


def test_policy_text_is_verbatim_from_company_md():
    md = data_dir("ridgeline", "COMPANY.md").read_text().replace("**", "")
    for p in data.policies():
        assert p.text in md, p.title
    det = next(p for p in data.policies() if p.title == "Detention").text
    for fact in ("2 hours free", "$75/hour", "15-minute increments", "$600 per stop", "signed BOL"):
        assert fact in det


def test_code_constants_match_policy_text():
    text = " ".join(p.text for p in data.policies())
    assert f"${int(data.DETENTION_RATE_PER_HOUR)}/hour" in text
    assert f"${int(data.DETENTION_CAP_PER_STOP)} per stop" in text
    assert f"{int(data.DETENTION_FREE_HOURS)} hours free" in text
    assert f"within {data.CLAIM_WINDOW_MONTHS} months" in text
    assert f"valid for {data.QUOTE_VALID_HOURS} hours" in text
    assert f"${data.SPOT_SIGNOFF_OVER_USD:,}" in text


@pytest.mark.parametrize("hours,billable,charge,capped", [
    (1.5, 0.0, 0.0, False), (2.0, 0.0, 0.0, False), (2.2, 0.0, 0.0, False),
    (2.25, 0.25, 18.75, False), (4.25, 2.25, 168.75, False), (10.0, 8.0, 600.0, False),
    (12.0, 10.0, 600.0, True),
])
def test_detention_math(hours, billable, charge, capped):
    est = data.detention_estimate(hours)
    assert est["billable_hours"] == billable
    assert est["estimated_charge_usd"] == charge
    assert est["cap_applied"] is capped


def test_load_views():
    lv = data.load_view(data.loads()["RL-48307"])
    assert lv["check_call_overdue"] is True and lv["eta_update_due_to_shipper"] is True
    assert data.load_view(data.loads()["RL-48213"])["check_call_overdue"] is False
    det = data.load_view(data.loads()["RL-48399"])["detention"]
    assert det["clock_running"] and det["hours_on_site"] == 4.0 and "billing_note" in det


def test_claim_window_and_quotes():
    assert data.claim_window(data.loads()["RL-48122"])["eligible"] is True
    assert data.claim_window(data.loads()["RL-47011"])["eligible"] is False
    assert data.claim_window(data.loads()["RL-48213"])["eligible"] is False  # not delivered
    status = {q: data.quote_view(v)["status"] for q, v in data.quotes().items()}
    assert status == {"Q-77120": "valid", "Q-77098": "expired", "Q-77131": "pending pricing manager sign-off"}


class _RT:
    def __init__(self, account=None, store=None):
        self.state = {"account": account} if account else {}
        self.store = store
        self.config = {"configurable": {"thread_id": "t"}}
        self.tool_call_id = "call-1"


def test_tools_scope_to_bound_account():
    from langchain_agents.support.tools import get_quote_status, lookup_load, lookup_shipper

    own = lookup_load.func("RL-48213", runtime=_RT("ACCT-1001"))
    assert json.loads(own)["carrier"]["name"] == "Cypress Lane Trucking"
    other = lookup_load.func("RL-48307", runtime=_RT("ACCT-1001"))
    assert other.startswith("No load RL-48307 found on account ACCT-1001")
    assert "No shipper" in lookup_shipper.func("ACCT-1002", runtime=_RT("ACCT-1001"))
    assert "No quote" in get_quote_status.func("Q-77098", runtime=_RT("ACCT-1001"))
    # unbound conversation can look up any load (e.g. ops staff using the chat)
    assert "Northstar" in lookup_load.func("rl-48307", runtime=_RT())


def test_search_policy_tool_returns_exact_text():
    from langchain_agents.support.tools import search_policy

    out = json.loads(search_policy.invoke({"query": "claim filing deadline"}))
    assert out[0]["policy"] == "Claims" and "within 9 months of delivery" in out[0]["text"]
