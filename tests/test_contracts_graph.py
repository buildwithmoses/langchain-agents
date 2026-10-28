"""Graph wiring with a scripted fake model: lawyer interrupt, approve/edit/reject,
approved-only redlines, quote-retry then fail-closed, runner output. Offline."""

import json
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_agents.contracts.fakes import FakeLLM, compliant_extraction
from langchain_agents.contracts.pipeline import build_graph, normalize_decisions
from langchain_agents.contracts.runner import auto_approve_all, review

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
BASE = (FIX / "01-bluegrass-haulers.md").read_text()

# Plant two deviations into the compliant agreement's text, and a fake extraction that reports them.
TEXT = (BASE.replace("within thirty (30) days after receipt of the invoice", "within fifteen (15) days after receipt of the invoice")
            .replace("located in Shelby County, Tennessee", "located in New Castle County, Delaware"))
DEVIATING = compliant_extraction(
    payment_terms={"payment_days": 15, "quotes": ["Ridgeline shall pay each undisputed invoice within fifteen (15) days after receipt of the invoice"]},
    governing_law={"venue": "New Castle County, Delaware", "quotes": [
        "This Agreement shall be governed by the laws of the State of Tennessee",
        "shall be brought exclusively in the state or federal courts located in New Castle County, Delaware."]},
)


def start(llm, text=TEXT):
    g = build_graph(llm, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    res = g.invoke({"contract_text": text, "contract_name": "test.md"}, cfg)
    return g, cfg, res


def test_graph_pauses_for_lawyer_with_memo_and_redlines():
    g, cfg, res = start(FakeLLM(DEVIATING))
    assert "__interrupt__" in res
    payload = res["__interrupt__"][0].value
    assert payload["type"] == "lawyer_review"
    assert [r["rule_id"] for r in payload["redlines"]] == ["payment_terms", "venue"]
    assert "Payment terms" in payload["memo_md"] and "DEVIATION" in payload["memo_md"]
    assert g.get_state(cfg).next == ("lawyer_review",)  # nothing finalized before the lawyer decides
    assert not g.get_state(cfg).values.get("redlines_md")


def test_only_approved_and_edited_redlines_reach_redlines_md():
    g, cfg, res = start(FakeLLM(DEVIATING))
    ids = {r["rule_id"]: r["id"] for r in res["__interrupt__"][0].value["redlines"]}
    out = g.invoke(Command(resume={"decisions": [
        {"id": ids["payment_terms"], "action": "edit", "text": "Ridgeline shall pay within thirty (30) days. [lawyer edit]"},
        {"id": ids["venue"], "action": "reject", "note": "accept Delaware venue for this carrier"},
    ]}), cfg)
    md = out["redlines_md"]
    assert "[lawyer edit]" in md and "edited by lawyer" in md
    assert "Fake redline text for venue" not in md and "Venue" not in md and "New Castle" not in md
    assert "1 approved, 1 rejected" in md


def test_missing_and_invalid_decisions_fail_closed_to_reject():
    g, cfg, res = start(FakeLLM(DEVIATING))
    out = g.invoke(Command(resume={"decisions": [{"id": "R1", "action": "yolo"}]}), cfg)
    assert out["decisions"]["R1"]["action"] == "reject"
    assert out["decisions"]["R2"]["action"] == "reject"
    assert "No redlines were approved." in out["redlines_md"]


def test_normalize_decisions_shapes():
    reds = [{"id": "R1"}, {"id": "R2"}, {"id": "R3"}]
    d = normalize_decisions({"decisions": {"R1": "approve", "R2": {"action": "edit", "text": ""}}}, reds)
    assert d["R1"]["action"] == "approve"
    assert d["R2"]["action"] == "reject"  # edit with no text
    assert d["R3"]["action"] == "reject"  # no decision
    assert all(v["action"] == "reject" for v in normalize_decisions("garbage", reds).values())


def test_compliant_contract_has_no_findings_and_no_memo_call():
    llm = FakeLLM()
    g, cfg, res = start(llm, BASE)
    payload = res["__interrupt__"][0].value
    assert payload["redlines"] == []
    state = g.get_state(cfg).values
    assert {f["status"] for f in state["findings"]} == {"compliant"}
    assert [c["name"] for c in llm.calls if c["name"] == "draft_memo"] == []
    assert len([c for c in llm.calls if c["name"].startswith("extract_terms")]) == 3  # three term groups


def test_bad_quote_is_retried_once_then_marked_lawyer_must_check():
    bad = compliant_extraction(termination={"termination_notice_days": 30, "quotes": ["Either party may end this deal on 30 days notice."]})
    llm = FakeLLM(bad)  # the retry returns the same bad quote
    g, cfg, res = start(llm, BASE)
    retries = [c for c in llm.calls if c["name"].startswith("reextract")]
    assert len(retries) == 1 and retries[0]["fields"] == ["termination"]
    state = g.get_state(cfg).values
    f = {x["rule_id"]: x for x in state["findings"]}
    assert f["termination_notice"]["status"] == "needs_check"
    assert "lawyer must check" in f["termination_notice"]["reason"].lower()
    assert "termination" in state["first_pass_problems"]
    assert "Not verified - lawyer must check" in state["memo_md"]


def test_retry_that_fixes_the_quote_is_accepted():
    bad = compliant_extraction(termination={"quotes": ["Either party may end this deal on 30 days notice."]})
    llm = FakeLLM(bad, retry_extraction=compliant_extraction())
    g, cfg, res = start(llm, BASE)
    f = {x["rule_id"]: x for x in g.get_state(cfg).values["findings"]}
    assert f["termination_notice"]["status"] == "compliant"
    assert f["termination_notice"]["section_refs"] == ["11.2"]  # located by code, not taken from the model


def test_extraction_failure_fails_closed():
    llm = FakeLLM(fail_names=("extract_terms_2",))  # payment / non-solicit / governing-law group errors
    g, cfg, res = start(llm, BASE)
    f = {x["rule_id"]: x for x in g.get_state(cfg).values["findings"]}
    for rid in ("payment_terms", "non_solicit", "governing_law", "venue"):
        assert f[rid]["status"] == "needs_check", rid
    assert f["indemnity"]["status"] == "compliant"
    assert any("failed" in e for e in g.get_state(cfg).values["errors"])


def test_runner_writes_run_folder(tmp_path):
    result = review(FIX / "01-bluegrass-haulers.md", decide=auto_approve_all, llm=FakeLLM(), out_root=tmp_path)
    out = Path(result["out_dir"])
    assert out.name.startswith("contracts-")
    assert {p.name for p in out.iterdir()} == {"memo.md", "findings.json", "redlines.md"}
    data = json.loads((out / "findings.json").read_text())
    assert {"cost_usd", "latency_s", "model_calls"} <= set(data["run"])
    assert len(data["findings"]) == 13


def test_graph_module_imports_without_keys(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import importlib

    import langchain_agents.contracts.graph as gmod
    importlib.reload(gmod)
    assert {"lawyer_review", "verify", "reextract", "rules"} <= set(gmod.graph.get_graph().nodes)
