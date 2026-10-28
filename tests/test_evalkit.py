"""Eval kit unit tests. Offline: no LangSmith, no model."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from langchain_agents.core.gate import check
from langchain_agents.evalkit import experiment, mine, monitor, review

T0 = datetime(2026, 10, 28, tzinfo=timezone.utc)


def run(i, text, *, secs=2.0, error=None, out="ok", interrupt=False):
    outputs = {"messages": [{"type": "human", "content": text}, {"type": "ai", "content": out}]}
    if interrupt:
        outputs["__interrupt__"] = [{"value": {"action": "file_claim_intake"}}]
    return SimpleNamespace(id=f"r{i}", trace_id=f"t{i}", name="LangGraph", start_time=T0 + timedelta(minutes=i),
                           end_time=T0 + timedelta(minutes=i, seconds=secs), error=error,
                           inputs={"messages": [{"type": "human", "content": text}]}, outputs=outputs)


def test_candidate_extracts_first_human_and_last_ai():
    c = mine.to_candidate(run(1, "Where is load RF-1001?", out="In transit, ETA 4pm"))
    assert c.input_text == "Where is load RF-1001?"
    assert c.output_text == "In transit, ETA 4pm"
    assert c.latency_s == 2.0


def test_signals():
    cands = [mine.to_candidate(r) for r in [
        run(1, "a"), run(2, "b"), run(3, "c"), run(4, "slow one", secs=30),
        run(5, "boom", error="TimeoutError"), run(6, "claim please", interrupt=True), run(7, "blank", out=""),
    ]]
    cands[0].feedback = {"user_score": 0}
    got = {c.input_text: c.signals for c in mine.score_signals(cands)}
    assert got["slow one"] == ["slow"]
    assert "error" in got["boom"]
    assert got["claim please"] == ["human_interrupt"]
    assert got["blank"] == ["empty_output"]
    assert got["a"] == ["negative_feedback"]
    assert got["b"] == []


def test_dedupe_keeps_flagged_copy_of_near_duplicates():
    cands = [mine.to_candidate(run(1, "where is my load RF-1001")),
             mine.to_candidate(run(2, "Where is my load RF-1001?", error="x")),
             mine.to_candidate(run(3, "what is the detention rate"))]
    mine.score_signals(cands)
    kept = mine.dedupe(cands)
    assert len(kept) == 2
    assert any(c.run_id == "r2" for c in kept)   # the errored copy wins


def test_stratified_sample_takes_flagged_first_then_spreads():
    cands = [mine.to_candidate(run(i, f"question number {i} about lane {i}")) for i in range(20)]
    cands[5].signals = ["error"]
    sample = mine.stratified_sample(cands, 5)
    assert sample[0].run_id == "r5" and len(sample) == 5
    assert len({c.run_id for c in sample}) == 5


def _item(must_do=("states $75/hour after 2 free hours",)):
    return {"run_id": "r1", "input_text": "what's detention?", "output_text": "x", "signals": [],
            "triage": {"category": "detention_question", "looks_wrong": False, "why": "", "must_do": list(must_do), "must_not": []}}


def test_review_accept_edit_reject():
    answers = iter(["a", "e", '["new item"]', "", "r"])
    out = review.review([_item(), _item(), _item()], ask=lambda _: next(answers), show=lambda _: None)
    assert len(out) == 2
    assert out[1]["outputs"]["must_do"] == ["new item"]
    assert out[0]["metadata"]["source_run_id"] == "r1"


def test_accept_all_skips_items_without_a_rubric():
    assert review.accept_all([_item(must_do=())]) == []
    assert len(review.accept_all([_item()])) == 1


def test_reply_text_scores_a_pause_as_a_pause():
    paused = {"__interrupt__": [SimpleNamespace(value={"tool": "escalate_to_ops"})], "messages": []}
    assert experiment._reply_text(paused).startswith("[paused for human approval]")
    done = {"messages": [SimpleNamespace(type="human", content="hi"), SimpleNamespace(type="ai", content="Hello")]}
    assert experiment._reply_text(done) == "Hello"


def test_code_evaluators():
    assert experiment.no_error({"error": "boom"})["score"] == 0
    assert experiment.answered({"reply": "ok"})["score"] == 1


def test_monitor_rule_body():
    body = monitor.rule_body("proj-1", "ds-1", "evalkit: x")
    assert body["session_id"] == "proj-1" and body["add_to_dataset_id"] == "ds-1"
    assert 'eq(status, "error")' in body["filter"] and body["sampling_rate"] == 1.0
    with pytest.raises(ValueError):
        monitor.rule_body("p", "d", "n", sampling_rate=0)


def test_gate_floors_and_baseline():
    assert check({"no_error": 1.0, "rubric": 0.9}, floors={"no_error": 1.0}, baseline={"rubric": 0.92}, max_drop=0.05) == []
    fails = check({"no_error": 0.95, "rubric": 0.8}, floors={"no_error": 1.0}, baseline={"rubric": 0.92}, max_drop=0.05)
    assert len(fails) == 2


def test_resume_runs_are_not_eval_inputs():
    c = mine.to_candidate(run(1, "x"))
    c.inputs = {"input": {"goto": [], "resume": {"decisions": [{"type": "approve"}]}}}
    assert mine.is_resume(c)
    assert not mine.is_resume(mine.to_candidate(run(2, "where is my load")))


def test_multiturn_pairs_reply_with_last_question_and_keeps_tool_evidence():
    r = run(1, "first question")
    r.inputs = {"messages": [{"type": "human", "content": "How does detention work?"},
                             {"type": "ai", "content": "2 hours free..."},
                             {"type": "human", "content": "Send RL-48251 detention to billing"}]}
    r.outputs = {"messages": [
        {"type": "human", "content": "How does detention work?"},
        {"type": "ai", "content": "", "tool_calls": [{"name": "search_policy", "args": {"q": "detention"}}]},
        {"type": "human", "content": "Send RL-48251 detention to billing"},
        {"type": "ai", "content": "", "tool_calls": [{"name": "lookup_load", "args": {"load_id": "RL-48251"}}]},
        {"type": "tool", "name": "lookup_load", "content": "{\"status\": \"delivered\"}"},
        {"type": "ai", "content": "Requesting a detention review for RL-48251."}]}
    c = mine.to_candidate(r)
    assert c.input_text == "Send RL-48251 detention to billing"
    assert "lookup_load" in c.tool_evidence and "search_policy" not in c.tool_evidence
    assert c.output_text == "Requesting a detention review for RL-48251."


def test_turn_paused_for_approval_is_not_answered_with_the_previous_reply():
    r = run(1, "x")
    r.inputs = {"messages": [{"type": "human", "content": "Send RL-48251 detention to billing"}]}
    r.outputs = {"messages": [
        {"type": "human", "content": "How does detention work?"},
        {"type": "ai", "content": "2 hours free, then $75/hour."},
        {"type": "human", "content": "Send RL-48251 detention to billing"},
        {"type": "ai", "content": "", "tool_calls": [{"id": "c1", "name": "lookup_load", "args": {}}]},
        {"type": "tool", "name": "lookup_load", "tool_call_id": "c1", "content": "{}"},
        {"type": "ai", "content": "", "tool_calls": [{"id": "c2", "name": "request_detention_review", "args": {"hours": 4.25}}]}]}
    c = mine.score_signals([mine.to_candidate(r)])[0]
    assert c.output_text == "[paused for human approval: request_detention_review]"
    assert "human_interrupt" in c.signals


def test_example_carries_thread_history():
    r = run(1, "x")
    r.inputs = {"messages": [{"type": "human", "content": "Anything on RL-48410?"}]}
    r.outputs = {"messages": [
        {"type": "human", "content": "This is Jordan, account ACCT-1001."},
        {"type": "ai", "content": "Hi Jordan, how can I help?"},
        {"type": "tool", "name": "x", "content": "{}"},
        {"type": "human", "content": "Anything on RL-48410?"},
        {"type": "ai", "content": "It's booked for Friday."}]}
    c = mine.to_candidate(r)
    ex = review.to_example({**c.to_json(), "triage": {"must_do": ["states it is booked"], "must_not": []}})
    assert [m["role"] for m in ex["inputs"]["messages"]] == ["user", "assistant", "user"]
    assert "ACCT-1001" in ex["inputs"]["messages"][0]["content"]


def test_replay_marks_paused_turns_and_carries_state():
    r = run(1, "x")
    r.inputs = {"messages": [{"type": "human", "content": "Our dock closes at 3pm."}]}
    r.outputs = {"account": "ACCT-1001", "messages": [
        {"type": "human", "content": "Send RL-48251 detention to billing"},
        {"type": "ai", "content": "", "tool_calls": [{"id": "c2", "name": "request_detention_review", "args": {}}]},
        {"type": "human", "content": "Our dock closes at 3pm."},
        {"type": "ai", "content": "Saved."}]}
    c = mine.to_candidate(r)
    assert [m["role"] for m in c.history] == ["user", "assistant", "user"]
    assert c.history[1]["content"] == mine.PAUSE_NOTE and "tool_calls" not in c.history[1]
    ex = review.to_example({**c.to_json(), "triage": {"must_do": ["saves it"], "must_not": []}})
    assert ex["inputs"]["account"] == "ACCT-1001"


def test_replay_keeps_completed_tool_calls_and_results():
    r = run(1, "x")
    r.outputs = {"messages": [
        {"type": "human", "content": "Send RL-48251 detention to billing"},
        {"type": "ai", "content": "", "tool_calls": [{"id": "c1", "name": "request_detention_review", "args": {"hours": 4.25}}]},
        {"type": "tool", "name": "request_detention_review", "tool_call_id": "c1", "content": "filed DET-1"},
        {"type": "ai", "content": "Filed under DET-1."},
        {"type": "human", "content": "Our dock closes at 3pm."},
        {"type": "ai", "content": "Saved."}]}
    h = mine.to_candidate(r).history
    assert [m["role"] for m in h] == ["user", "assistant", "tool", "assistant", "user"]
    assert h[1]["tool_calls"][0]["id"] == h[2]["tool_call_id"] == "c1"
    from langchain_core.messages import convert_to_messages
    assert len(convert_to_messages(h)) == 5   # valid LangChain input


def test_paused_when_required_is_a_code_check():
    ref = {"must_do": ["Pauses for human approval before sending the review"]}
    assert experiment.paused_when_required({"paused": True}, ref)["score"] == 1
    assert experiment.paused_when_required({"paused": False}, ref)["score"] == 0
    assert experiment.paused_when_required({"paused": False}, {"must_do": ["States the rate"]})["score"] is None
