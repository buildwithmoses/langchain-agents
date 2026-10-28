"""The graph with a scripted model and no keys: approval interrupt, guard, number check."""

import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_agents.analyst.agent import build_graph
from langchain_agents.analyst.testing import ScriptedModel, shared_db, sql_call


@pytest.fixture(scope="module")
def db():
    return shared_db()


def make(db, script, **kw):
    model = ScriptedModel(script=list(script))
    return build_graph(model=model, db=db, checkpointer=InMemorySaver(), **kw), model


def ask(graph, q, thread=None):
    cfg = {"configurable": {"thread_id": thread or str(uuid.uuid4())}}
    return graph.invoke({"messages": [{"role": "user", "content": q}]}, cfg), cfg


def tool_msgs(graph, cfg):
    return [m for m in graph.get_state(cfg).values["messages"] if isinstance(m, ToolMessage)]


def test_simple_question_runs_query_and_passes_number_check(db):
    n = db.execute("SELECT count(*) FROM loads WHERE status = 'Cancelled'").rows[0][0]
    g, _ = make(db, [sql_call("SELECT count(*) AS cancelled FROM loads WHERE status = 'Cancelled'"),
                     f"There were {n} cancelled loads."])
    out, _ = ask(g, "How many loads were cancelled?")
    f = out["final"]
    assert f["mode"] == "answer" and f["number_check"]["passed"]
    assert f["queries"][0]["rows"] == [[n]]
    assert "LIMIT 200" in f["queries"][0]["sql"]  # enforced limit is what ran, and is shown
    assert "```sql" in f["markdown"] and "| cancelled |" in f["markdown"]


def test_pii_aggregate_pauses_for_approval_then_runs_on_approve(db):
    sql = "SELECT count(*) AS n FROM carriers WHERE contact_email LIKE '%carrier1%'"
    n = db.execute(sql, allowed_pii={("carriers", "contact_email")}).rows[0][0]
    g, _ = make(db, [sql_call(sql), f"{n} carriers match."])
    out, cfg = ask(g, "How many carriers have carrier1 in their email?")
    intr = out["__interrupt__"][0].value
    assert intr["type"] == "sql_approval" and "carriers.contact_email" in " ".join(intr["reasons"])
    assert g.get_state(cfg).next == ("tools",)  # nothing ran yet
    out = g.invoke(Command(resume={"decision": "approve"}), cfg)
    f = out["final"]
    assert f["mode"] == "answer" and f["queries"][0]["approved"] and f["queries"][0]["rows"] == [[n]]
    assert f["approvals"][0]["decision"] == "approve"


def test_reject_means_the_query_never_runs(db):
    g, _ = make(db, [sql_call("SELECT count(*) FROM carriers WHERE contact_phone LIKE '(901)%'"),
                     "The reviewer did not approve that query, so I have no result."])
    out, cfg = ask(g, "How many carriers have a 901 phone?")
    assert out["__interrupt__"]
    out = g.invoke(Command(resume={"decision": "reject", "reason": "not needed"}), cfg)
    f = out["final"]
    assert f["queries"] == [] and f["approvals"][0]["decision"] == "reject"
    assert "REJECTED" in tool_msgs(g, cfg)[0].content


def test_edit_runs_the_reviewers_sql(db):
    g, _ = make(db, [sql_call("SELECT count(*) FROM carriers WHERE contact_email LIKE '%a%'"),
                     lambda msgs: AIMessage(content=f"There are {msgs[-1].content.split()[-1]} carriers.")])
    out, cfg = ask(g, "How many carriers?")
    out = g.invoke(Command(resume={"decision": "edit", "sql": "SELECT count(*) FROM carriers"}), cfg)
    f = out["final"]
    assert f["queries"][0]["edited_by_reviewer"] and f["queries"][0]["rows"] == [[1800]]
    assert f["mode"] == "answer"


def test_reviewer_edit_still_goes_through_the_guard(db):
    g, _ = make(db, [sql_call("SELECT count(*) FROM carriers WHERE contact_email LIKE '%a%'"), "Nothing ran."])
    out, cfg = ask(g, "q")
    out = g.invoke(Command(resume={"decision": "edit", "sql": "DROP TABLE loads"}), cfg)
    assert out["final"]["queries"] == []
    assert "BLOCKED" in tool_msgs(g, cfg)[0].content


def test_big_scan_pauses_for_approval(db):
    g, _ = make(db, [sql_call("SELECT status, count(*) FROM check_calls GROUP BY status"), "Done."])
    out, _ = ask(g, "Check call statuses?")
    assert "check_calls" in " ".join(out["__interrupt__"][0].value["reasons"])


@pytest.mark.parametrize("sql", ["DELETE FROM loads WHERE status = 'Cancelled'", "DROP TABLE loads",
                                 "SELECT contact_phone FROM carriers", "SELECT 1; DROP TABLE loads"])
def test_blocked_query_never_reaches_db_and_model_sees_reason(db, sql):
    g, _ = make(db, [sql_call(sql), "I can't do that: I only have read access and can't share contact details."])
    out, cfg = ask(g, "do the thing")
    f = out["final"]
    assert f["queries"] == [] and len(f["blocked"]) == 1
    assert tool_msgs(g, cfg)[0].content.startswith("BLOCKED")
    assert db.execute("SELECT count(*) FROM loads").rows[0][0] == 14400


def test_number_check_rewrites_once_then_passes(db):
    g, model = make(db, [sql_call("SELECT count(*) FROM loads"), "There were 15,000 loads.", "There were 14,400 loads."])
    out, _ = ask(g, "How many loads?")
    f = out["final"]
    assert f["mode"] == "answer" and f["answer"] == "There were 14,400 loads."
    assert "15,000" in model.seen[-1][-1].content  # feedback named the bad number


def test_number_check_fails_closed_to_table_only(db):
    g, _ = make(db, [sql_call("SELECT count(*) AS n FROM loads"), "There were 15,000 loads.", "Still 15,000 loads."])
    out, _ = ask(g, "How many loads?")
    f = out["final"]
    assert f["mode"] == "table_only" and "15,000" not in f["answer"]
    assert "| n |" in f["markdown"] and "14400" in f["markdown"]


def test_number_check_can_be_disabled_for_regression_evals(db):
    g, _ = make(db, [sql_call("SELECT count(*) FROM loads"), "There were 15,000 loads."], verify_numbers=False)
    out, _ = ask(g, "How many loads?")
    assert out["final"]["mode"] == "answer" and not out["final"]["number_check"]["passed"]


def test_without_guard_pii_flows_through_which_is_what_the_eval_measures(db):
    g, _ = make(db, [sql_call("SELECT contact_phone FROM carriers LIMIT 2"), "Here they are."], guard=False)
    out, _ = ask(g, "phones")
    phone = out["final"]["queries"][0]["rows"][0][0]
    assert db.find_pii_in_text(str(phone))


def test_pii_in_model_prose_is_redacted(db):
    name = db.execute("SELECT contact_name FROM carriers WHERE carrier_id = 3", enforce=False).rows[0][0]
    g, _ = make(db, [f"You could ask {name} directly."])
    out, _ = ask(g, "who should I call?")
    assert name not in out["final"]["answer"] and out["final"]["pii_redacted_from_answer"] == 1


def test_multi_turn_keeps_history_and_resets_turn_state(db):
    g, model = make(db, [sql_call("SELECT count(*) FROM loads"), "There were 14,400 loads.",
                         sql_call("SELECT count(*) FROM carriers"), "And 1,800 carriers."])
    tid = str(uuid.uuid4())
    ask(g, "How many loads?", tid)
    out, _ = ask(g, "And carriers?", tid)
    f = out["final"]
    assert len(f["queries"]) == 1 and f["queries"][0]["rows"] == [[1800]]
    assert any("14,400" in str(m.content) for m in model.seen[-1])


def test_model_call_cap(db):
    script = [sql_call("SELECT 1", f"c{i}") for i in range(8)] + ["unused"]
    g, _ = make(db, script)
    out, _ = ask(g, "loop forever")
    assert out["final"]["answer"]  # ended instead of looping
    assert len(out["final"]["queries"]) == 8


def test_number_check_accepts_numbers_from_earlier_turns(db):
    g, _ = make(db, [sql_call("SELECT count(*) FROM loads"), "There were 14,400 loads.",
                     sql_call("SELECT count(*) FROM carriers"), "1,800 carriers moved those 14,400 loads."])
    tid = str(uuid.uuid4())
    ask(g, "How many loads?", tid)
    out, _ = ask(g, "And carriers?", tid)
    assert out["final"]["mode"] == "answer" and out["final"]["number_check"]["passed"]
