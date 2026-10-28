"""Brief rendering, and the whole graph (deep agent -> verify -> revise ->
finalize) driven by a scripted fake model. Offline, no keys."""

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_agents.research.agent import build_graph
from langchain_agents.research.render import render_brief, write_outputs
from langchain_agents.research.run import run_question, stats

MEMO = "corpus:ridgeline-market-memo-q3-2026"
GOOD = "Reefer spot rates out of the Southeast averaged $2.87 per mile in Q3 2026, up 14% from $2.52 in Q2."
FAKE = "Southeast reefer rates doubled in Q3 because every carrier left the market."


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed list of AIMessages in order (the lead and its subagents
    share it, which is deterministic because they run sequentially here)."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, args, i):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}])


def brief(claims, answer):
    return AIMessage(json.dumps({"answerable": True, "answer": answer, "claims": claims, "gaps": []}))


def script():
    good = {"claim": "Southeast reefer spot rates averaged $2.87 per mile in Q3 2026.", "source_id": MEMO, "quote": GOOD}
    fake = {"claim": "Rates doubled because carriers left.", "source_id": MEMO, "quote": FAKE}
    unread = {"claim": "Diesel rose 31 cents.", "source_id": "corpus:gulf-fuel-watch-2026-w41",
              "quote": "up 31 cents from $3.63 at the start of July"}
    notes = {"source_id": MEMO, "title": "Memo", "relevant": True, "quotes": [{"quote": GOOD, "why": "rate level"}]}
    return [
        tool_call("task", {"description": f"Read {MEMO} for: what drives reefer rates?", "subagent_type": "reader"}, 1),
        tool_call("read_source", {"source_id": MEMO}, 2),          # reader
        AIMessage(json.dumps(notes)),                              # reader structured output
        brief([good, fake, unread], "Rates hit $2.87 [1]. They doubled [2]. Diesel rose [3]."),   # draft
        brief([good, fake], "Rates hit $2.87 [1]. They doubled [2]."),                          # one revision
    ]


def test_graph_drops_fabricated_and_unread_claims_after_one_revision(tmp_path):
    model = ScriptedModel(messages=iter(script()))
    res = run_question("What's driving reefer rates?", mode="corpus", graph=build_graph(model, mode="corpus"),
                       out_dir=tmp_path)
    state, meta = res["state"], res["meta"]
    final = state["final"]
    assert [c["quote"] for c in final["claims"]] == [GOOD]
    assert final["answer"] == "Rates hit $2.87 [1]."
    assert {d["reason"] for d in final["dropped"]} == {"quote_not_found"}
    first = state["rounds"][0]
    assert first["n_claims"] == 3 and first["n_verified"] == 1
    assert {c["reason"] for c in first["checks"] if not c["verified"]} == {"quote_not_found", "source_not_fetched"}
    assert meta["stats"]["revisions"] == 1 and meta["stats"]["sources_read"] == 1
    md = (tmp_path / "brief.md").read_text()
    assert md.index("## Answer") < md.index("## Findings") < md.index("## What we couldn't verify")
    assert FAKE not in md.split("## What we couldn't verify")[0]
    claims = json.loads((tmp_path / "claims.json").read_text())
    assert len(claims["raw_brief"]["claims"]) == 3 and len(claims["final"]["claims"]) == 1


def test_graph_fails_closed_when_agent_returns_no_brief(tmp_path):
    model = ScriptedModel(messages=iter([AIMessage("I could not finish.")]))
    res = run_question("Anything?", mode="corpus", graph=build_graph(model, mode="corpus"), write=False)
    final = res["state"]["final"]
    assert final["claims"] == [] and final["answerable"] is False


def test_render_brief_sections_and_labels():
    final = {"answerable": True, "answer": "Rates rose [1].", "answer_sentences_dropped": 1,
             "claims": [{"n": 1, "claim": "Rates rose 14%.", "source_id": "https://ex.example/a", "quote": "up 14%"}],
             "dropped": [{"reason": "quote_not_found", "claim": "Made up", "source_id": "https://ex.example/b"}],
             "gaps": ["No Q4 data"]}
    meta = {"mode": "web", "cost_usd": 0.0123, "latency_s": 4.2, "model": "m",
            "stats": {"raw_claims": 2, "raw_verified": 1, "final_claims": 1, "final_dropped": 1}}
    md = render_brief("Q?", final, {"https://ex.example/a": {"title": "Ex A"}}, meta)
    assert md.index("## Answer") < md.index("## Findings") < md.index("## What we couldn't verify") < md.index("## Run")
    assert "[Ex A](https://ex.example/a)" in md and "Dropped claim (quote_not_found): Made up" in md
    assert "Gap: No Q4 data" in md and "$0.0123" in md


def test_stats_counts_raw_and_final():
    state = {"rounds": [{"n_claims": 3, "n_verified": 1, "checks": [
        {"verified": False, "reason": "quote_not_found"}, {"verified": False, "reason": "source_not_fetched"},
        {"verified": True, "reason": "ok"}]}], "final": {"claims": [1], "dropped": [1, 2]},
        "sources": {"a": {"text": "x"}, "b": {"text": ""}}, "searches": {"s": {}}}
    s = stats(state)
    assert (s["raw_claims"], s["raw_verified"], s["final_dropped"], s["sources_read"], s["sources_failed"]) == (3, 1, 2, 1, 1)
    assert s["raw_failed_by_reason"] == {"quote_not_found": 1, "source_not_fetched": 1}


def test_write_outputs_strips_source_text_from_run_json(tmp_path):
    state = {"final": {"answerable": False, "answer": "x", "claims": [], "dropped": [], "gaps": []},
             "sources": {"a": {"text": "secret long text", "title": "A"}}, "rounds": []}
    write_outputs(tmp_path, "Q", state, {"mode": "corpus", "stats": {}})
    assert "secret long text" not in (tmp_path / "run.json").read_text()


def test_reader_that_skips_read_source_is_sent_back_to_read():
    """Regression: Haiku once returned reader JSON without calling read_source."""
    notes_empty = {"source_id": MEMO, "title": "", "relevant": False, "quotes": []}
    notes = {"source_id": MEMO, "title": "Memo", "relevant": True, "quotes": [{"quote": GOOD, "why": "rate"}]}
    good = {"claim": "Southeast reefer spot rates averaged $2.87 per mile in Q3 2026.", "source_id": MEMO, "quote": GOOD}
    model = ScriptedModel(messages=iter([
        tool_call("task", {"description": f"Read {MEMO}", "subagent_type": "reader"}, 1),
        AIMessage(json.dumps(notes_empty)),               # reader tries to finish without reading
        tool_call("read_source", {"source_id": MEMO}, 2),  # after the nudge it reads
        AIMessage(json.dumps(notes)),
        brief([good], "Rates hit $2.87 [1]."),
    ]))
    res = run_question("Rates?", mode="corpus", graph=build_graph(model, mode="corpus"), write=False)
    assert MEMO in res["state"]["sources"]
    assert len(res["state"]["final"]["claims"]) == 1 and res["meta"]["stats"]["revisions"] == 0
