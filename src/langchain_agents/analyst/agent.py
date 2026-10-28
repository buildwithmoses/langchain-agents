"""The analyst graph (custom LangGraph StateGraph).

    start -> agent --tool calls--> tools --> agent ...
                   \\--answer----> number_check --pass--> finalize -> END
                                       |  \\--fail, first time--> rewrite --> number_check
                                       \\--fail again--> finalize (table without prose)

Why a custom graph and not create_agent + middleware: the guarantees here are steps
in a fixed order (guard -> maybe interrupt -> execute -> verify numbers -> maybe
rewrite -> fail closed). As explicit nodes the model can't route around them, the
interrupt sits exactly where the SQL is about to run, and every branch is testable
with a fake model. The model only chooses SQL and words.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .db import QueryError, ReadOnlyDB
from .numbers import check_numbers
from .prompts import NUMBER_CHECK_FEEDBACK, system_prompt
from .render import render_final
from .sqlguard import review

MAX_MODEL_CALLS = 8  # per turn; runaway-loop and cost protection
PRIOR_QUERIES_KEPT = 10
ROWS_TO_MODEL = 60
GUARD_NAMES = {"number_check"}


# ---- tool schemas (the graph executes them; the model only sees the signatures) -----------
@tool
def list_tables() -> str:
    """List the tables in the Ridgeline TMS with row counts."""
    raise NotImplementedError  # executed by the tools node


@tool
def describe_table(table: str) -> str:
    """Columns (with types and PII flags) and 3 sample rows, PII masked, for one table."""
    raise NotImplementedError


@tool
def run_query(sql: str) -> str:
    """Run ONE read-only SQLite SELECT (or WITH ... SELECT). Returns columns and rows.
    Queries are checked before they run; some need a human's approval first."""
    raise NotImplementedError


TOOLS = [list_tables, describe_table, run_query]


class AnalystState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    queries: list[dict]        # executed this turn
    prior_queries: list[dict]  # results from earlier turns (the number check may cite them)
    guard_log: list[dict]      # every run_query decision this turn
    approvals: list[dict]      # human decisions this turn
    model_calls: int
    rewrites: int
    number_check: dict
    final: dict


# ---- retries: transient only, fail closed -------------------------------------------------------
_TRANSIENT = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
              "OverloadedError", "ServiceUnavailableError", "TimeoutError", "ConnectionError",
              "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}


def is_transient(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return any(c.__name__ in _TRANSIENT for c in type(exc).__mro__)


def invoke_with_retry(llm, messages, retries: int = 3, base_delay: float = 1.0):
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            if attempt == retries or not is_transient(e):
                raise
            time.sleep(base_delay * 2 ** attempt)


def _text(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type", "text") == "text")


def _last_question(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.name not in GUARD_NAMES:
            return _text(m)
    return ""


def _format_result_for_model(res: dict) -> str:
    cols, rows = res["columns"], res["rows"]
    lines = [" | ".join(cols)]
    lines += [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:ROWS_TO_MODEL]]
    more = ""
    if len(rows) > ROWS_TO_MODEL:
        more = f"\n({len(rows) - ROWS_TO_MODEL} more rows not shown to you; the user sees up to {len(rows)})"
    if res.get("truncated"):
        more += "\n(Result was capped by the row limit; aggregate if you need totals.)"
    if res.get("redacted_cells"):
        more += f"\n({res['redacted_cells']} cells were masked as [PII].)"
    return f"{len(rows)} row(s):\n" + "\n".join(lines) + more


def _parse_decision(value: Any) -> dict:
    if isinstance(value, str):
        return {"decision": value.strip().lower()}
    if isinstance(value, dict):
        d = dict(value)
        d["decision"] = str(d.get("decision", d.get("type", ""))).strip().lower()
        return d
    return {"decision": "reject", "reason": "unrecognised approval response"}


def build_graph(model: BaseChatModel | None = None, db: ReadOnlyDB | None = None, *, checkpointer: Any = None,
                guard: bool = True, verify_numbers: bool = True, prompt: str = "default", name: str = "analyst"):
    """Compile the analyst graph.

    model: any chat model that supports bind_tools (tests pass a fake). Default: get_model().
    guard / verify_numbers / prompt="naive": switch off a safety layer, ONLY to measure what it is
    worth in evals.
    checkpointer: leave None under `langgraph dev` (the platform supplies one); pass an
    InMemorySaver for the CLI, tests and evals (the approval interrupt needs one).
    """
    if model is None:
        from langchain_agents.core.config import get_model

        model = get_model()
    db = db or ReadOnlyDB()
    sys_msg = SystemMessage(system_prompt(db, prompt))
    llm_tools = model.bind_tools(TOOLS)

    def start(state: AnalystState) -> dict:
        prior = [*state.get("prior_queries", []),
                 *({"sql": q.get("sql", ""), "rows": q.get("rows", [])} for q in state.get("queries", []))]
        return {"question": _last_question(state.get("messages", [])), "queries": [], "guard_log": [],
                "prior_queries": prior[-PRIOR_QUERIES_KEPT:],
                "approvals": [], "model_calls": 0, "rewrites": 0, "number_check": {}, "final": {}}

    def agent(state: AnalystState) -> dict:
        n = state.get("model_calls", 0)
        if n >= MAX_MODEL_CALLS:
            return {"messages": [AIMessage(content="I stopped after too many steps without a verified answer. "
                                                   "Please rephrase or narrow the question.", name="limit")]}
        msg = invoke_with_retry(llm_tools, [sys_msg, *state["messages"]])
        return {"messages": [msg], "model_calls": n + 1}

    def route_agent(state: AnalystState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state.get("model_calls", 0) <= MAX_MODEL_CALLS:
            return "tools"
        return "number_check"

    def execute_reviewed(sql: str, decision) -> tuple[str, dict | None]:
        try:
            res = db.execute(decision.sql_to_run, allowed_pii=decision.pii_touched).to_dict()
        except QueryError as e:
            return f"ERROR from SQLite: {e}", None
        res["submitted_sql"] = sql
        res["limit_applied"] = decision.limit_applied
        return _format_result_for_model(res), res

    def tools(state: AnalystState) -> dict:
        last = state["messages"][-1]
        queries = list(state.get("queries", []))
        log = list(state.get("guard_log", []))
        approvals = list(state.get("approvals", []))
        out = []
        for call in last.tool_calls:
            name, args, cid = call["name"], call.get("args") or {}, call["id"]
            try:
                if name == "list_tables":
                    content = json.dumps(db.list_tables())
                elif name == "describe_table":
                    content = json.dumps(db.describe_table(str(args.get("table", ""))), default=str)
                elif name == "run_query":
                    sql = str(args.get("sql", ""))
                    if not guard:
                        try:
                            res = db.execute(sql, enforce=False).to_dict()
                            res["submitted_sql"] = sql
                            queries.append(res)
                            log.append({"sql": sql, "status": "unchecked", "reasons": []})
                            content = _format_result_for_model(res)
                        except QueryError as e:
                            log.append({"sql": sql, "status": "db_error", "reasons": [str(e)]})
                            content = f"ERROR from SQLite: {e}"
                        out.append(ToolMessage(content=content, tool_call_id=cid, name=name))
                        continue
                    decision = review(sql, db)
                    entry = decision.to_dict()
                    if decision.blocked:
                        log.append(entry)
                        content = "BLOCKED by the SQL guard (not executed): " + " ".join(decision.reasons)
                    elif decision.status == "needs_approval":
                        resp = _parse_decision(interrupt({
                            "type": "sql_approval",
                            "question": state.get("question", ""),
                            "sql": decision.sql_to_run,
                            "reasons": decision.reasons,
                            "allowed_decisions": ["approve", "edit", "reject"],
                        }))
                        approvals.append({"sql": decision.sql_to_run, "reasons": decision.reasons, **resp})
                        entry["approval"] = resp["decision"]
                        if resp["decision"] == "approve":
                            content, res = execute_reviewed(sql, decision)
                            if res:
                                res["approved"] = True
                                queries.append(res)
                        elif resp["decision"] == "edit" and resp.get("sql"):
                            edited = review(str(resp["sql"]), db)
                            entry["edited"] = edited.to_dict()
                            if edited.blocked:
                                content = "BLOCKED by the SQL guard: the reviewer's edited SQL failed the checks: " \
                                          + " ".join(edited.reasons)
                            else:
                                # The human wrote it and it passed the static guard; no second prompt.
                                content, res = execute_reviewed(str(resp["sql"]), edited)
                                if res:
                                    res["approved"] = True
                                    res["edited_by_reviewer"] = True
                                    queries.append(res)
                                    content = "The reviewer edited your SQL before it ran. Ran:\n" \
                                              + edited.sql_to_run + "\n" + content
                        else:
                            reason = resp.get("reason") or "no reason given"
                            content = (f"REJECTED by the human reviewer ({reason}). The query did not run. "
                                       "Do not resubmit it; tell the user it was not approved.")
                        log.append(entry)
                    else:
                        log.append(entry)
                        content, res = execute_reviewed(sql, decision)
                        if res:
                            queries.append(res)
                else:
                    content = f"Unknown tool {name!r}."
            except QueryError as e:
                content = f"ERROR: {e}"
            out.append(ToolMessage(content=content, tool_call_id=cid, name=name))
        return {"messages": out, "queries": queries, "guard_log": log, "approvals": approvals}

    def number_check(state: AnalystState) -> dict:
        answer = _text(state["messages"][-1])
        # Earlier turns' results count as evidence too ("vs 74 in September", said last turn).
        evidence = [*state.get("prior_queries", []), *state.get("queries", [])]
        nc = check_numbers(answer, evidence, state.get("question", "")).to_dict()
        return {"number_check": nc}

    def route_check(state: AnalystState) -> str:
        nc = state.get("number_check") or {}
        if nc.get("passed") or not verify_numbers:
            return "finalize"
        if state.get("rewrites", 0) < 1 and state.get("model_calls", 0) < MAX_MODEL_CALLS:
            return "rewrite"
        return "finalize"

    def rewrite(state: AnalystState) -> dict:
        bad = ", ".join(state["number_check"]["unsupported"])
        nudge = HumanMessage(content=NUMBER_CHECK_FEEDBACK.format(bad=bad), name="number_check")
        msgs = [sys_msg, *state["messages"], nudge]
        msg = invoke_with_retry(model, msgs)  # no tools: words only
        if isinstance(msg, AIMessage) and msg.tool_calls:
            msg = AIMessage(content=_text(msg) or "")
        return {"messages": [nudge, msg], "rewrites": state.get("rewrites", 0) + 1,
                "model_calls": state.get("model_calls", 0) + 1}

    def finalize(state: AnalystState) -> dict:
        answer = _text(state["messages"][-1]).strip()
        nc = state.get("number_check") or {}
        queries = state.get("queries", [])
        mode = "answer"
        if verify_numbers and not nc.get("passed", False):
            mode = "table_only"
            answer = ("I couldn't verify every number in my written summary against the query results, "
                      "so here is the result table without commentary." if queries else
                      "I couldn't produce an answer I could verify against the data. Please rephrase the question.")
        leaks = db.find_pii_in_text(answer) if guard else []
        if leaks:
            for v in leaks:
                answer = answer.replace(v, "[PII]")
        final = {
            "answer": answer,
            "mode": mode,
            "number_check": nc,
            "queries": [{k: q[k] for k in ("sql", "columns", "rows", "truncated", "elapsed_ms") if k in q}
                        | {"approved": q.get("approved", False), "edited_by_reviewer": q.get("edited_by_reviewer", False),
                           "limit_applied": q.get("limit_applied")} for q in queries],
            "blocked": [g for g in state.get("guard_log", []) if g.get("status") == "blocked"],
            "approvals": state.get("approvals", []),
            "pii_redacted_from_answer": len(leaks),
        }
        final["markdown"] = render_final(final)
        update: dict = {"final": final}
        if mode == "table_only" or leaks:
            update["messages"] = [AIMessage(content=final["markdown"], name="analyst_final")]
        return update

    g = StateGraph(AnalystState)
    g.add_node("start", start)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("number_check", number_check)
    g.add_node("rewrite", rewrite)
    g.add_node("finalize", finalize)
    g.add_edge(START, "start")
    g.add_edge("start", "agent")
    g.add_conditional_edges("agent", route_agent, {"tools": "tools", "number_check": "number_check"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("number_check", route_check, {"finalize": "finalize", "rewrite": "rewrite"})
    g.add_edge("rewrite", "number_check")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name=name)
