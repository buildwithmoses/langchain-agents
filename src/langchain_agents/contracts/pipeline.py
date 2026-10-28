"""The contract review graph (LangGraph StateGraph).

    ingest -> extract -> verify --(failed quotes, first pass)--> reextract -> verify
                                  -> rules -> draft -> lawyer_review [interrupt] -> finalize

Why a plain StateGraph and not Deep Agents: the work is a fixed pipeline with
deterministic steps between two model calls. There is nothing to plan or
delegate, and a planning agent would add cost, latency and a place for the
model to skip the verifier or decide compliance itself. LangGraph gives the
pieces that matter here: typed state, a conditional retry edge, an interrupt
for the lawyer, and checkpointed resume.

Eval ablations (never used in production, selected by state["variant"]):
- "no_verifier": accept every extracted quote/value without checking.
- "model_decides": skip extraction+rules; the model judges compliance itself.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import prompts
from .llm import ChatStructuredLLM, StructuredLLM, is_transient
from .playbook import RULES, VERIFY_SPEC, Finding, evaluate_rules
from .render import memo_markdown, redlines_markdown
from .schemas import TERM_MODELS, MemoDraft, ModelJudgments, chunk_terms, subset_schema
from .sections import Section, locate, render_for_prompt, split_sections
from .verify import verify_term

VALID_ACTIONS = {"approve", "edit", "reject"}


class ReviewState(TypedDict, total=False):
    # input
    contract_path: str
    contract_text: str
    contract_name: str
    variant: str
    # derived
    sections: list[dict]
    carrier: str
    extraction: dict
    first_pass_problems: dict
    verification: dict
    attempts: int
    findings: list[dict]
    summary: str
    redlines: list[dict]
    memo_md: str
    decisions: dict
    redlines_md: str
    errors: list[str]


def _resolve_contract(path: str) -> Path:
    from ..core.config import data_dir
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = data_dir("contracts", path)
    if not p.is_file():
        raise FileNotFoundError(f"Contract not found: {path}")
    return p


def build_graph(llm: StructuredLLM | BaseChatModel | None = None, *, checkpointer: Any = None):
    """Compile the review graph. Pass a fake `llm` (anything with .structured()) in tests.
    Leave checkpointer None under `langgraph dev` (the platform supplies one); pass an
    InMemorySaver for the CLI and evals - the lawyer interrupt needs a checkpointer."""
    if llm is None or isinstance(llm, BaseChatModel):
        llm = ChatStructuredLLM(llm)

    def ingest(state: ReviewState) -> dict:
        text = state.get("contract_text")
        name = state.get("contract_name") or "contract"
        if not text:
            p = _resolve_contract(state["contract_path"])
            text, name = p.read_text(), p.name
        secs = split_sections(text)
        return {"contract_text": text, "contract_name": name, "sections": [s.to_dict() for s in secs],
                "attempts": 0, "errors": [], "variant": state.get("variant") or "baseline"}

    def _sections(state) -> list[Section]:
        return [Section(**s) for s in state["sections"]]

    def _extract_groups(state: ReviewState, terms: list[str], user: str, config, name: str,
                        with_carrier: bool) -> tuple[dict, list[str], str]:
        """Run the term groups concurrently. A group that fails (non-transient) fails
        closed: its terms are marked _failed and become needs_check."""
        groups = chunk_terms(terms)
        results: dict[str, dict] = {}
        errors: list[str] = []
        carrier = ""

        def call(i: int, group: list[str]):
            schema = subset_schema(group, with_carrier=with_carrier and i == 0)
            return llm.structured(schema, prompts.EXTRACT_SYSTEM, user + f"\n\nTERMS TO EXTRACT NOW: {', '.join(group)}",
                                  config=config, name=f"{name}_{i + 1}")

        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = {pool.submit(call, i, g): (i, g) for i, g in enumerate(groups)}
            for fut, (i, group) in futures.items():
                try:
                    out = fut.result()
                except Exception as exc:  # noqa: BLE001
                    if is_transient(exc):
                        raise
                    errors.append(f"{name} group {group} failed: {str(exc)[:300]}")
                    results.update({t: _failed_term() for t in group})
                    continue
                if with_carrier and i == 0:
                    carrier = getattr(out, "carrier_name", "") or ""
                results.update({t: getattr(out, t).model_dump() for t in group})
        return results, errors, carrier

    def extract(state: ReviewState, config: RunnableConfig) -> dict:
        contract = render_for_prompt(_sections(state))
        user = prompts.EXTRACT_USER.format(name=state["contract_name"], contract=contract)
        terms, errors, carrier = _extract_groups(state, list(TERM_MODELS), user, config, "extract_terms", True)
        return {"extraction": {"carrier_name": carrier, **terms}, "carrier": carrier, "attempts": 1,
                "errors": [*state.get("errors", []), *errors]}

    def verify(state: ReviewState) -> dict:
        text, secs = state["contract_text"], _sections(state)
        no_verifier = state.get("variant") == "no_verifier"
        result: dict[str, dict] = {}
        for term, spec in VERIFY_SPEC.items():
            t = state["extraction"].get(term) or {}
            if t.get("_failed"):
                problems = ["extraction failed - lawyer must check"]
            elif no_verifier:
                problems = []
            else:
                extra = {"text_fields": ("governing_law_state", "venue")} if term == "governing_law" else {}
                problems = verify_term(t, text, **spec, **extra)
            refs = sorted({r for q in t.get("quotes") or [] if (r := locate(q, secs, text))}, key=_ref_key)
            if not refs and no_verifier and t.get("section_ref"):
                refs = [t["section_ref"]]
            result[term] = {"verified": not problems, "problems": problems, "section_refs": refs,
                            "model_section_ref": t.get("section_ref", ""), "attempts": state.get("attempts", 1)}
        out: dict = {"verification": result}
        if state.get("attempts", 1) == 1:
            out["first_pass_problems"] = {k: v["problems"] for k, v in result.items() if v["problems"]}
        return out

    def route_after_verify(state: ReviewState) -> str:
        failed = [k for k, v in state["verification"].items() if v["problems"]]
        return "reextract" if failed and state.get("attempts", 1) < 2 else "rules"

    def reextract(state: ReviewState, config: RunnableConfig) -> dict:
        """The one retry: only the terms that failed verification, with the reasons."""
        failed = [k for k, v in state["verification"].items() if v["problems"] and not state["extraction"][k].get("_failed")]
        if not failed:
            return {"attempts": 2}
        problems = "\n".join(f"- {k}: {'; '.join(state['verification'][k]['problems'])}" for k in failed)
        user = prompts.RETRY_USER.format(problems=problems, terms=", ".join(failed), name=state["contract_name"],
                                         contract=render_for_prompt(_sections(state)))
        terms, errors, _ = _extract_groups(state, failed, user, config, "reextract_failed_terms", False)
        # A retry group that errored keeps its first-pass values (verify will flag them again).
        extraction = dict(state["extraction"])
        extraction.update({k: v for k, v in terms.items() if not v.get("_failed")})
        return {"extraction": extraction, "attempts": 2, "errors": [*state.get("errors", []), *errors]}

    def rules(state: ReviewState, config: RunnableConfig) -> dict:
        if state.get("variant") == "model_decides":
            return {"findings": _model_decides(llm, state, config)}
        findings = evaluate_rules(state["extraction"], state["verification"], state["contract_text"])
        return {"findings": [f.to_dict() for f in findings]}

    def draft(state: ReviewState, config: RunnableConfig) -> dict:
        findings = state["findings"]
        devs = [f for f in findings if f["status"] == "deviation"]
        checks = [f for f in findings if f["status"] == "needs_check"]
        carrier = state.get("carrier") or state["contract_name"]
        drafts: dict[str, dict] = {}
        summary = ""
        errors = list(state.get("errors", []))
        if devs or checks:
            dev_txt = "\n".join(f"- {f['rule_id']} | {f['playbook']} | {f['reason']} | quotes: {json.dumps(f['quotes'])}"
                                for f in devs) or "(none)"
            chk_txt = "\n".join(f"- {f['title']}: {f['reason']}" for f in checks) or "(none)"
            try:
                memo = llm.structured(MemoDraft, prompts.MEMO_SYSTEM,
                                      prompts.MEMO_USER.format(carrier=carrier, name=state["contract_name"],
                                                               deviations=dev_txt, needs_check=chk_txt),
                                      config=config, name="draft_memo")
                summary = memo.summary
                drafts = {i.rule_id: i.model_dump() for i in memo.items}
            except Exception as exc:
                if is_transient(exc):
                    raise
                errors.append(f"memo drafting failed, using playbook fallback text: {exc}")
        else:
            summary = ("Every playbook term was found, its quote verified against the contract text, and the rules "
                       "engine found no deviations. No redlines are proposed.")
        if not summary:
            summary = f"{len(devs)} deviation(s) from the playbook; see below. (Summary drafting failed.)"
        redlines = []
        for n, f in enumerate(sorted(devs, key=lambda f: list(RULES).index(f["rule_id"])), 1):
            d = drafts.get(f["rule_id"]) or {}
            redlines.append({
                "id": f"R{n}", "rule_id": f["rule_id"], "title": f["title"], "severity": f["severity"],
                "section_refs": f["section_refs"], "quotes": f["quotes"],
                "explanation": d.get("explanation") or f["reason"],
                "suggested_text": d.get("suggested_redline") or RULES[f["rule_id"]].standard_clause,
                "drafted_by": "model" if d.get("suggested_redline") else "playbook_fallback",
            })
        memo_md = memo_markdown(carrier=carrier, name=state["contract_name"], summary=summary,
                                findings=findings, redlines=redlines)
        return {"summary": summary, "redlines": redlines, "memo_md": memo_md, "errors": errors}

    def lawyer_review(state: ReviewState) -> dict:
        answer = interrupt({
            "type": "lawyer_review",
            "instructions": "Approve, edit or reject each redline. Only approved or edited redlines go into redlines.md. "
                            "Nothing is sent to the carrier.",
            "carrier": state.get("carrier"),
            "memo_md": state["memo_md"],
            "redlines": [{k: r[k] for k in ("id", "rule_id", "title", "severity", "section_refs", "quotes",
                                            "explanation", "suggested_text")} for r in state["redlines"]],
            "allowed_actions": sorted(VALID_ACTIONS),
        })
        return {"decisions": normalize_decisions(answer, state["redlines"])}

    def finalize(state: ReviewState) -> dict:
        md = redlines_markdown(carrier=state.get("carrier") or state["contract_name"], name=state["contract_name"],
                               redlines=state["redlines"], decisions=state["decisions"])
        return {"redlines_md": md}

    g = StateGraph(ReviewState)
    g.add_node("ingest", ingest)
    g.add_node("extract", extract)
    g.add_node("verify", verify)
    g.add_node("reextract", reextract)
    g.add_node("rules", rules)
    g.add_node("draft", draft)
    g.add_node("lawyer_review", lawyer_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", lambda s: "rules" if s.get("variant") == "model_decides" else "extract",
                            ["rules", "extract"])
    g.add_edge("extract", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["reextract", "rules"])
    g.add_edge("reextract", "verify")
    g.add_edge("rules", "draft")
    g.add_edge("draft", "lawyer_review")
    g.add_edge("lawyer_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name="contract-review")


def normalize_decisions(answer: Any, redlines: list[dict]) -> dict[str, dict]:
    """Lawyer decisions keyed by redline id. Fail closed: a missing, unknown or
    malformed decision, or an edit with no text, counts as a reject."""
    raw: dict[str, Any] = {}
    if isinstance(answer, dict):
        items = answer.get("decisions", answer)
        if isinstance(items, list):
            raw = {d.get("id"): d for d in items if isinstance(d, dict)}
        elif isinstance(items, dict):
            raw = {k: (v if isinstance(v, dict) else {"action": v}) for k, v in items.items()}
    out: dict[str, dict] = {}
    for r in redlines:
        d = raw.get(r["id"]) or {}
        action = str(d.get("action", "")).lower().strip()
        text = (d.get("text") or "").strip()
        note = (d.get("note") or "").strip()
        if action not in VALID_ACTIONS:
            out[r["id"]] = {"action": "reject", "note": note or "no valid decision given (treated as reject)"}
        elif action == "edit" and not text:
            out[r["id"]] = {"action": "reject", "note": "edit with no replacement text (treated as reject)"}
        else:
            out[r["id"]] = {"action": action, "text": text, "note": note}
    return out


def _ref_key(ref: str):
    parts = re.findall(r"\d+|[A-Za-z]+", ref)
    return [(0, int(p)) if p.isdigit() else (1, p) for p in parts]


def _failed_term() -> dict:
    return {"found": False, "section_ref": "", "quotes": [], "_failed": True}


def _model_decides(llm: StructuredLLM, state: ReviewState, config) -> list[dict]:
    """Ablation: the model judges each rule. Code does not check quotes or values."""
    playbook = "\n".join(f"- {r.rule_id}: {r.playbook}" for r in RULES.values())
    res: ModelJudgments = llm.structured(
        ModelJudgments, prompts.MODEL_DECIDES_SYSTEM.format(playbook=playbook),
        prompts.EXTRACT_USER.format(name=state["contract_name"],
                                    contract=render_for_prompt([Section(**s) for s in state["sections"]])),
        config=config, name="model_decides_compliance")
    by_id = {j.rule_id: j for j in res.judgments}
    out = []
    for r in RULES.values():
        j = by_id.get(r.rule_id)
        status = j.status if j else "compliant"   # the ablation trusts the model, including omissions
        out.append(Finding(r.rule_id, r.title, r.severity, status, r.playbook, "model judgment",
                           "Model judged this rule." if j else "Model omitted this rule.", [],
                           [j.quote] if j and j.quote else []).to_dict())
    return out
