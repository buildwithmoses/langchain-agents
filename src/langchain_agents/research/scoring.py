"""Eval scoring. Two kinds of metric, kept apart on purpose:

- From code (exact): citation precision of the raw draft and of the final brief,
  claims dropped, unanswerable handling, cost, latency.
- From a judge model (a different family from the agent, `judge_model_id()`):
  answer correctness against the reference answer, and whether each verified
  quote actually supports its claim (the verifier proves the quote exists; it
  can't prove the quote means what the claim says).
"""

from __future__ import annotations

import json
from statistics import median

from pydantic import BaseModel, ConfigDict, Field

from langchain_agents.core.config import get_model, judge_model_id


class Correctness(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    score: float = Field(description="1.0 fully correct and complete on the key facts, 0.5 partly, 0.0 wrong or missing.")
    reason: str = Field(description="One sentence.")


class SupportItem(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    n: int = Field(description="Claim number.")
    supported: bool = Field(description="True if the quote, on its own, supports everything the claim states.")


class Support(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    items: list[SupportItem] = Field(description="One entry per claim, in order.")


CORRECTNESS_PROMPT = """You grade a research brief against a reference answer.
Question: {question}
Reference answer (ground truth): {reference}

Brief answer: {answer}
Brief findings:
{claims}

Score 1.0 if the brief's answer and findings state the reference's key facts correctly (wording may differ; \
extra correct detail is fine), 0.5 if some key facts are missing or one is wrong, 0.0 if mostly wrong or missing. \
Any number that contradicts the reference counts as wrong."""

SUPPORT_PROMPT = """For each numbered claim, decide whether its quote, on its own, supports what the claim states. \
Paraphrase, shortening and dropping qualifiers are fine. Mark it NOT supported only if the claim asserts a fact, \
number, cause or conclusion that the quote does not contain, or contradicts the quote.

{items}"""


def _judge(schema):
    return get_model(judge_model_id()).with_structured_output(schema, method="json_schema")


def judge_correctness(question: str, reference: str, final: dict, callbacks=None) -> Correctness:
    claims = "\n".join(f"[{c['n']}] {c['claim']}" for c in final.get("claims") or []) or "(none)"
    prompt = CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=final.get("answer", ""),
                                       claims=claims)
    return _judge(Correctness).invoke(prompt, config={"callbacks": callbacks or []})


def judge_support(final: dict, callbacks=None) -> list[bool]:
    claims = final.get("claims") or []
    if not claims:
        return []
    items = "\n\n".join(f"Claim {c['n']}: {c['claim']}\nQuote: \"{c['quote']}\"" for c in claims)
    res = _judge(Support).invoke(SUPPORT_PROMPT.format(items=items), config={"callbacks": callbacks or []})
    by_n = {i.n: i.supported for i in res.items}
    return [bool(by_n.get(c["n"], False)) for c in claims]  # missing verdict counts as unsupported


def summarize(rows: list[dict]) -> dict:
    """rows: one dict per example with code + judge scores."""
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    raw_cited = sum(r["raw_claims"] for r in rows)
    raw_ok = sum(r["raw_verified"] for r in rows)
    final_cited = sum(r["final_claims"] for r in rows)
    final_ok = sum(r["final_reverified"] for r in rows)
    judged = sum(len(r["support"]) for r in rows)
    unsupported = sum(sum(1 for s in r["support"] if not s) for r in rows)
    lat = [r["latency_s"] for r in rows]
    return {
        "answer_correctness": round(sum(r["correctness"] for r in ans) / len(ans), 4) if ans else None,
        "citation_precision_final": round(final_ok / final_cited, 4) if final_cited else 1.0,
        "citation_precision_raw": round(raw_ok / raw_cited, 4) if raw_cited else 1.0,
        "unsupported_claim_rate": round(unsupported / judged, 4) if judged else 0.0,
        "unanswerable_handled": round(sum(r["refused"] for r in unans) / len(unans), 4) if unans else None,
        "expected_source_read": round(sum(r["expected_source_read"] for r in ans) / len(ans), 4) if ans else None,
        "raw_claims": raw_cited, "raw_failed": raw_cited - raw_ok,
        "raw_failed_by_reason": _merge([r["raw_failed_by_reason"] for r in rows]),
        "recovered_on_revision": sum(r["recovered_on_revision"] for r in rows),
        "final_claims": final_cited, "final_dropped": sum(r["final_dropped"] for r in rows),
        "examples": len(rows), "errors": sum(1 for r in rows if r.get("error")),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 4),
        "cost_usd_per_question": round(sum(r["cost_usd"] for r in rows) / len(rows), 4) if rows else 0.0,
        "latency_p50_s": round(median(lat), 1) if lat else 0.0,
        "latency_max_s": round(max(lat), 1) if lat else 0.0,
    }


def _merge(dicts: list[dict]) -> dict:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
