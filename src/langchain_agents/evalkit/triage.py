"""The model's one job in the kit: label each mined trace and draft what a good
answer must do (a rubric), so a person can accept, edit or reject it. It never
writes the dataset and never decides what ships."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from ..core.config import get_model
from ..core.cost import CostTracker
from .mine import Candidate


class Triage(BaseModel):
    category: str = Field(description="Short snake_case label for what the user wanted, e.g. load_status, detention_question")
    looks_wrong: bool = Field(description="True if the agent's answer looks wrong, unsafe, incomplete or off-policy")
    why: str = Field(description="One sentence on what the agent did well or badly")
    must_do: list[str] = Field(description="2-4 checkable things a correct answer must do or contain")
    must_not: list[str] = Field(description="0-3 things a correct answer must not do")


PROMPT = """You are helping an engineer turn production traces into an evaluation dataset.

Customer context:
{context}

Below is one real conversation turn from the customer's agent. Label it and write a short rubric a
reviewer could check ANOTHER answer against. Rubric items must be concrete and checkable ("states detention
is $75/hour after 2 free hours", "does not promise a credit"), not vague ("is helpful").

Write the rubric from what the USER needed and what the POLICY requires, not from how this particular reply
happened to word things. Only require facts any correct answer must include. Don't require incidental
details (a weekday, "yesterday", an MC number, a formatting choice) that a different correct answer could
reasonably leave out. If a consequential action needs human approval, "pauses for approval of X" is a
correct outcome.

USER INPUT:
{input}

TOOL CALLS AND RESULTS IN THIS TURN (the agent's evidence; facts that appear here are not fabricated):
{evidence}

AGENT OUTPUT:
{output}

SIGNALS FROM THE TRACE: {signals}"""


def triage_all(cands: list[Candidate], context: str, model_id: str | None = None,
               workers: int = 6) -> tuple[list[dict], dict]:
    tracker = CostTracker()
    llm = get_model(model_id).with_structured_output(Triage, method="json_schema")

    def one(c: Candidate) -> dict:
        try:
            t = llm.invoke(PROMPT.format(context=context[:4000], input=c.input_text[:3000],
                                         output=c.output_text[:3000] or "(empty)",
                                         evidence=c.tool_evidence or "(none)",
                                         signals=", ".join(c.signals) or "none"),
                           config={"callbacks": [tracker], "run_name": "evalkit-triage"})
            return {**c.to_json(), "triage": t.model_dump()}
        except Exception as exc:  # noqa: BLE001 - a failed label goes to the human, never dropped
            return {**c.to_json(), "triage": None, "triage_error": f"{type(exc).__name__}: {exc}"[:300]}

    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, cands))
    return out, tracker.summary()
