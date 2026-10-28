"""Typed extraction schema. Every field is required (no defaults): smaller models
silently omit optional fields, and an omitted field must never read as "compliant".
Nullable fields are still required - the model has to say null explicitly."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

QUOTES = ("Verbatim quotes copied character-for-character from the contract that establish this term "
          "(1-3 short quotes, each a sentence or less). If the value depends on a defined term or an "
          "exhibit, ALSO quote the definition/exhibit text that gives the value. If a later clause "
          "overrides or reverses an earlier one, quote both. Empty list if not found.")


class TermBase(BaseModel):
    found: bool = Field(description="True if the contract contains a clause addressing this term.")
    section_ref: str = Field(description="Section number(s) where the clause is, e.g. '6.2' or 'Exhibit B, B.1'. Empty string if not found.")
    quotes: list[str] = Field(description=QUOTES)


class Indemnity(TermBase):
    direction: Literal["mutual", "carrier_indemnifies_ridgeline", "ridgeline_indemnifies_carrier", "none"] = Field(
        description="Who indemnifies whom, AFTER applying any clause anywhere in the contract that overrides or "
        "voids indemnity. mutual = each party indemnifies the other for its own negligence; "
        "carrier_indemnifies_ridgeline = only the carrier indemnifies the broker; "
        "ridgeline_indemnifies_carrier = only the broker (Ridgeline) indemnifies the carrier; none = no indemnity.")


class CargoLiability(TermBase):
    cargo_limit_usd: int | None = Field(description="The maximum dollar amount the carrier is liable for per load/shipment for cargo loss or damage, "
                                        "resolving defined terms and exhibits. null if full value with no stated cap.")
    released_rates: bool = Field(description="True if the contract (body, definitions or exhibits) states rates are 'released rates' or based on a released value.")


class PaymentTerms(TermBase):
    payment_days: int | None = Field(description="Days Ridgeline (the broker) has to pay the carrier's invoice under standard terms (not quick pay).")


class NonSolicit(TermBase):
    non_solicit_months: int | None = Field(description="Months after the last load/shipment during which the carrier may not solicit the broker's shippers/customers.")


class GoverningLaw(TermBase):
    governing_law_state: str | None = Field(description="Governing-law state exactly as written in the contract (e.g. 'Tennessee', 'TN', 'Delaware').")
    venue: str | None = Field(description="Venue/forum for disputes exactly as written (e.g. 'Shelby County, Tennessee'). null if no venue clause.")


class Termination(TermBase):
    termination_notice_days: int | None = Field(description="Days of written notice required to terminate for convenience/without cause, resolving defined terms "
                                                "such as 'Notice Period'.")
    either_party: bool = Field(description="True if either party may terminate on that notice (not only one side).")


class AutoRenewal(TermBase):
    auto_renewal: bool = Field(description="True if the agreement renews automatically.")
    renewal_cancel_notice_days: int | None = Field(description="Days before the end of a term that notice of non-renewal/cancellation must be given. null if no auto-renewal.")


class Insurance(TermBase):
    auto_liability_usd: int | None = Field(description="Required automobile liability limit in USD (look in exhibits too).")
    cargo_insurance_usd: int | None = Field(description="Required motor truck CARGO INSURANCE limit in USD. This is insurance, not the cargo liability cap.")
    general_liability_usd: int | None = Field(description="Required commercial general liability limit in USD.")
    certificate_holder_named: bool = Field(description="True only if the contract says Ridgeline is named as CERTIFICATE HOLDER on certificates of insurance.")


class LiabilityCap(TermBase):
    """The general limitation-of-liability clause. The per-shipment cargo loss/damage limit
    (released value) is a different term (cargo_liability) - do not report it here."""
    found: bool = Field(description="True if the contract has ANY limitation-of-liability or limitation-of-damages clause, "
                        "including a clause that only waives consequential/punitive damages. Quote it.")
    cap_present: bool = Field(description="True if there is a cap on the carrier's OVERALL or aggregate liability under the agreement "
                              "(e.g. 'aggregate liability shall not exceed the charges paid'). A consequential-damages waiver "
                              "or the per-shipment cargo loss/damage limit does NOT count - those belong to other terms.")
    cap_covers_carrier_negligence: bool = Field(description="True if that overall cap applies to liability arising from the CARRIER'S OWN "
                                                "negligence. False if there is no overall cap, or the contract says the limitation does not "
                                                "limit liability for a party's own negligence.")


class Extraction(BaseModel):
    """All playbook terms for one carrier agreement."""
    carrier_name: str = Field(description="The carrier's legal name.")
    indemnity: Indemnity
    cargo_liability: CargoLiability
    payment_terms: PaymentTerms
    non_solicit: NonSolicit
    governing_law: GoverningLaw
    termination: Termination
    auto_renewal: AutoRenewal
    insurance: Insurance
    liability_cap: LiabilityCap


TERM_MODELS: dict[str, type[TermBase]] = {
    name: field.annotation  # type: ignore[misc]
    for name, field in Extraction.model_fields.items() if name != "carrier_name"
}


# Anthropic's strict structured output rejects the full 9-term schema ("The compiled
# grammar is too large"), so extraction runs as three parallel calls of three terms.
EXTRACTION_GROUPS: list[list[str]] = [
    ["indemnity", "cargo_liability", "liability_cap"],
    ["payment_terms", "non_solicit", "governing_law"],
    ["termination", "auto_renewal", "insurance"],
]


def subset_schema(terms: list[str], *, with_carrier: bool = False) -> type[BaseModel]:
    """Schema with only the given terms (one extraction group, or the retry of failed terms)."""
    fields: dict = {}
    if with_carrier:
        fields["carrier_name"] = (str, Field(description="The carrier's legal name."))
    fields.update({t: (TERM_MODELS[t], ...) for t in terms})
    return create_model("TermExtraction", **fields)  # type: ignore[call-overload]


def chunk_terms(terms: list[str]) -> list[list[str]]:
    """Split terms into groups no larger than the extraction groups, keeping group order."""
    out = []
    for g in EXTRACTION_GROUPS:
        part = [t for t in g if t in terms]
        if part:
            out.append(part)
    return out


# ---- memo drafting ---------------------------------------------------------

class RedlineDraft(BaseModel):
    rule_id: str = Field(description="The rule_id of the deviation this item addresses (copy exactly).")
    explanation: str = Field(description="2-4 plain-English sentences for the General Counsel: what the clause says, why it deviates "
                             "from the playbook, and the business risk to Ridgeline.")
    suggested_redline: str = Field(description="Replacement contract language Ridgeline would propose, written as a clause.")


class MemoDraft(BaseModel):
    summary: str = Field(description="3-5 sentence plain-English summary of the agreement's risk for the General Counsel.")
    items: list[RedlineDraft] = Field(description="Exactly one item per deviation listed in the input.")


# ---- eval ablation: the model judges compliance itself ------------------------

class RuleJudgment(BaseModel):
    rule_id: str
    status: Literal["compliant", "deviation"]
    quote: str


class ModelJudgments(BaseModel):
    carrier_name: str
    judgments: list[RuleJudgment] = Field(description="One judgment per rule_id listed in the instructions.")
