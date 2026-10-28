"""Scripted stand-in for the model, for offline tests (no keys, no network).

FakeLLM returns canned term extractions for whatever subset schema it is asked
for, and a memo item for every rule_id listed in the memo prompt. It records
every call so tests can assert on the retry path."""

from __future__ import annotations

import copy
import re

from .schemas import MemoDraft, RedlineDraft

# A correct extraction of fixtures/contracts/01-bluegrass-haulers.md (fully compliant).
COMPLIANT_01 = {
    "carrier_name": "Bluegrass Haulers LLC",
    "indemnity": {"found": True, "section_ref": "7.1", "direction": "mutual", "quotes": [
        "Each Party (the \"Indemnifying Party\") shall defend, indemnify and hold harmless the other Party"]},
    "cargo_liability": {"found": True, "section_ref": "6.2", "cargo_limit_usd": 100000, "released_rates": False, "quotes": [
        "Carrier's liability shall be for the full actual value of the cargo, up to One Hundred Thousand Dollars ($100,000) per Load."]},
    "liability_cap": {"found": True, "section_ref": "9.2", "cap_present": False, "cap_covers_carrier_negligence": False, "quotes": [
        "Nothing in this Section 9 limits (a) Carrier's liability for cargo under Section 6"]},
    "payment_terms": {"found": True, "section_ref": "5.2", "payment_days": 30, "quotes": [
        "Ridgeline shall pay each undisputed invoice within thirty (30) days after receipt of the invoice and all required documents."]},
    "non_solicit": {"found": True, "section_ref": "10.1", "non_solicit_months": 12, "quotes": [
        "During the Term and for twelve (12) months after the last Load tendered by Ridgeline to Carrier"]},
    "governing_law": {"found": True, "section_ref": "13.1, 13.2", "governing_law_state": "Tennessee",
                      "venue": "Shelby County, Tennessee", "quotes": [
        "This Agreement shall be governed by the laws of the State of Tennessee",
        "shall be brought exclusively in the state or federal courts located in Shelby County, Tennessee."]},
    "termination": {"found": True, "section_ref": "11.2", "termination_notice_days": 30, "either_party": True, "quotes": [
        "Either Party may terminate this Agreement for any reason upon thirty (30) days' prior written notice to the other Party."]},
    "auto_renewal": {"found": True, "section_ref": "11.1", "auto_renewal": True, "renewal_cancel_notice_days": 30, "quotes": [
        "shall automatically renew for successive one (1) year terms unless either Party gives written notice of non-renewal at least thirty (30) days before the end of the then-current term."]},
    "insurance": {"found": True, "section_ref": "8.1, 8.2", "auto_liability_usd": 1000000, "cargo_insurance_usd": 100000,
                  "general_liability_usd": 1000000, "certificate_holder_named": True, "quotes": [
        "limits of not less than One Million Dollars ($1,000,000) combined single limit per occurrence",
        "motor truck cargo insurance with limits of not less than One Hundred Thousand Dollars ($100,000) per Load",
        "commercial general liability insurance with limits of not less than One Million Dollars ($1,000,000) per occurrence",
        "naming Ridgeline Freight, LLC as certificate holder"]},
}


def compliant_extraction(**overrides) -> dict:
    data = copy.deepcopy(COMPLIANT_01)
    for term, patch in overrides.items():
        if isinstance(patch, dict) and term in data:
            data[term].update(patch)
        else:
            data[term] = patch
    return data


class FakeLLM:
    def __init__(self, extraction: dict | None = None, retry_extraction: dict | None = None,
                 fail_names: tuple[str, ...] = ()) -> None:
        self.extraction = extraction or compliant_extraction()
        self.retry_extraction = retry_extraction
        self.fail_names = fail_names
        self.calls: list[dict] = []

    def structured(self, schema, system, user, *, config=None, name=""):
        self.calls.append({"name": name, "schema": schema.__name__, "fields": list(schema.model_fields)})
        if any(name.startswith(f) for f in self.fail_names):
            raise ValueError(f"scripted non-transient failure in {name}")
        if schema is MemoDraft:
            ids = re.findall(r"^- (\w+) \|", user, re.M)
            return MemoDraft(summary=f"Fake summary: {len(ids)} deviation(s).", items=[
                RedlineDraft(rule_id=i, explanation=f"Fake explanation for {i}.", suggested_redline=f"Fake redline text for {i}.")
                for i in ids])
        source = self.retry_extraction if name.startswith("reextract") and self.retry_extraction else self.extraction
        return schema.model_validate({f: source[f] for f in schema.model_fields})
