"""Ridgeline's carrier-agreement playbook as deterministic rules.

Source: fixtures/ridgeline/COMPANY.md, "Contract playbook". The model extracts
values; this module decides compliance. Each rule reads one extracted term.

Statuses: "compliant", "deviation", or "needs_check" (the term could not be
verified - a lawyer must check it; never treated as compliant).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .verify import normalize

MIN_CARGO_LIABILITY = 100_000
MIN_PAYMENT_DAYS = 30
MIN_NON_SOLICIT_MONTHS = 12
STANDARD_TERMINATION_DAYS = 30
SIGNOFF_TERMINATION_DAYS = 60
MAX_RENEWAL_CANCEL_DAYS = 30
MIN_AUTO_LIABILITY = 1_000_000
MIN_CARGO_INSURANCE = 100_000
MIN_GENERAL_LIABILITY = 1_000_000
GOVERNING_STATE = "TN"
VENUE_COUNTY = "shelby"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    term: str
    severity: str           # high | medium | low (termination 31-60 days is downgraded to low)
    title: str
    playbook: str
    standard_clause: str    # fallback redline if the drafting model fails


RULES: dict[str, Rule] = {r.rule_id: r for r in [
    Rule("indemnity", "indemnity", "high", "Indemnity",
         "Carrier indemnifies Ridgeline for carrier negligence. Mutual is acceptable; one-way in the carrier's favor is not.",
         "Carrier shall defend, indemnify and hold harmless Ridgeline and its customers from and against all claims, losses "
         "and expenses arising out of the negligence, willful misconduct or breach of this Agreement by Carrier, its "
         "employees, drivers or agents. Ridgeline shall indemnify Carrier to the extent of Ridgeline's own negligence."),
    Rule("cargo_liability", "cargo_liability", "high", "Cargo liability",
         "Full value up to at least $100,000 per load (Carmack). No released rates capping liability below $100,000 without legal sign-off.",
         "Carrier shall be liable under 49 U.S.C. § 14706 for the full actual value of lost or damaged cargo, up to not less "
         "than One Hundred Thousand Dollars ($100,000) per load. No released rates or tariff limitations shall apply."),
    Rule("liability_cap_own_negligence", "liability_cap", "high", "Limitation of liability",
         "Liability caps that apply to the carrier's own negligence are not acceptable.",
         "No limitation of liability in this Agreement shall apply to Carrier's liability for its own negligence or willful "
         "misconduct, its cargo liability, or its indemnification obligations."),
    Rule("payment_terms", "payment_terms", "medium", "Payment terms",
         "Net 30 or longer for Ridgeline to pay. Quick pay optional.",
         "Ridgeline shall pay each undisputed invoice within thirty (30) days after receipt of the invoice and signed proof "
         "of delivery. Carrier may elect quick pay within two (2) days for a fee of three percent (3%)."),
    Rule("non_solicit", "non_solicit", "medium", "Non-solicitation",
         "No back-solicitation of Ridgeline's shippers for 12 months after the last load.",
         "During the term and for twelve (12) months after the last load tendered by Ridgeline, Carrier shall not solicit "
         "or accept freight directly from any shipper introduced to Carrier by Ridgeline."),
    Rule("governing_law", "governing_law", "medium", "Governing law",
         "Governing law: Tennessee.",
         "This Agreement shall be governed by the laws of the State of Tennessee, except to the extent preempted by federal law."),
    Rule("venue", "governing_law", "medium", "Venue",
         "Venue: Shelby County, Tennessee.",
         "Any action arising out of this Agreement shall be brought exclusively in the state or federal courts located in "
         "Shelby County, Tennessee."),
    Rule("termination_notice", "termination", "medium", "Termination notice",
         "Either party may terminate on 30 days' written notice. Longer than 60 days needs legal sign-off.",
         "Either party may terminate this Agreement for any reason upon thirty (30) days' prior written notice."),
    Rule("auto_renewal", "auto_renewal", "medium", "Auto-renewal",
         "Auto-renewal is acceptable only if notice to cancel is 30 days or less.",
         "This Agreement renews automatically for successive one (1) year terms unless either party gives written notice "
         "of non-renewal at least thirty (30) days before the end of the then-current term."),
    Rule("insurance_auto", "insurance", "medium", "Auto liability insurance",
         "Auto liability insurance at least $1,000,000.",
         "Automobile liability insurance with a combined single limit of not less than One Million Dollars ($1,000,000)."),
    Rule("insurance_cargo", "insurance", "medium", "Cargo insurance",
         "Cargo insurance at least $100,000.",
         "Motor truck cargo insurance with a limit of not less than One Hundred Thousand Dollars ($100,000) per load."),
    Rule("insurance_gl", "insurance", "medium", "General liability insurance",
         "General liability insurance at least $1,000,000.",
         "Commercial general liability insurance with limits of not less than One Million Dollars ($1,000,000) per occurrence."),
    Rule("certificate_holder", "insurance", "low", "Certificate holder",
         "Ridgeline named as certificate holder on the carrier's insurance.",
         "Carrier shall cause its insurers to issue certificates of insurance naming Ridgeline Freight, LLC as certificate "
         "holder, with thirty (30) days' prior written notice of cancellation."),
]}
HIGH_SEVERITY = {r.rule_id for r in RULES.values() if r.severity == "high"}

# Terms whose absence is acceptable. A model claim of "absent" is only accepted if a
# keyword scan of the contract agrees; otherwise the term becomes needs_check.
ABSENCE_OK_GUARDS = {
    "auto_renewal": re.compile(r"auto(?:matic(?:ally)?)?[- ]?renew|renew(?:s|ed)? automatically|successive .{0,40}terms?|evergreen", re.I),
    "liability_cap": re.compile(r"limitation of liability|aggregate liability|maximum liability|in no event shall .{0,80}liabil", re.I),
}

US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
    "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
    "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas",
    "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
}
_NAME_TO_ABBR = {v: k for k, v in US_STATES.items()}
_ALIASES = {"tenn": "TN", "tenn.": "TN", "del": "DE", "del.": "DE"}


def normalize_state(raw: str | None) -> str | None:
    """'Tennessee', 'TN', 'the State of Tennessee', 'Tenn.' -> 'TN'. None if unknown."""
    if not raw:
        return None
    t = normalize(raw).replace(".", " ").strip()
    t = re.sub(r"^(the )?(laws? of )?(the )?(state|commonwealth) of ", "", t).strip(" ,")
    t = re.sub(r"\s+law$", "", t).strip()
    if t.upper() in US_STATES:
        return t.upper()
    if t in _NAME_TO_ABBR:
        return _NAME_TO_ABBR[t]
    if t in _ALIASES:
        return _ALIASES[t]
    states = states_mentioned(t)
    return states[0] if len(states) == 1 else None


def states_mentioned(text: str) -> list[str]:
    """States named in free text. Full names match case-insensitively; two-letter
    abbreviations only as UPPERCASE tokens ("TN"), so the words "in"/"or"/"me" are
    not read as Indiana/Oregon/Maine."""
    t = " " + re.sub(r"[^a-z ]", " ", normalize(text)) + " "
    t = re.sub(r"\s+", " ", t)
    found = [abbr for abbr, name in US_STATES.items() if f" {name} " in t]
    if "WV" in found and "VA" in found and " virginia " not in t.replace("west virginia", ""):
        found.remove("VA")
    for tok in re.findall(r"\b([A-Z]{2})\b", text or ""):
        if tok in US_STATES and tok not in found:
            found.append(tok)
    return found


def normalize_county(raw: str | None) -> tuple[str | None, list[str]]:
    """'Shelby County, Tennessee' -> ('shelby', ['TN']). Returns the county name and any states mentioned."""
    if not raw:
        return None, []
    t = normalize(raw)
    m = re.search(r"([a-z][a-z .'-]*?)\s+county", t)
    states = states_mentioned(raw)
    county = m.group(1).strip() if m else None
    if county:
        county = re.sub(r"^.*\b(?:in|of|located in|courts of|sitting in|the)\s+", "", county).strip()
    return county, states


@dataclass
class Finding:
    rule_id: str
    title: str
    severity: str
    status: str                       # compliant | deviation | needs_check
    playbook: str
    found_value: str
    reason: str
    section_refs: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _usd(v: int | None) -> str:
    return "not stated" if v is None else f"${v:,}"


def evaluate_rules(extraction: dict, verification: dict, contract_text: str = "") -> list[Finding]:
    """Apply every rule. `extraction` is Extraction.model_dump(); `verification` maps
    term -> {"verified": bool, "problems": [...], "section_refs": [...]}."""
    out: list[Finding] = []
    for rule in RULES.values():
        term = extraction.get(rule.term) or {}
        ver = verification.get(rule.term) or {}
        refs, quotes = list(ver.get("section_refs") or []), list(term.get("quotes") or [])

        def make(status: str, value: str, reason: str, severity: str | None = None) -> Finding:
            return Finding(rule.rule_id, rule.title, severity or rule.severity, status, rule.playbook,
                           value, reason, refs, quotes)

        # Fail closed: a term whose quote/value was not verified (or that has no
        # verification record at all) is never judged - the lawyer must check it.
        if not ver.get("verified"):
            problems = "; ".join(ver.get("problems") or ["unverified"])
            out.append(make("needs_check", "unverified",
                            f"Not found - lawyer must check. The extracted quote/value could not be verified against the contract text ({problems})."))
            continue
        out.append(_RULE_FNS[rule.rule_id](rule, term, contract_text, make))
    return out


def _absent(make, msg: str, status: str = "deviation"):
    return make(status, "not present", msg)


def _r_indemnity(rule, t, text, make):
    if not t.get("found"):
        return _absent(make, "No indemnity clause found: the carrier does not indemnify Ridgeline for its negligence.")
    d = t.get("direction")
    if d in ("mutual", "carrier_indemnifies_ridgeline"):
        return make("compliant", d, "Carrier indemnifies Ridgeline for carrier negligence.")
    return make("deviation", d or "unknown", "Indemnity does not run from the carrier to Ridgeline "
                f"({'one-way in the carrier' + chr(39) + 's favor' if d == 'ridgeline_indemnifies_carrier' else 'no carrier indemnity'}).")


def _r_cargo(rule, t, text, make):
    if not t.get("found"):
        return make("needs_check", "not present", "No cargo liability clause found - lawyer must check what liability standard applies.")
    limit, released = t.get("cargo_limit_usd"), bool(t.get("released_rates"))
    value = f"limit {_usd(limit)}{', released rates' if released else ''}"
    if limit is not None and limit < MIN_CARGO_LIABILITY:
        return make("deviation", value, f"Cargo liability is capped at {_usd(limit)} per load, below the $100,000 minimum"
                    f"{' (released rates)' if released else ''}.")
    if released and limit is None:
        return make("needs_check", value, "Released rates apply but no dollar cap was extracted - lawyer must check the released value.")
    return make("compliant", value, "Cargo liability is at least $100,000 per load or full value.")


def _r_cap(rule, t, text, make):
    if not t.get("found"):
        if ABSENCE_OK_GUARDS["liability_cap"].search(text):
            return make("needs_check", "not extracted", "The model reported no limitation-of-liability clause, but the contract "
                        "contains limitation language - lawyer must check.")
        return make("compliant", "not present", "No limitation-of-liability clause (keyword scan agrees).")
    if t.get("cap_covers_carrier_negligence"):
        return make("deviation", "cap applies to carrier's own negligence",
                    "A liability cap applies to liability arising from the carrier's own negligence.")
    return make("compliant", "cap excludes carrier negligence" if t.get("cap_present") else "no cap",
                "No cap applies to the carrier's own negligence.")


def _r_payment(rule, t, text, make):
    days = t.get("payment_days")
    if not t.get("found") or days is None:
        return make("needs_check", "not stated", "Payment term not found - lawyer must check.")
    if days < MIN_PAYMENT_DAYS:
        return make("deviation", f"net {days}", f"Ridgeline must pay in {days} days, shorter than net 30.")
    return make("compliant", f"net {days}", f"Net {days} is not shorter than net 30.")


def _r_nonsolicit(rule, t, text, make):
    if not t.get("found"):
        return _absent(make, "No non-solicitation clause protecting Ridgeline's shippers.")
    months = t.get("non_solicit_months")
    if months is None:
        return make("needs_check", "duration not stated", "Non-solicit duration unclear - lawyer must check.")
    if months < MIN_NON_SOLICIT_MONTHS:
        return make("deviation", f"{months} months", f"Non-solicit runs {months} months, shorter than 12 months after the last load.")
    return make("compliant", f"{months} months", "Non-solicit is at least 12 months.")


def _r_law(rule, t, text, make):
    raw = t.get("governing_law_state")
    if not t.get("found") or not raw:
        return _absent(make, "No governing-law clause; playbook requires Tennessee.")
    st = normalize_state(raw)
    if st == GOVERNING_STATE:
        return make("compliant", raw, "Governed by Tennessee law.")
    if st is None:
        return make("needs_check", raw, f"Could not recognise governing-law state {raw!r} - lawyer must check.")
    return make("deviation", raw, f"Governed by {raw} law, not Tennessee.")


def _r_venue(rule, t, text, make):
    raw = t.get("venue")
    if not t.get("found") or not raw:
        return _absent(make, "No venue clause; playbook requires Shelby County, Tennessee.")
    county, states = normalize_county(raw)
    other_states = [s for s in states if s != GOVERNING_STATE]
    if county == VENUE_COUNTY and not other_states:
        return make("compliant", raw, "Venue is Shelby County, Tennessee.")
    if county is None:
        return make("needs_check", raw, f"Could not identify a county in venue {raw!r} - lawyer must check.")
    return make("deviation", raw, f"Venue is {raw}, not Shelby County, Tennessee.")


def _r_termination(rule, t, text, make):
    days = t.get("termination_notice_days")
    if not t.get("found") or days is None:
        return make("needs_check", "not stated", "Termination-for-convenience notice not found - lawyer must check.")
    if not t.get("either_party"):
        return make("deviation", f"{days} days, one side only", "Only one party may terminate for convenience; playbook requires either party.")
    if days > SIGNOFF_TERMINATION_DAYS:
        return make("deviation", f"{days} days", f"{days}-day termination notice exceeds 60 days and needs legal sign-off.")
    if days > STANDARD_TERMINATION_DAYS:
        return make("deviation", f"{days} days", f"{days}-day termination notice is longer than the 30-day standard "
                    "(within 60 days, so no sign-off needed, but should be negotiated to 30).", severity="low")
    return make("compliant", f"{days} days", f"{days}-day notice meets the 30-day standard.")


def _r_renewal(rule, t, text, make):
    if not t.get("found") or not t.get("auto_renewal"):
        if not t.get("found") and ABSENCE_OK_GUARDS["auto_renewal"].search(text):
            return make("needs_check", "not extracted", "The model reported no auto-renewal clause, but the contract contains "
                        "renewal language - lawyer must check.")
        return make("compliant", "no auto-renewal", "No auto-renewal.")
    days = t.get("renewal_cancel_notice_days")
    if days is None:
        return make("needs_check", "auto-renews, notice not stated", "Auto-renewal cancel notice not found - lawyer must check.")
    if days > MAX_RENEWAL_CANCEL_DAYS:
        return make("deviation", f"{days} days to cancel", f"Auto-renews unless cancelled {days} days ahead; playbook allows at most 30.")
    return make("compliant", f"{days} days to cancel", "Auto-renewal cancel notice is 30 days or less.")


def _insurance(field_name: str, minimum: int, label: str):
    def fn(rule, t, text, make):
        if not t.get("found"):
            return make("needs_check", "not present", "No insurance clause found - lawyer must check.")
        v = t.get(field_name)
        if v is None:
            return make("needs_check", "not stated", f"{label} limit not found - lawyer must check (check exhibits).")
        if v < minimum:
            return make("deviation", _usd(v), f"{label} of {_usd(v)} is below the {_usd(minimum)} minimum.")
        return make("compliant", _usd(v), f"{label} of {_usd(v)} meets the {_usd(minimum)} minimum.")
    return fn


def _r_cert(rule, t, text, make):
    if not t.get("found"):
        return make("needs_check", "not present", "No insurance clause found - lawyer must check.")
    if t.get("certificate_holder_named"):
        return make("compliant", "Ridgeline named", "Ridgeline is named as certificate holder.")
    return make("deviation", "not named", "Ridgeline is not named as certificate holder.")


_RULE_FNS = {
    "indemnity": _r_indemnity,
    "cargo_liability": _r_cargo,
    "liability_cap_own_negligence": _r_cap,
    "payment_terms": _r_payment,
    "non_solicit": _r_nonsolicit,
    "governing_law": _r_law,
    "venue": _r_venue,
    "termination_notice": _r_termination,
    "auto_renewal": _r_renewal,
    "insurance_auto": _insurance("auto_liability_usd", MIN_AUTO_LIABILITY, "Auto liability insurance"),
    "insurance_cargo": _insurance("cargo_insurance_usd", MIN_CARGO_INSURANCE, "Cargo insurance"),
    "insurance_gl": _insurance("general_liability_usd", MIN_GENERAL_LIABILITY, "General liability insurance"),
    "certificate_holder": _r_cert,
}

# Per-term verification settings (which values must appear in the quotes).
VERIFY_SPEC: dict[str, dict] = {
    "indemnity": {},
    "cargo_liability": {"numeric_fields": ("cargo_limit_usd",)},
    "payment_terms": {"numeric_fields": ("payment_days",)},
    "non_solicit": {"numeric_fields": ("non_solicit_months",)},
    "governing_law": {},
    "termination": {"numeric_fields": ("termination_notice_days",)},
    "auto_renewal": {"numeric_fields": ("renewal_cancel_notice_days",)},
    "insurance": {"numeric_fields": ("auto_liability_usd", "cargo_insurance_usd", "general_liability_usd"),
                  "required_phrases": {"certificate_holder_named": "certificate holder"}},
    "liability_cap": {},
}
