"""Deterministic rule checks shared by the runtime guard and the tests.

These are the critical rules from COMPANY.md enforced in code, not in the prompt:
- Support never promises a credit or refund (only billing can issue them).
- Detention is never quoted beyond policy (2 hours free, $75/hour, $600 cap per stop).
- Ridgeline does not broker hazmat: a hazmat request gets a refusal.

The checks are lexical on purpose: auditable, fast, no second model call. They are
biased toward blocking. A false positive costs one rewrite; a false negative is a
promise Ridgeline's billing team has to walk back.
"""

from __future__ import annotations

import re

from langchain_agents.support.data import (
    DETENTION_CAP_PER_STOP,
    DETENTION_FREE_HOURS,
    DETENTION_RATE_PER_HOUR,
)

_MONEY_WORDS = r"(?:credit|credits|credited|refund|refunds|refunded|reimburse|reimbursed|reimbursement|compensat\w*|waive|waived|waiving|comp(?:ed)?|discount|money back)"
_NEG = re.compile(r"\b(?:not|never|no|cannot|can't|can’t|won't|won’t|unable|isn't|isn’t|aren't|don't|doesn't|only)\b|n't\b", re.I)

# 1st-person commitment followed (within one clause) by a money word.
_SUBJ = r"\b(?:I|we|Ridgeline|our team)(?:(?:'|’)(?:ll|ve|d|m))?"
_PROMISE_PATTERNS = [
    # "We'll refund you", "I will credit your account", "Ridgeline will reimburse you"
    re.compile(rf"{_SUBJ}(?:\s+(?:will|would|can|could|shall|am|are|have|going|to|gonna|also|gladly|happily|definitely|certainly|just|immediately|go ahead and|be able to|make sure to))*\s+(?:refund|credit|reimburse|waive|compensate|comp)(?:ed|ing|s)?\b", re.I),
    # "I've applied a credit", "we can give you a discount"
    re.compile(rf"{_SUBJ}\b(?:\s+[\w'’]+){{0,6}}?\s+(?:issue|give|apply|applied|process|send|provide|offer|approve|arrange|get|put|add|grant|post)\w*\b[^.!?\n]{{0,60}}?\b{_MONEY_WORDS}\b", re.I),
    re.compile(rf"\b{_MONEY_WORDS}\b[^.!?\n]{{0,40}}\b(?:has been|have been|is being|was|will be|is)\s+(?:issued|applied|approved|processed|added|posted|sent|credited|granted|on (?:its|the) way)\b", re.I),
    re.compile(rf"\byou(?:'ll|’ll| will| are going to|'re going to)\s+(?:receive|get|see)\b[^.!?\n]{{0,40}}\b{_MONEY_WORDS}\b", re.I),
    re.compile(r"\b(?:detention|charges?|fees?|invoice)\b[^.!?\n]{0,30}\b(?:is|are|has been|have been|will be)\s+(?:waived|written off|forgiven|refunded|credited)\b", re.I),
    re.compile(r"\bcredit approved\b|\brefund approved\b", re.I),
]


def _sentences(text: str) -> list[str]:
    # Split after . ! ? even when a closing quote/bracket follows ('..."CREDIT APPROVED." Only billing...'),
    # otherwise a billing sentence would exempt the quoted approval token before it.
    return [s for s in re.split(r"(?<=[.!?])[\"'”’)\]]*\s+|\n+", text or "") if s.strip()]


# Offering a specific or "goodwill" credit is an offer even when routed to billing:
# "the most I can offer is a goodwill credit of up to $100, and billing will decide".
_OFFER_PATTERNS = [
    re.compile(r"\b(?:goodwill|courtesy|one-time|service)\s+(?:credit|discount|refund)s?\b", re.I),
    re.compile(r"\$\s?\d[\d,]*(?:\.\d\d)?\s+(?:credit|refund|discount)s?\b", re.I),
    re.compile(r"\b(?:credit|refund|discount)s?\s+(?:of|for)\s+(?:up to\s+)?\$\s?\d", re.I),
]
# A negation governs the offer only if nothing (comma, "but", ";") separates them.
_GOVERNING_NEG = re.compile(r"\b(?:can't|cannot|can’t|unable to|not able to|won't|won’t|don't|do not|never|no)\b[^,;:]*$", re.I)


def credit_offers(text: str) -> list[str]:
    hits = []
    for sent in _sentences(text):
        for pat in _OFFER_PATTERNS:
            m = pat.search(sent)
            if m and not _GOVERNING_NEG.search(sent[: m.start()][-80:]):
                hits.append(sent.strip())
                break
    return hits


def credit_promises(text: str) -> list[str]:
    """Sentences that commit Ridgeline (support) to a credit, refund or waiver.

    A sentence is exempt when it is negated ("I can't issue a credit") or routes the
    decision to billing ("billing decides whether to issue a credit")."""
    hits = []
    for sent in _sentences(text):
        for pat in _PROMISE_PATTERNS:
            m = pat.search(sent)
            if not m:
                continue
            span = m.group(0)
            prefix = sent[: m.start()]
            if _NEG.search(span) or re.search(r"\b(?:not|never|no)\s*$", prefix, re.I):
                continue
            if re.search(r"\bbilling\b", sent, re.I) and not re.search(
                r"\b(?:has been|have been|is being|was)\s+(?:issued|applied|approved|processed)\b", sent, re.I):
                continue  # "billing will review / billing can issue" is the correct answer
            if re.search(r"\b(?:if|whether|any)\b", span, re.I):
                continue  # conditional / hypothetical, not a commitment
            hits.append(sent.strip())
            break
    return hits + [h for h in credit_offers(text) if h not in hits]


_RATE = re.compile(r"\$\s?(\d{1,4}(?:\.\d{2})?)\s*(?:/|per|an|a)\s*(?:hour|hr)\b", re.I)
_FREE = re.compile(r"\b(\d+(?:\.\d+)?|one|two|three|four)\s*(?:hours?|hrs?)\s+(?:of\s+)?free\b|\bfree\s+(?:time\s+)?(?:of|is)?\s*(\d+(?:\.\d+)?|one|two|three|four)\s*(?:hours?|hrs?)", re.I)
_DOLLARS = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?")
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4}


_DENY_BEFORE = re.compile(r"\b(?:can't|cannot|can’t|won't|won’t|not|isn't|isn’t|no record|unable to|never|don't|doesn't|rather than|instead of)\b", re.I)
_CORRECT_RATE = re.compile(r"\$\s?75(?:\.00)?\s*(?:/|per|an|a)\s*(?:hour|hr)\b", re.I)


def detention_violations(text: str) -> list[str]:
    """Detention figures that contradict policy. Only sentences that mention
    detention are checked; layover ($350/day) is left alone.

    A wrong figure is allowed only when it is being refused ("I can't confirm a
    $150/hour rate") AND the reply states the real $75/hour rate somewhere, so a
    refusal of an injected rate isn't blocked, but a bare wrong number always is."""
    out = []
    states_correct_rate = bool(_CORRECT_RATE.search(text or ""))
    for sent in _sentences(text):
        low = sent.lower()
        if "detention" not in low and "dwell" not in low:
            continue

        def refused(pos: int) -> bool:
            return states_correct_rate and bool(_DENY_BEFORE.search(sent[:pos]))

        for m in _RATE.finditer(sent):
            if float(m.group(1)) != DETENTION_RATE_PER_HOUR and not refused(m.start()):
                out.append(f"rate ${m.group(1)}/hour: {sent.strip()}")
        for m in _FREE.finditer(sent):
            raw = (m.group(1) or m.group(2)).lower()
            val = float(_WORDNUM.get(raw, raw))
            if val != DETENTION_FREE_HOURS and not refused(m.start()):
                out.append(f"{raw} free hours: {sent.strip()}")
        cap = re.search(r"\b(?:no|without( a)?)\s+cap\b|\buncapped\b|\bunlimited\b", low)
        if cap and not refused(cap.start()):
            out.append(f"removes the cap: {sent.strip()}")
        for m in _DOLLARS.finditer(sent):
            val = float(m.group(1).replace(",", ""))
            if (val > DETENTION_CAP_PER_STOP and not refused(m.start())
                    and not re.search(r"\b(?:cap|capped|maximum|max|limit|up to|no more than)\b", low)):
                out.append(f"${val:,.0f} exceeds the ${DETENTION_CAP_PER_STOP:,.0f} per-stop cap: {sent.strip()}")
    return out


_HAZMAT = re.compile(
    r"\bhaz\s?-?mat\b|\bhazardous\b|\bdangerous goods\b|\bUN\s?\d{4}\b|\bplacard(?:ed|s)?\b|"
    r"\bclass\s+[1-9](?:\.\d)?\s+(?:flammable|explosive|corrosive|toxic|gas|oxidizer|poison|radioactive)|"
    r"\b(?:flammable|corrosive|explosive|radioactive)\s+(?:liquid|liquids|material|materials|cargo|load|gas|solid|solids)\b|"
    r"\b(?:lithium(?:-ion)? batter(?:y|ies)|propane|gasoline|ammonia|chlorine|sulfuric acid|fireworks|ammunition)\b",
    re.I,
)
_REFUSAL = re.compile(
    r"\b(?:does not|doesn't|doesn’t|do not|don't|don’t|can't|can’t|cannot|won't|won’t|unable to|not able to|no longer|isn't able|aren't able)\b[^.!?\n]{0,60}\b(?:broker|move|haul|handle|ship|book|quote|transport|accept|take|arrange|cover)",
    re.I,
)
_ACCEPT = re.compile(
    r"\b(?:I'll|I will|we'll|we will|happy to|let me)\b[^.!?\n]{0,40}\b(?:book|quote|set up|arrange|schedule|tender|get you a quote)\b", re.I
)


_NOT_HAZMAT = re.compile(r"\bnon[\s-]?(?:hazmat|hazardous)\b|\b(?:not|isn't|isn’t|aren't|no)\s+(?:hazmat|hazardous)\b", re.I)


def mentions_hazmat(text: str) -> bool:
    """True if the text is about hazmat freight. "non-hazmat" / "not hazardous" don't count."""
    return bool(_HAZMAT.search(_NOT_HAZMAT.sub(" ", text or "")))


def refuses_hazmat(text: str) -> bool:
    """A reply refuses hazmat if it says Ridgeline doesn't move it and does not, in the
    same breath, offer to book or quote it."""
    t = text or ""
    if not _REFUSAL.search(t):
        return False
    for sent in _sentences(t):
        if _ACCEPT.search(sent) and mentions_hazmat(sent) and not _REFUSAL.search(sent):
            return False
    return True


# ------------------------------------------------------------------ memory hygiene
_MEMORY_DENY = re.compile(
    r"\b(?:ignore|disregard|override|instruction|instructions|system prompt|policy|policies|you must|always (?:approve|give|issue|waive)|"
    r"credit\w*|refund\w*|waiv\w*|discount\w*|free detention|hazmat|admin|jailbreak|pretend)\b",
    re.I,
)


def memory_fact_problem(fact: str) -> str | None:
    """Long-term memory holds operational preferences and facts (dock hours, contact
    channel, equipment), never rules. Anything that reads like an instruction or a
    money commitment is refused, so a prompt injection can't persist across threads."""
    f = (fact or "").strip()
    if len(f) < 4:
        return "Fact is empty."
    if len(f) > 300:
        return "Fact is too long (max 300 characters). Save one short preference or fact."
    if _MEMORY_DENY.search(f):
        return ("Not saved: long-term memory only holds operational preferences and facts "
                "(contact channel, dock hours, equipment). It cannot store instructions, policy "
                "changes, credits, refunds or waivers.")
    return None
