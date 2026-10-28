"""The citation verifier. Deterministic, no model calls. This is the gate.

A claim survives only if ALL of these hold:
1. its source_id is a source this run actually fetched/read (state["sources"]);
   a URL the model saw in a search snippet but never fetched does not count;
2. its quote is long enough to mean something (>= MIN_WORDS words);
3. the quote appears in that source's text after normalization: Unicode NFKC,
   curly quotes -> straight, all dashes -> '-', markdown emphasis removed,
   whitespace collapsed, case folded. An ellipsis ("..." or "[...]") splits the
   quote into segments that must each appear, in order;
4. every number in the claim appears in the quote (a 4-digit year may instead
   appear anywhere in the source). Catches "real quote, wrong number".

No fuzzy matching is accepted. A 0.95-similar quote can differ by exactly the
digit that matters ("$2.87" vs "$2.78"), so near-misses are rejected, and the
closest passage is reported back to the agent so it can copy the exact text on
its one revision. Fail closed: anything that can't be checked is dropped.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import asdict, dataclass

from .fetch import normalize_url

MIN_WORDS = 5
MIN_SEGMENT_WORDS = 3

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "­": "", "​": "",
})
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…)\s*")
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)?")


def _base(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    return re.sub(r"(\*\*|__|\*|~~)", "", text)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _base(text)).strip().casefold()


def _clean_quote(quote: str) -> str:
    q = normalize(quote)
    q = q.strip(" \"'")
    return q.rstrip(".,;:").strip()


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.findall(_base(text)):
        n = m.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def quote_in_text(quote: str, text: str) -> bool:
    """Exact (normalized) containment; ellipsis segments must appear in order."""
    hay = normalize(text)
    segments = [s.strip(" \"'").rstrip(".,;:").strip() for s in _ELLIPSIS.split(_clean_quote(quote))]
    segments = [s for s in segments if s]
    if not segments:
        return False
    pos = 0
    for seg in segments:
        i = hay.find(seg, pos)
        if i < 0:
            return False
        pos = i + len(seg)
    return True


def closest_passage(quote: str, text: str) -> tuple[float, str]:
    """Best-matching window of the source (original casing) for feedback. Not
    used to accept anything."""
    words = re.sub(r"\s+", " ", _base(text)).split(" ")
    q = _clean_quote(quote)
    qwords = q.split(" ")
    n = len(qwords)
    if not words or not q:
        return 0.0, ""
    anchors = set(qwords[:3]) | set(qwords[-3:])
    best = (0.0, "")
    for i, w in enumerate(words):
        if w.casefold().strip(".,;:\"'()") not in anchors and i % 25:
            continue
        for start in (i, max(0, i - n + 1)):
            window = " ".join(words[start:start + n])
            r = difflib.SequenceMatcher(None, q, window.casefold(), autojunk=False).ratio()
            if r > best[0]:
                best = (r, window)
    return round(best[0], 3), best[1]


@dataclass
class ClaimCheck:
    index: int
    claim: str
    source_id: str
    quote: str
    verified: bool
    reason: str  # ok | source_not_fetched | quote_too_short | quote_not_found | number_not_in_quote
    detail: str = ""
    closest_ratio: float | None = None
    closest_passage: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_source(source_id: str, sources: dict[str, dict]) -> dict | None:
    sid = (source_id or "").strip()
    if sid in sources:
        return sources[sid]
    if sid.startswith(("http://", "https://")):
        want = normalize_url(sid)
        for key, src in sources.items():
            if normalize_url(key) == want or normalize_url(src.get("final_url") or "") == want:
                return src
    return None


def check_claim(index: int, claim: dict, sources: dict[str, dict]) -> ClaimCheck:
    c = ClaimCheck(index, claim.get("claim", ""), claim.get("source_id", ""), claim.get("quote", ""), False, "")
    src = resolve_source(c.source_id, sources)
    if src is None or not src.get("text"):
        c.reason, c.detail = "source_not_fetched", "this source was not fetched/read in this run"
        return c
    text = src["text"]
    if len(_clean_quote(c.quote).split()) < MIN_WORDS or any(
        len(s.split()) < MIN_SEGMENT_WORDS for s in _ELLIPSIS.split(_clean_quote(c.quote)) if s.strip()
    ):
        c.reason, c.detail = "quote_too_short", f"quotes need at least {MIN_WORDS} words ({MIN_SEGMENT_WORDS} per ellipsis segment)"
        return c
    if not quote_in_text(c.quote, text):
        c.reason, c.detail = "quote_not_found", "the quote does not appear in the source text"
        c.closest_ratio, c.closest_passage = closest_passage(c.quote, text)
        return c
    missing = _numbers(c.claim) - _numbers(c.quote)
    src_numbers = _numbers(text)
    missing = {n for n in missing if not (re.fullmatch(r"(19|20)\d\d", n) and n in src_numbers)}
    if missing:
        c.reason, c.detail = "number_not_in_quote", f"numbers in the claim but not in the quote: {sorted(missing)}"
        return c
    c.verified, c.reason = True, "ok"
    return c


def verify_claims(claims: list[dict], sources: dict[str, dict]) -> list[ClaimCheck]:
    return [check_claim(i, c, sources) for i, c in enumerate(claims, start=1)]


_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def filter_answer(answer: str, verified: set[int], renumber: dict[int, int]) -> tuple[str, int]:
    """Keep only answer sentences whose every citation [n] is a verified claim;
    uncited sentences are dropped too (the summary may not say anything the
    findings don't back). Returns (answer, sentences_dropped)."""
    kept, dropped = [], 0
    for sent in _SENT.split((answer or "").strip()):
        refs = [int(x) for x in _CITE.findall(sent)]
        if refs and all(r in verified for r in refs):
            kept.append(_CITE.sub(lambda m: f"[{renumber[int(m.group(1))]}]", sent))
        elif sent.strip():
            dropped += 1
    return " ".join(kept), dropped


NO_ANSWER = ("We could not verify an answer to this question from the sources we read. "
             "See what we couldn't verify below.")


def finalize(brief: dict | None, checks: list[ClaimCheck]) -> dict:
    """The customer-facing brief: verified claims only, renumbered, answer
    sentences filtered to those backed by verified claims."""
    if not brief:
        return {"answerable": False, "answer": NO_ANSWER, "claims": [], "dropped": [], "gaps": [
            "The research run did not produce a brief (failed closed)."], "answer_sentences_dropped": 0}
    ok = [c for c in checks if c.verified]
    renumber = {c.index: n for n, c in enumerate(ok, start=1)}
    answer, sent_dropped = filter_answer(brief.get("answer", ""), set(renumber), renumber)
    answerable = bool(brief.get("answerable")) and bool(ok)
    if not answerable:
        answer = NO_ANSWER if not ok else (
            "We could not answer this question from the sources we read. The verified findings below are "
            "related context only, not an answer.")
    elif not answer:
        answer = "The summary didn't survive verification; the verified findings below are what the sources support."
    return {
        "answerable": answerable,
        "answer": answer,
        "claims": [{"n": renumber[c.index], "claim": c.claim, "source_id": c.source_id, "quote": c.quote} for c in ok],
        "dropped": [c.to_dict() for c in checks if not c.verified],
        "gaps": list(brief.get("gaps") or []),
        "answer_sentences_dropped": sent_dropped,
    }
