"""Quote and value verification, in code.

A term only counts as verified when (1) every quote the model gave appears in
the contract text after normalization, (2) every numeric value it extracted
appears as a number inside those quotes, and (3) term-specific phrases are
present (e.g. "certificate holder" when it claims Ridgeline is named as one).
Anything else is a verification failure: the extraction is retried once, then
the term is marked "not found - lawyer must check". Never compliant without a
verified quote.
"""

from __future__ import annotations

import re
import unicodedata

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "‑": "-", " ": " ", "§": " ",
})
_MARKUP = re.compile(r"[*_`>#]|\[§[^\]]*\]|\[[^\]]*\|[^\]]*\]")
_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    t = _MARKUP.sub(" ", t)
    t = _WS.sub(" ", t).strip().lower()
    return t


def _fragments(quote: str) -> list[str]:
    q = unicodedata.normalize("NFKC", quote or "").translate(_TRANSLATE)
    parts = [normalize(p) for p in _ELLIPSIS.split(q)]
    # Trim punctuation the model may add or drop at the edges of each fragment.
    parts = [p.strip(" .,;:\"'()") for p in parts]
    return [p for p in parts if p]


def find_span(quote: str, text: str) -> tuple[int, int] | None:
    """Span in normalize(text) where the quote's fragments occur in order, else None.
    Fragments shorter than 12 characters are too weak to verify on their own."""
    frags = _fragments(quote)
    if not frags or sum(len(f) for f in frags) < 12:
        return None
    hay = normalize(text)
    pos, first = 0, None
    for f in frags:
        i = hay.find(f, pos)
        if i < 0:
            return None
        first = i if first is None else first
        pos = i + len(f)
    return (first or 0, pos)


def quote_in_text(quote: str, text: str) -> bool:
    return find_span(quote, text) is not None


# ---- numbers inside quotes ---------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUM = re.compile(r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(million|thousand|k|m)?\b", re.I)
_WORD = re.compile(r"[a-z]+")


def _word_numbers(text: str) -> set[int]:
    """'one hundred twenty-five thousand' -> 125000, 'forty-five' -> 45."""
    found: set[int] = set()
    words = _WORD.findall(text.lower().replace("-", " "))
    total = current = 0
    active = False
    for w in words + ["<end>"]:
        if w in _UNITS:
            current += _UNITS[w]; active = True
        elif w in _TENS:
            current += _TENS[w]; active = True
        elif w == "hundred" and active:
            current *= 100
        elif w in ("thousand", "million") and active:
            total += current * _SCALES[w]; current = 0
        elif w == "and" and active:
            continue
        else:
            if active:
                found.add(total + current)
            total = current = 0
            active = False
    return found


def numbers_in(text: str) -> set[int]:
    nums: set[int] = set()
    for m in _NUM.finditer(text or ""):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        scale = (m.group(2) or "").lower()
        if scale in ("million", "m"):
            value *= 1_000_000
        elif scale in ("thousand", "k"):
            value *= 1_000
        if value == int(value):
            nums.add(int(value))
    return nums | _word_numbers(text or "")


def verify_term(term: dict, contract_text: str, *, numeric_fields: tuple[str, ...] = (),
                text_fields: tuple[str, ...] = (), required_phrases: dict[str, str] | None = None) -> list[str]:
    """Problems with one extracted term (empty list = verified)."""
    problems: list[str] = []
    if not term.get("found"):
        return problems
    quotes = [q for q in term.get("quotes") or [] if (q or "").strip()]
    if not quotes:
        return ["found=true but no quote given"]
    for q in quotes:
        if not quote_in_text(q, contract_text):
            problems.append(f"quote not found verbatim in the contract: {q[:120]!r}")
    joined = " ".join(quotes)
    nums = numbers_in(joined)
    for field in numeric_fields:
        value = term.get(field)
        if value is not None and int(value) not in nums:
            problems.append(f"{field}={value} does not appear in the quoted text")
    norm_joined = normalize(joined)
    for field in text_fields:
        value = term.get(field)
        if value and normalize(str(value)) not in norm_joined:
            problems.append(f"{field}={value!r} does not appear in the quoted text")
    for field, phrase in (required_phrases or {}).items():
        if term.get(field) and phrase not in norm_joined:
            problems.append(f"{field}=true but the quotes do not contain {phrase!r}")
    return problems
