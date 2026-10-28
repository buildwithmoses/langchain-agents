"""Number check: every number in the answer must come from the query results.

A number in the prose is supported if it equals a result value (numeric cells, or
numbers inside text cells such as dates) up to the precision it is written at
("$7,546.34", "$7,546", "7.5k", "about 7,500"), or as a percentage of a ratio cell
(0.515 -> "51.5%"). Also allowed: numbers the user typed, numbers written in the SQL
that ran (thresholds such as "at least 50 loads", visible in the SQL we show),
the number of rows returned, and calendar years. Anything else is unsupported.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NUM_RE = re.compile(
    r"(?<![\w.])(?P<neg>-)?\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s?(?:%|percent\b|[kK]\b|[mM]\b|million\b|thousand\b|billion\b|bn\b))?"
)
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)
CODE = re.compile(r"```.*?```|`[^`]*`", re.S)
MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "billion": 1e9, "bn": 1e9}
MAX_ROUNDING_REL = 0.05  # "about 7,500" for 7,546 is fine; "8,000" for 7,546 is not


@dataclass
class Mention:
    text: str
    value: float
    decimals: int
    percent: bool
    trailing_zeros: int


@dataclass
class NumberCheck:
    passed: bool
    checked: int
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checked": self.checked, "unsupported": self.unsupported}


def extract_mentions(text: str) -> list[Mention]:
    text = CODE.sub(" ", text or "")
    text = LIST_MARKER.sub(" ", text)
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        suffix = (m.group("suffix") or "").strip().lower()
        value = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        trailing = 0 if decimals else len(raw) - len(raw.rstrip("0"))
        if suffix in MULT:
            value *= MULT[suffix]
            # precision of "7.5k" is 100
            trailing = int(round(math.log10(MULT[suffix]))) - decimals
            decimals = 0
        if m.group("neg"):
            value = -value
        out.append(Mention(text=m.group(0).strip(), value=value, decimals=decimals,
                           percent=suffix in ("%", "percent"), trailing_zeros=max(0, trailing)))
    return out


def result_numbers(queries: list[dict]) -> set[float]:
    nums: set[float] = set()
    for q in queries:
        for row in q.get("rows", []):
            for v in row:
                if isinstance(v, bool):
                    nums.add(float(v))
                elif isinstance(v, (int, float)) and math.isfinite(v):
                    nums.add(float(v))
                elif isinstance(v, str):
                    for m in re.finditer(r"\d+(?:\.\d+)?", v.replace(",", "")):
                        nums.add(float(m.group()))
        nums.add(float(len(q.get("rows", []))))
    return nums


def text_numbers(*texts: str) -> set[float]:
    return {abs(m.value) for t in texts for m in extract_mentions(t or "")}


def _matches(m: Mention, r: float) -> bool:
    target = abs(m.value)
    cands = [abs(r)]
    if m.percent:
        cands.append(abs(r) * 100)
    for c in cands:
        tol = 0.5 * 10 ** (-m.decimals) + 1e-9
        if abs(target - c) <= tol:
            return True
        if m.trailing_zeros:
            tol = 0.5 * 10 ** m.trailing_zeros
            if abs(target - c) <= tol and c and abs(target - c) / abs(c) <= MAX_ROUNDING_REL:
                return True
    return False


def check_numbers(answer: str, queries: list[dict], question: str = "") -> NumberCheck:
    mentions = extract_mentions(answer)
    allowed = result_numbers(queries)
    free = text_numbers(question, *[q.get("sql", "") for q in queries])
    unsupported = []
    for m in mentions:
        v = abs(m.value)
        if not m.percent and m.decimals == 0 and 1990 <= v <= 2035:
            continue  # a calendar year
        if v in free:
            continue
        if any(_matches(m, r) for r in allowed):
            continue
        unsupported.append(m.text)
    return NumberCheck(passed=not unsupported, checked=len(mentions), unsupported=unsupported)
