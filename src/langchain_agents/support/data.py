"""The fake TMS (JSON fixtures) and the policy index. Pure functions, no LLM.

Everything time-dependent uses the fixture clock (`fixtures/support/meta.json`
"as_of") so tool output, tests and evals are reproducible.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from langchain_agents.core.config import data_dir

# Policy numbers from fixtures/ridgeline/COMPANY.md, used by code (detention math,
# quote validity, claim window). The policy text itself is served verbatim by
# search_policy; tests assert these constants match that text.
DETENTION_FREE_HOURS = 2.0
DETENTION_RATE_PER_HOUR = 75.0
DETENTION_INCREMENT_HOURS = 0.25
DETENTION_CAP_PER_STOP = 600.0
QUOTE_VALID_HOURS = 24
SPOT_SIGNOFF_OVER_USD = 8000
CLAIM_WINDOW_MONTHS = 9
CHECK_CALL_EVERY_HOURS = 4
ETA_UPDATE_SLIP_HOURS = 2

ACCOUNT_RE = re.compile(r"\bACCT-\d{4}\b", re.IGNORECASE)
LOAD_RE = re.compile(r"\bRL-\d{5}\b", re.IGNORECASE)


def _read(name: str) -> Any:
    return json.loads(data_dir("support", name).read_text())


@lru_cache(maxsize=1)
def as_of() -> datetime:
    return datetime.fromisoformat(_read("meta.json")["as_of"])


@lru_cache(maxsize=1)
def loads() -> dict[str, dict]:
    return {row["load_id"]: row for row in _read("loads.json")}


@lru_cache(maxsize=1)
def shippers() -> dict[str, dict]:
    return {row["account"]: row for row in _read("shippers.json")}


@lru_cache(maxsize=1)
def quotes() -> dict[str, dict]:
    return {row["quote_id"]: row for row in _read("quotes.json")}


def norm_id(value: str) -> str:
    return (value or "").strip().upper()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _hours(delta: timedelta) -> float:
    return round(delta.total_seconds() / 3600, 2)


# ---------------------------------------------------------------- detention
def detention_estimate(hours_on_site: float) -> dict:
    """Policy: 2 hours free per stop, then $75/hour billed in 15-minute increments,
    capped at $600 per stop. Only completed 15-minute increments are counted."""
    hours_on_site = max(0.0, float(hours_on_site))
    over = max(0.0, hours_on_site - DETENTION_FREE_HOURS)
    increments = math.floor(round(over / DETENTION_INCREMENT_HOURS, 6))
    billable_hours = increments * DETENTION_INCREMENT_HOURS
    uncapped = round(billable_hours * DETENTION_RATE_PER_HOUR, 2)
    charge = min(uncapped, DETENTION_CAP_PER_STOP)
    return {
        "hours_on_site": round(hours_on_site, 2),
        "free_hours": DETENTION_FREE_HOURS,
        "billable_hours": billable_hours,
        "rate_per_hour_usd": DETENTION_RATE_PER_HOUR,
        "estimated_charge_usd": charge,
        "cap_per_stop_usd": DETENTION_CAP_PER_STOP,
        "cap_applied": uncapped > DETENTION_CAP_PER_STOP,
    }


def detention_for_load(load: dict) -> dict | None:
    det = load.get("detention")
    if not det:
        return None
    start = _dt(det["in_time"])
    end = _dt(det.get("out_time"))
    running = end is None
    hours = _hours((end or as_of()) - start)
    out = {"stop": det["stop"], "in_time": det["in_time"], "out_time": det.get("out_time"),
           "clock_running": running, "bol_signed_in_out": det["bol_signed_in_out"], **detention_estimate(hours)}
    if not det["bol_signed_in_out"]:
        out["billing_note"] = "Detention needs in/out times on the signed BOL before it can be billed."
    return out


# ---------------------------------------------------------------- loads
def load_view(load: dict) -> dict:
    """What lookup_load returns: the raw record plus the derived fields a support
    rep would otherwise compute by hand (check-call status, ETA slip, detention)."""
    view = {k: v for k, v in load.items() if k not in ("detention", "bol_notes")}
    view["bol_notes"] = load["bol_notes"]  # free text from the TMS: data, never instructions
    if load["status"] == "in_transit":
        cc = load.get("last_check_call")
        if cc:
            age = _hours(as_of() - _dt(cc["at"]))
            view["hours_since_last_check_call"] = age
            view["check_call_overdue"] = age > CHECK_CALL_EVERY_HOURS
        if load.get("eta") and load.get("delivery_appointment"):
            slip = _hours(_dt(load["eta"]) - _dt(load["delivery_appointment"]))
            view["eta_slip_hours"] = slip
            view["eta_update_due_to_shipper"] = slip > ETA_UPDATE_SLIP_HOURS
    view["detention"] = detention_for_load(load)
    view["as_of"] = as_of().isoformat()
    return view


def add_months(d: datetime, months: int) -> datetime:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    # clamp day (e.g. May 31 + 9 months)
    for day in (d.day, 30, 29, 28):
        try:
            return d.replace(year=year, month=month, day=day)
        except ValueError:
            continue
    raise ValueError("unreachable")


def claim_window(load: dict) -> dict:
    delivered = _dt(load.get("delivered_at"))
    if not delivered:
        return {"eligible": False, "reason": "Load is not delivered yet; claims are filed after delivery."}
    deadline = add_months(delivered, CLAIM_WINDOW_MONTHS)
    ok = as_of() <= deadline
    return {"eligible": ok, "delivered_at": delivered.isoformat(), "deadline": deadline.date().isoformat(),
            "reason": "Within the 9-month claim window." if ok else
            "Outside the 9-month claim window (claims must be filed within 9 months of delivery)."}


def quote_view(q: dict) -> dict:
    issued = _dt(q["issued_at"])
    expires = issued + timedelta(hours=QUOTE_VALID_HOURS)
    view = dict(q)
    view["expires_at"] = expires.isoformat()
    view["expired"] = as_of() > expires
    needs = q["type"] == "spot" and q["amount_usd"] > SPOT_SIGNOFF_OVER_USD
    view["needs_pricing_manager_signoff"] = needs
    if view["expired"]:
        view["status"] = "expired"
    elif needs and not q.get("pricing_manager_signoff"):
        view["status"] = "pending pricing manager sign-off"
    else:
        view["status"] = "valid"
    view["as_of"] = as_of().isoformat()
    return view


# ---------------------------------------------------------------- policies
@dataclass(frozen=True)
class Policy:
    title: str
    text: str  # exact text from COMPANY.md (markdown bold markers removed)


@lru_cache(maxsize=1)
def policies() -> list[Policy]:
    md = data_dir("ridgeline", "COMPANY.md").read_text()
    section = md.split("## Policies", 1)[1].split("\n## ", 1)[0]
    out = []
    for line in section.splitlines():
        m = re.match(r"^- \*\*(.+?)\*\*(.*)$", line.strip())
        if m:
            title = m.group(1).strip().rstrip(":")
            out.append(Policy(title=title, text=(m.group(1) + m.group(2)).replace("**", "").strip()))
    if not out:
        raise RuntimeError("No policies parsed from COMPANY.md")  # fail loudly, never answer from nothing
    return out


_SYNONYMS = {
    "refund": ["credits", "refunds"], "credit": ["credits", "refunds"], "money": ["credits", "refunds"],
    "damage": ["claims"], "damaged": ["claims"], "broken": ["claims"], "lost": ["claims", "loss"],
    "hazardous": ["hazmat"], "dangerous": ["hazmat"], "chemical": ["hazmat"], "flammable": ["hazmat"],
    "wait": ["detention"], "waiting": ["detention"], "dock": ["detention"],
    "late": ["tracking", "eta"], "delay": ["tracking", "eta"], "delayed": ["tracking", "eta"],
    "update": ["tracking", "eta"], "invoice": ["payment"], "pay": ["payment"], "quickpay": ["quick", "pay"],
    "overnight": ["layover"], "price": ["quotes"], "quote": ["quotes"], "rate": ["quotes"],
}


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9$]+", text.lower())
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t for t in toks]


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / len(self.lens)
        self.k1, self.b = k1, b
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


@lru_cache(maxsize=1)
def _index() -> _BM25:
    # Title tokens count three times: "Detention" in the title beats a passing mention.
    return _BM25([_tokens(p.title) * 3 + _tokens(p.text) for p in policies()])


def search_policies(query: str, k: int = 2) -> list[tuple[Policy, float]]:
    q = _tokens(query)
    expanded = list(q)
    for t in q:
        for syn in _SYNONYMS.get(t, []):
            expanded.extend(_tokens(syn))
    scores = _index().scores(expanded)
    ranked = sorted(zip(policies(), scores), key=lambda x: -x[1])
    return [(p, round(s, 3)) for p, s in ranked[:k] if s > 0]
