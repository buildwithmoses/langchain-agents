"""Markdown for the memo and the final redlines. Pure functions."""

from __future__ import annotations

BANNER = ("> DRAFT FOR LAWYER REVIEW. Prepared by the contract review agent for Marcus Hale, General Counsel. "
          "Nothing has been sent to the carrier. Fictional demo data.")
STATUS_LABEL = {"deviation": "DEVIATION", "needs_check": "LAWYER MUST CHECK", "compliant": "ok"}
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def _q(quotes: list[str]) -> str:
    return "\n".join(f"> {q.strip()}" for q in quotes) if quotes else "> (no quote)"


def memo_markdown(*, carrier: str, name: str, summary: str, findings: list[dict], redlines: list[dict]) -> str:
    devs = sorted([f for f in findings if f["status"] == "deviation"], key=lambda f: SEV_ORDER.get(f["severity"], 9))
    checks = [f for f in findings if f["status"] == "needs_check"]
    by_rule = {r["rule_id"]: r for r in redlines}
    lines = [f"# Contract review memo: {carrier}", "", f"Agreement: `{name}`", "", BANNER, "", "## Summary", "", summary.strip(), "",
             f"**{len(devs)} deviation(s)** ({sum(f['severity'] == 'high' for f in devs)} high severity), "
             f"**{len(checks)} item(s) the lawyer must check**, "
             f"{sum(f['status'] == 'compliant' for f in findings)} rule(s) compliant.", "",
             "## Playbook checklist", "", "| Rule | Severity | Status | Contract says | Section |", "|---|---|---|---|---|"]
    for f in sorted(findings, key=lambda f: (f["status"] == "compliant", SEV_ORDER.get(f["severity"], 9))):
        lines.append(f"| {f['title']} | {f['severity']} | {STATUS_LABEL[f['status']]} | {f['found_value']} | "
                     f"{', '.join(f['section_refs']) or '-'} |")
    if devs:
        lines += ["", "## Deviations and suggested redlines", ""]
        for i, f in enumerate(devs, 1):
            r = by_rule.get(f["rule_id"], {})
            lines += [f"### {i}. {f['title']} ({f['severity']}) - §{', '.join(f['section_refs']) or '?'}", "",
                      f"**Playbook:** {f['playbook']}", "", f"**Finding (rules engine):** {f['reason']}", "",
                      "**Contract text:**", "", _q(f["quotes"]), "",
                      f"**Plain English:** {r.get('explanation', '').strip()}", "",
                      f"**Suggested redline** (id `{r.get('id', '?')}`):", "", f"> {r.get('suggested_text', '').strip()}", ""]
    if checks:
        lines += ["", "## Not verified - lawyer must check", ""]
        for f in checks:
            lines.append(f"- **{f['title']}**: {f['reason']}")
    return "\n".join(lines).rstrip() + "\n"


def redlines_markdown(*, carrier: str, name: str, redlines: list[dict], decisions: dict[str, dict]) -> str:
    approved = [r for r in redlines if decisions.get(r["id"], {}).get("action") in ("approve", "edit")]
    rejected = [r for r in redlines if r not in approved]
    lines = [f"# Approved redlines: {carrier}", "", f"Agreement: `{name}`", "",
             "> Approved by the General Counsel. Not sent to the carrier by this system.", ""]
    if not approved:
        lines.append("No redlines were approved.")
    for i, r in enumerate(approved, 1):
        d = decisions[r["id"]]
        text = d.get("text") if d["action"] == "edit" else r["suggested_text"]
        lines += [f"## {i}. {r['title']} - §{', '.join(r['section_refs']) or '?'}"
                  f"{' (edited by lawyer)' if d['action'] == 'edit' else ''}", "",
                  "**Current text:**", "", _q(r["quotes"]), "", "**Replace with:**", "", f"> {text.strip()}", ""]
        if d.get("note"):
            lines += [f"_Lawyer note: {d['note']}_", ""]
    lines += ["", f"_{len(approved)} approved, {len(rejected)} rejected or undecided (not included)._"]
    return "\n".join(lines).rstrip() + "\n"
