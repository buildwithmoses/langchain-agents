"""Deterministic sectioning. No model involved.

Splits a contract into top-level sections (markdown headings: "## 6. Cargo
Liability", "## Article 4. ...", "## Exhibit B - ...") and numbered subsections
("6.2", "A.4"). Sections are used to label the prompt and to locate every
verified quote, so the section reference in a finding comes from code, not
from the model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .verify import find_span, normalize

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
ARTICLE_NUM = re.compile(r"^(?:article|section)?\s*(\d+)\b[.:]?\s*(.*)$", re.I)
EXHIBIT = re.compile(r"^(exhibit|schedule|appendix|annex)\s+([A-Z0-9]+)\b", re.I)
SUBSECTION = re.compile(r"^\s*(?:\*\*)?([A-Z]?\d{0,3}\.\d+(?:\.\d+)*)\.?(?:\*\*)?\s+\S")


@dataclass
class Section:
    ref: str          # "6.2", "6", "Exhibit A", "A.4", "Preamble"
    title: str        # heading text of the enclosing top-level section
    level: int        # 1 = top-level section, 2 = numbered subsection
    parent: str       # ref of the enclosing top-level section ("" for top level)
    text: str
    start: int        # character offsets into the original contract text
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


def _top_ref(heading: str) -> str:
    h = heading.strip().strip("*").strip()
    m = EXHIBIT.match(h)
    if m:
        return f"{m.group(1).title()} {m.group(2).upper()}"
    m = ARTICLE_NUM.match(h)
    if m:
        return m.group(1)
    return h[:60]


def split_sections(text: str) -> list[Section]:
    """Top-level sections and numbered subsections, in document order.

    Subsection text runs until the next subsection or heading. Top-level section
    text covers the heading and all its subsections.
    """
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    total = pos

    tops: list[Section] = []
    subs: list[Section] = []
    cur_top = Section("Preamble", "Preamble", 1, "", "", 0, 0)
    cur_sub: Section | None = None

    def close_sub(at: int) -> None:
        nonlocal cur_sub
        if cur_sub is not None:
            cur_sub.end = at
            cur_sub.text = text[cur_sub.start:at].strip()
            subs.append(cur_sub)
            cur_sub = None

    def close_top(at: int) -> None:
        cur_top.end = at
        cur_top.text = text[cur_top.start:at].strip()
        if cur_top.text:
            tops.append(cur_top)

    for line, start in zip(lines, offsets):
        hm = HEADING.match(line)
        if hm:
            close_sub(start)
            close_top(start)
            title = hm.group(2).strip()
            # The document title (a single '#') stays in the preamble.
            if len(hm.group(1)) == 1 and not tops and cur_top.ref == "Preamble":
                cur_top = Section("Preamble", title, 1, "", "", 0, 0)
                continue
            cur_top = Section(_top_ref(title), title, 1, "", "", start, start)
            continue
        sm = SUBSECTION.match(line)
        if sm and cur_top.ref != "Preamble":
            close_sub(start)
            cur_sub = Section(sm.group(1), cur_top.title, 2, cur_top.ref, "", start, start)
    close_sub(total)
    close_top(total)

    ordered = sorted(tops + subs, key=lambda s: (s.start, s.level))
    return _dedupe_refs(ordered)


def _dedupe_refs(sections: list[Section]) -> list[Section]:
    seen: dict[str, int] = {}
    for s in sections:
        n = seen.get(s.ref, 0)
        seen[s.ref] = n + 1
        if n:
            s.ref = f"{s.ref} ({n + 1})"
    return sections


def render_for_prompt(sections: list[Section]) -> str:
    """The contract as the model sees it: each top-level section under a clear
    label, so definitions and exhibits are visibly part of the agreement."""
    out = []
    for s in sections:
        if s.level == 1:
            label = s.ref if s.ref == s.title else f"{s.ref} | {s.title}"
            out.append(f"=== [{label}] ===\n{s.text}")
    return "\n\n".join(out)


def locate(quote: str, sections: list[Section], full_text: str) -> str | None:
    """Most specific section ref containing the quote, or None if it isn't in the text."""
    if find_span(quote, full_text) is None:
        return None
    # Most specific first: numbered subsections, then top-level sections.
    for level in (2, 1):
        for s in sections:
            if s.level == level and find_span(quote, s.text) is not None:
                return s.ref if level == 1 or not s.parent else _fmt_sub(s)
    return None


def _fmt_sub(s: Section) -> str:
    if s.parent.lower().startswith(("exhibit", "schedule", "appendix", "annex")):
        return f"{s.parent}, {s.ref}"
    return s.ref


__all__ = ["Section", "split_sections", "render_for_prompt", "locate", "normalize"]
