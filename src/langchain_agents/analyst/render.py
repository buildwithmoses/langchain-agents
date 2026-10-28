"""Markdown output: answer, the SQL that ran, the result table."""

from __future__ import annotations

TABLE_ROWS = 25


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1000 else f"{v:g}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def markdown_table(columns: list[str], rows: list, max_rows: int = TABLE_ROWS) -> str:
    if not columns:
        return "_(no columns)_"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows[:max_rows]:
        lines.append("| " + " | ".join(_cell(v) for v in r) + " |")
    if not rows:
        lines.append("| " + " | ".join("" for _ in columns) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_{len(rows) - max_rows} more rows not shown ({len(rows)} total)._")
    return "\n".join(lines)


def render_final(final: dict) -> str:
    parts = [final["answer"]]
    for i, q in enumerate(final.get("queries", []), 1):
        label = f"SQL {i}" if len(final["queries"]) > 1 else "SQL"
        tags = []
        if q.get("approved"):
            tags.append("approved by reviewer")
        if q.get("edited_by_reviewer"):
            tags.append("edited by reviewer")
        if q.get("limit_applied"):
            tags.append(f"LIMIT {q['limit_applied']} enforced")
        tag = f" ({', '.join(tags)})" if tags else ""
        parts.append(f"**{label}**{tag}\n```sql\n{q['sql']}\n```")
        trunc = " (truncated by the row limit)" if q.get("truncated") else ""
        parts.append(f"**Result** - {len(q['rows'])} row(s){trunc}\n\n" + markdown_table(q["columns"], q["rows"]))
    for b in final.get("blocked", []):
        parts.append(f"_Blocked query (not run): {' '.join(b.get('reasons', []))}_")
    return "\n\n".join(parts)
