"""The SQL guard: code, not the prompt, decides what reaches the database.

review(sql, db) -> GuardDecision with status:
  "blocked"         never runs (writes/DDL/ATTACH/PRAGMA, stacked statements, raw PII
                    in the output, unparseable or unknown tables/columns)
  "needs_approval"  runs only after a human approves (PII used in COUNT/filters/joins,
                    or a full scan of a table above SCAN_ROW_THRESHOLD rows)
  "ok"              runs now

Three independent checks have to agree a query is a read: a token denylist, the
sqlglot parse tree, and (at execution) the SQLite authorizer in db.py. Anything the
guard cannot understand is blocked: fail closed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import TokenType

from .db import MAX_ROWS, SCAN_ROW_THRESHOLD, QueryError, ReadOnlyDB

logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_CHARS = 6000
# Statement keywords that never belong in a read. REPLACE is special-cased: replace()
# is also a string function.
DENY_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
    "REINDEX", "ANALYZE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "TRUNCATE", "GRANT",
    "REVOKE", "UPSERT", "MERGE", "INTO", "EXPLAIN", "LOAD_EXTENSION", "TRANSACTION",
}
WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Pragma,
    exp.Attach, exp.Detach, exp.Transaction, exp.Commit, exp.Rollback, exp.Merge, exp.TruncateTable, exp.Into,
)
READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
SAFE_PII_AGGREGATES = (exp.Count,)  # COUNT reveals no value; MIN/MAX/GROUP_CONCAT do


@dataclass
class GuardDecision:
    status: str  # "ok" | "needs_approval" | "blocked"
    sql: str  # what was submitted
    sql_to_run: str = ""  # what will run (LIMIT enforced)
    reasons: list[str] = field(default_factory=list)  # why blocked / why approval is needed
    pii_touched: set[tuple[str, str]] = field(default_factory=set)  # aggregate/filter use only
    limit_applied: int | None = None
    alias_map: dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"

    def to_dict(self) -> dict:
        return {"status": self.status, "sql": self.sql, "sql_to_run": self.sql_to_run, "reasons": self.reasons,
                "pii_touched": sorted(f"{t}.{c}" for t, c in self.pii_touched), "limit_applied": self.limit_applied}


def _block(sql: str, *reasons: str) -> GuardDecision:
    return GuardDecision(status="blocked", sql=sql, reasons=list(reasons))


def _strip_trailing(sql: str, tokens) -> str:
    """Cut at the end of the last real token: drops trailing ';' and trailing comments
    (an unterminated /* comment at the end would otherwise swallow an appended LIMIT)."""
    real = [t for t in tokens if t.token_type != TokenType.SEMICOLON]
    return sql[: real[-1].end + 1] if real else ""


def static_check(sql: str, db: ReadOnlyDB, max_rows: int = MAX_ROWS) -> GuardDecision:
    """Everything that can be decided without touching the database."""
    if not isinstance(sql, str) or not sql.strip():
        return _block(sql or "", "Empty query.")
    if len(sql) > MAX_SQL_CHARS:
        return _block(sql, f"Query longer than {MAX_SQL_CHARS} characters.")
    if "\x00" in sql:
        return _block(sql, "Query contains a NUL byte.")

    # 1. Tokens: stacked statements and write keywords, wherever they hide.
    try:
        tokens = sqlglot.tokenize(sql, read="sqlite")
    except Exception as e:  # noqa: BLE001 - fail closed on anything unexpected
        return _block(sql, f"Could not tokenize the query: {e}")
    semis = [i for i, t in enumerate(tokens) if t.token_type == TokenType.SEMICOLON]
    if any(any(t.token_type != TokenType.SEMICOLON for t in tokens[i + 1:]) for i in semis):
        return _block(sql, "Only one statement is allowed (found ';' followed by more SQL).")
    words = [t for t in tokens if t.token_type not in (TokenType.STRING, TokenType.IDENTIFIER, TokenType.NATIONAL_STRING)]
    for i, t in enumerate(words):
        up = t.text.upper()
        if up in DENY_KEYWORDS:
            return _block(sql, f"Read-only: '{up}' is not allowed. Only a single SELECT (or WITH ... SELECT) runs.")
        if up == "REPLACE" and (i == 0 or (i + 1 < len(words) and words[i + 1].text.upper() == "INTO")):
            return _block(sql, "Read-only: 'REPLACE' statements are not allowed.")
    if not words or words[0].text.upper() not in ("SELECT", "WITH"):
        return _block(sql, "Only SELECT or WITH ... SELECT queries are allowed.")

    # 2. Parse tree: exactly one read statement, no write node anywhere (CTE-hidden writes).
    try:
        parsed = [p for p in sqlglot.parse(sql, read="sqlite") if p is not None and not isinstance(p, exp.Semicolon)]
    except Exception as e:  # noqa: BLE001
        return _block(sql, f"Could not parse the query (SQLite dialect): {str(e).splitlines()[0][:200]}")
    if len(parsed) != 1:
        return _block(sql, "Only one statement is allowed.")
    root = parsed[0]
    if not isinstance(root, READ_ROOTS):
        return _block(sql, f"Only SELECT queries are allowed (got {type(root).__name__}).")
    for node in root.walk():
        if isinstance(node, WRITE_NODES):
            return _block(sql, f"Read-only: found a {type(node).__name__} inside the query.")
        if isinstance(node, exp.Anonymous) and str(node.name).lower() in ("load_extension", "readfile", "writefile"):
            return _block(sql, f"Function {node.name}() is not allowed.")

    alias_map = {}
    known = db.schema()
    for t in root.find_all(exp.Table):
        name = t.name.lower()
        if t.args.get("db") or t.args.get("catalog"):
            return _block(sql, "Schema-qualified tables (other databases) are not allowed.")
        alias_map[(t.alias_or_name or name).lower()] = name
        alias_map[name] = name

    # 3. PII policy, on the fully qualified tree (resolves aliases, expands SELECT *).
    try:
        qualified = qualify(root.copy(), schema=db.sqlglot_schema(), dialect="sqlite",
                            validate_qualify_columns=True, quote_identifiers=False)
    except Exception as e:  # noqa: BLE001 - unknown column/table, ambiguous names: fail closed
        msg = str(e).splitlines()[0][:300]
        return _block(sql, f"Could not resolve the query against the schema: {msg}")
    if any(isinstance(s, exp.Star) for s in qualified.find_all(exp.Star)
           if not isinstance(s.parent, exp.Count)):
        return _block(sql, "Could not expand SELECT * against the schema.")

    raw, touched = set(), set()
    cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
    for scope in traverse_scope(qualified):
        for col in scope.columns:
            src = scope.sources.get(col.table)
            if not isinstance(src, exp.Table):
                continue  # derived table/CTE: its own projection was checked in its scope
            table = src.name.lower()
            if table in cte_names or table not in known:
                continue
            if not db.is_pii(table, col.name):
                continue
            key = (table, col.name.lower())
            in_agg = in_projection = False
            node = col
            while node is not None and node is not scope.expression:
                if isinstance(node, SAFE_PII_AGGREGATES):
                    in_agg = True
                if node.parent is scope.expression and node.arg_key == "expressions":
                    in_projection = True
                node = node.parent
            if in_projection and not in_agg:
                raw.add(key)
            else:
                touched.add(key)
    if raw:
        cols = ", ".join(sorted(f"{t}.{c}" for t, c in raw))
        return _block(sql, f"PII policy: {cols} cannot be selected (contact/driver identity). "
                           "Aggregates like COUNT(...) are allowed with approval; use IDs or carrier/shipper names instead.")

    # 4. LIMIT: append one if missing, wrap if it is too large or not a literal.
    body = _strip_trailing(sql, tokens).strip()
    limit = root.args.get("limit")
    limit_applied = None
    if limit is None:
        sql_to_run = f"{body}\nLIMIT {max_rows}"
        limit_applied = max_rows
    else:
        val = limit.expression if isinstance(limit, exp.Limit) else None
        try:
            n = int(val.name) if isinstance(val, exp.Literal) and val.is_int else None
        except (TypeError, ValueError):
            n = None
        if n is not None and 0 <= n <= max_rows:
            sql_to_run = body
        else:
            sql_to_run = f"SELECT * FROM (\n{body}\n) LIMIT {max_rows}"
            limit_applied = max_rows

    reasons = []
    if touched:
        reasons.append("Touches PII columns (aggregate/filter use only): "
                       + ", ".join(sorted(f"{t}.{c}" for t, c in touched)))
    return GuardDecision(status="needs_approval" if reasons else "ok", sql=sql, sql_to_run=sql_to_run,
                         reasons=reasons, pii_touched=touched, limit_applied=limit_applied, alias_map=alias_map)


def review(sql: str, db: ReadOnlyDB, max_rows: int = MAX_ROWS,
           scan_threshold: int = SCAN_ROW_THRESHOLD) -> GuardDecision:
    """static_check plus the plan-based checks (SQLite itself prepares the statement under
    the read-only authorizer, so this also catches anything the parser missed)."""
    d = static_check(sql, db, max_rows=max_rows)
    if d.blocked:
        return d
    try:
        scans = db.full_scans(d.sql_to_run, d.alias_map, allowed_pii=d.pii_touched)
    except QueryError as e:
        return _block(sql, f"SQLite rejected the query: {e}")
    big = sorted({(t, n) for t, n in scans if n > scan_threshold})
    if big:
        d.reasons.append("Full scan without an index on " + ", ".join(f"{t} ({n:,} rows)" for t, n in big)
                         + f"; threshold {scan_threshold:,} rows")
        d.status = "needs_approval"
    return d


def looks_like_write(sql: str) -> bool:
    """Independent classifier used by the evals to count write attempts (not used for enforcement)."""
    import re

    s = re.sub(r"--[^\n]*|/\*.*?(\*/|$)", " ", sql or "", flags=re.S).upper()
    s = re.sub(r"'(?:[^']|'')*'", "''", s)
    return bool(re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|TRUNCATE)\b"
                          r"|^\s*REPLACE\b|\bREPLACE\s+INTO\b", s))
