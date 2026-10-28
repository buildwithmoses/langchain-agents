"""Read-only access to the Ridgeline TMS fixture. Every layer here assumes the one
above it failed:

1. The file is opened with a `mode=ro` URI and `PRAGMA query_only=ON`.
2. A SQLite authorizer allows only SELECT/READ/FUNCTION (no writes, ATTACH, PRAGMA,
   temp objects, transactions), denies dangerous functions, and denies any read of a
   PII column unless the caller says the guard approved that column.
3. A progress handler aborts any statement that runs past the timeout.
4. Rows are fetched with fetchmany(limit) whatever the SQL says.
5. Results are scanned for known PII values (the DLP check) before anyone sees them.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_agents.core.config import data_dir

from .fixture_db import PII_COLUMNS, ensure_db

MAX_ROWS = 200
TIMEOUT_S = 5.0
# A full scan of a table bigger than this needs a human's OK (check_calls is ~40k rows).
SCAN_ROW_THRESHOLD = 20_000
DENIED_FUNCTIONS = {"load_extension", "fts3_tokenizer", "readfile", "writefile", "edit", "sqlite_offset"}
MASK = "[PII]"


def default_db_path() -> Path:
    return data_dir("analyst", "ridgeline.db")


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float
    redacted_cells: int = 0

    def to_dict(self) -> dict:
        return {"sql": self.sql, "columns": self.columns, "rows": [list(r) for r in self.rows],
                "truncated": self.truncated, "elapsed_ms": round(self.elapsed_ms, 1),
                "redacted_cells": self.redacted_cells}


class QueryError(Exception):
    pass


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


@dataclass
class ReadOnlyDB:
    path: Path = field(default_factory=default_db_path)
    timeout_s: float = TIMEOUT_S
    max_rows: int = MAX_ROWS
    pii: dict[str, set[str]] = field(default_factory=lambda: {k: set(v) for k, v in PII_COLUMNS.items()})

    def __post_init__(self) -> None:
        self.path = ensure_db(Path(self.path))
        self._schema: dict[str, list[tuple[str, str]]] | None = None
        self._counts: dict[str, int] | None = None
        self._pii_values: tuple[set[str], set[str]] | None = None

    # ---- connections -------------------------------------------------------------
    def connect(self, allowed_pii: set[tuple[str, str]] | None = None, enforce: bool = True) -> sqlite3.Connection:
        """A fresh read-only connection. `enforce=False` skips the authorizer (used only by
        the no-guard eval variant to measure what the guard is worth; mode=ro still applies)."""
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=1.0, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")
        if enforce:
            allowed = allowed_pii or set()
            pii = self.pii

            def authorizer(action, arg1, arg2, dbname, source):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ:
                    table, column = (arg1 or "").lower(), (arg2 or "").lower()
                    if column in pii.get(table, ()) and (table, column) not in allowed:
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION:
                    return sqlite3.SQLITE_DENY if (arg2 or "").lower() in DENIED_FUNCTIONS else sqlite3.SQLITE_OK
                if action == getattr(sqlite3, "SQLITE_RECURSIVE", 33):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            con.set_authorizer(authorizer)
        return con

    # ---- metadata (read with the authorizer off, but never returned unmasked) -------
    def schema(self) -> dict[str, list[tuple[str, str]]]:
        if self._schema is None:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            try:
                tables = [r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
                self._schema = {t: [(r[1], r[2]) for r in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}
                self._counts = {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables}
            finally:
                con.close()
        return self._schema

    def row_counts(self) -> dict[str, int]:
        self.schema()
        return dict(self._counts or {})

    def sqlglot_schema(self) -> dict[str, dict[str, str]]:
        return {t: {c: (typ or "TEXT") for c, typ in cols} for t, cols in self.schema().items()}

    def is_pii(self, table: str, column: str) -> bool:
        return column.lower() in self.pii.get(table.lower(), set())

    def list_tables(self) -> list[dict]:
        counts = self.row_counts()
        return [{"table": t, "rows": counts[t], "columns": len(cols)} for t, cols in self.schema().items()]

    def describe_table(self, table: str, sample_rows: int = 3) -> dict:
        schema = self.schema()
        if table not in schema:
            raise QueryError(f"Unknown table {table!r}. Tables: {', '.join(schema)}")
        cols = schema[table]
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            rows = con.execute(f'SELECT * FROM "{table}" LIMIT ?', (sample_rows,)).fetchall()
        finally:
            con.close()
        masked = [[MASK if self.is_pii(table, c) else v for (c, _), v in zip(cols, r)] for r in rows]
        return {
            "table": table,
            "rows": self.row_counts()[table],
            "columns": [{"name": c, "type": t, "pii": self.is_pii(table, c)} for c, t in cols],
            "sample_rows": masked,
        }

    # ---- planning ----------------------------------------------------------------------
    def query_plan(self, sql: str, allowed_pii: set[tuple[str, str]] | None = None) -> list[str]:
        con = self.connect(allowed_pii)
        try:
            return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql)]
        except sqlite3.Error as e:
            raise QueryError(str(e)) from e
        finally:
            con.close()

    def full_scans(self, sql: str, alias_map: dict[str, str], allowed_pii=None) -> list[tuple[str, int]]:
        """Tables the plan reads end to end and their row counts. SQLite reports a SEARCH
        for an index range and a SCAN for a full pass, including a full pass over a covering
        index, which still reads every row."""
        counts = self.row_counts()
        out = []
        for detail in self.query_plan(sql, allowed_pii):
            m = re.match(r"SCAN (?:TABLE )?(\S+)", detail)
            if not m:
                continue
            name = m.group(1).strip('"')
            table = alias_map.get(name.lower(), name.lower())
            if table in counts:
                out.append((table, counts[table]))
        return out

    # ---- execution ---------------------------------------------------------------------
    def execute(self, sql: str, allowed_pii: set[tuple[str, str]] | None = None, enforce: bool = True) -> QueryResult:
        con = self.connect(allowed_pii, enforce=enforce)
        deadline = time.monotonic() + self.timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        t0 = time.perf_counter()
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise QueryError(f"Query exceeded the {self.timeout_s:.0f}s timeout and was stopped.") from e
            raise QueryError(str(e)) from e
        except (sqlite3.Error, sqlite3.Warning) as e:  # Warning: "only one statement at a time"
            raise QueryError(str(e)) from e
        finally:
            con.close()
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        res = QueryResult(sql=sql, columns=columns, rows=[tuple(r) for r in rows], truncated=truncated,
                          elapsed_ms=(time.perf_counter() - t0) * 1000)
        if enforce:
            res = self.redact(res)
        return res

    # ---- DLP: last line of defence ---------------------------------------------------------
    def pii_values(self) -> tuple[set[str], set[str]]:
        """(lowercased names/emails, digit-only phones) present in PII columns."""
        if self._pii_values is None:
            texts, phones = set(), set()
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            try:
                for table, cols in self.pii.items():
                    for col in cols:
                        for (v,) in con.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL'):
                            if "phone" in col:
                                phones.add(_digits(str(v)))
                            else:
                                texts.add(str(v).lower())
            finally:
                con.close()
            self._pii_values = (texts, phones)
        return self._pii_values

    def contains_pii(self, value: object) -> bool:
        if not isinstance(value, str) or len(value) < 5:
            return False
        texts, phones = self.pii_values()
        low = value.lower().strip()
        if low in texts:
            return True
        d = _digits(value)
        if len(d) >= 10 and (d in phones or d[-10:] in phones):
            return True
        if "@" in low and any(tok.strip(".,;:()<>") in texts for tok in low.split()):
            return True
        return False

    def find_pii_in_text(self, text: str) -> list[str]:
        """Known PII values that appear anywhere in free text (answers, tables)."""
        texts, phones = self.pii_values()
        low = text.lower()
        hits = []
        for m in re.finditer(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text):
            if _digits(m.group()) in phones:
                hits.append(m.group())
        for m in re.finditer(r"[\w.+-]+@[\w.-]+", low):
            if m.group().rstrip(".") in texts:
                hits.append(m.group())
        names = {t for t in texts if "@" not in t and " " in t}
        for m in re.finditer(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text):
            if m.group(1).lower() in names:
                hits.append(m.group(1))
        return hits

    def redact(self, res: QueryResult) -> QueryResult:
        n = 0
        rows = []
        for r in res.rows:
            new = []
            for v in r:
                if self.contains_pii(v):
                    new.append(MASK)
                    n += 1
                else:
                    new.append(v)
            rows.append(tuple(new))
        res.rows, res.redacted_cells = rows, n
        return res
