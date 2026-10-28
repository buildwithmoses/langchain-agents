"""SQL guard: writes, DDL, stacked statements, hidden writes, LIMIT, PII policy, scan approval."""

import pytest

from langchain_agents.analyst.db import MAX_ROWS
from langchain_agents.analyst.sqlguard import looks_like_write, review, static_check
from langchain_agents.analyst.testing import shared_db


@pytest.fixture(scope="module")
def db():
    return shared_db()

WRITES = [
    "DELETE FROM loads WHERE status = 'Cancelled'",
    "delete from loads",
    "UPDATE carriers SET safety_rating = 'Satisfactory' WHERE carrier_id = 12",
    "INSERT INTO loads (load_id) VALUES (999999)",
    "INSERT OR REPLACE INTO claims VALUES (1,1,'x','Loss',1,1,'Open')",
    "REPLACE INTO claims VALUES (1,1,'x','Loss',1,1,'Open')",
    "DROP TABLE loads",
    "drop view if exists v",
    "CREATE TABLE x (a int)",
    "CREATE TEMP VIEW v AS SELECT 1",
    "CREATE INDEX ix ON loads(status)",
    "ALTER TABLE loads ADD COLUMN x int",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "attach 'file:/tmp/x?mode=rwc' as y",
    "DETACH DATABASE evil",
    "PRAGMA writable_schema = 1",
    "PRAGMA query_only = OFF",
    "pragma table_info(loads)",
    "VACUUM INTO '/tmp/copy.db'",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "BEGIN; DELETE FROM loads; COMMIT",
    "SAVEPOINT a",
    "EXPLAIN SELECT 1",
    "SELECT * INTO backup FROM loads",
    "SELECT load_extension('/tmp/x.so')",
]

STACKED_AND_HIDDEN = [
    "SELECT 1; DROP TABLE loads",
    "SELECT 1;DROP TABLE loads;",
    "SELECT count(*) FROM loads; -- harmless\nDELETE FROM loads",
    "SELECT 1 /* ; */ ; UPDATE loads SET status='x'",
    "/* comment */ DELETE FROM loads",
    "-- just a comment\nDROP TABLE loads",
    "WITH d AS (DELETE FROM loads RETURNING *) SELECT * FROM d",
    "WITH x AS (SELECT 1) INSERT INTO claims SELECT * FROM claims",
    "WITH x AS (SELECT 1) UPDATE loads SET status = 'Cancelled'",
    "SELECT * FROM (SELECT 1); ATTACH 'x' AS y",
    "SELECT 1 UNION ALL SELECT 2; PRAGMA query_only=0",
    "DeLeTe FrOm loads",
    "SELECT 1\n;\nDROP TABLE loads",
]


@pytest.mark.parametrize("sql", WRITES + STACKED_AND_HIDDEN)
def test_writes_ddl_and_stacked_statements_are_blocked(db, sql):
    d = review(sql, db)
    assert d.status == "blocked", (sql, d)
    assert d.reasons


@pytest.mark.parametrize("sql", WRITES + STACKED_AND_HIDDEN)
def test_eval_write_classifier_agrees(sql):
    if sql.startswith("SELECT load_extension") or sql.startswith("EXPLAIN") or "INTO backup" in sql \
            or sql in ("ANALYZE", "SAVEPOINT a"):
        return  # not data writes; still blocked by the guard above
    assert looks_like_write(sql)


@pytest.mark.parametrize("sql", [
    "SELECT count(*) FROM loads WHERE status = 'Cancelled'",
    "select name, replace(name, 'LLC', '') AS short FROM carriers LIMIT 3",
    "SELECT 'DELETE FROM loads' AS s",            # write keyword inside a string literal
    "SELECT status AS \"update\" FROM loads LIMIT 1",  # ... or a quoted identifier
    "WITH m AS (SELECT shipper_id, count(*) n FROM loads GROUP BY 1) SELECT * FROM m ORDER BY n DESC LIMIT 5",
    "SELECT origin_city FROM loads UNION SELECT dest_city FROM loads",
    "SELECT count(*) FROM loads; ",
    "SELECT count(*) FROM loads; -- trailing comment",
])
def test_legitimate_reads_pass(db, sql):
    d = review(sql, db)
    assert d.status == "ok", d


def test_trailing_semicolon_and_comment_are_stripped_before_limit(db):
    d = review("SELECT load_id FROM loads -- note", db)
    assert d.sql_to_run.endswith(f"LIMIT {MAX_ROWS}")
    assert "--" not in d.sql_to_run.splitlines()[-1]
    rows = db.execute(d.sql_to_run).rows
    assert len(rows) == MAX_ROWS


def test_limit_is_injected_when_missing(db):
    d = static_check("SELECT load_id FROM loads", db)
    assert d.limit_applied == MAX_ROWS and d.sql_to_run.endswith(f"LIMIT {MAX_ROWS}")


def test_small_limit_is_kept(db):
    d = static_check("SELECT load_id FROM loads ORDER BY load_id LIMIT 7", db)
    assert d.limit_applied is None and d.sql_to_run == "SELECT load_id FROM loads ORDER BY load_id LIMIT 7"


@pytest.mark.parametrize("sql", [
    "SELECT load_id FROM loads LIMIT 100000",
    "SELECT load_id FROM loads LIMIT -1",
    "SELECT load_id FROM loads LIMIT (SELECT 100000)",
    "SELECT load_id FROM loads LIMIT 5, 100000",
])
def test_large_or_computed_limit_is_capped(db, sql):
    d = review(sql, db)
    assert d.status == "ok" and d.limit_applied == MAX_ROWS
    assert len(db.execute(d.sql_to_run).rows) <= MAX_ROWS


def test_union_gets_a_limit(db):
    d = review("SELECT origin_city FROM loads UNION ALL SELECT dest_city FROM loads", db)
    assert len(db.execute(d.sql_to_run).rows) == MAX_ROWS


RAW_PII = [
    "SELECT contact_phone FROM carriers",
    "SELECT CONTACT_PHONE FROM CARRIERS",
    "SELECT \"contact_phone\" FROM carriers",
    "SELECT c.contact_phone AS p FROM carriers AS c",
    "SELECT name, contact_email AS e FROM carriers",
    "SELECT driver_name, driver_phone FROM loads",
    "SELECT * FROM carriers",
    "SELECT c.* FROM carriers c",
    "SELECT l.load_id, c.* FROM loads l JOIN carriers c USING (carrier_id)",
    "SELECT * FROM loads",                         # loads has driver PII
    "SELECT * FROM shippers",
    "SELECT x.p FROM (SELECT driver_phone AS p FROM loads) x",
    "WITH d AS (SELECT driver_phone FROM loads) SELECT count(*) FROM d",
    "SELECT max(contact_name) FROM carriers",
    "SELECT group_concat(contact_email) FROM carriers",
    "SELECT upper(contact_name) FROM carriers",
    "SELECT substr(contact_phone, 1, 8) FROM carriers",
    "SELECT contact_phone || '' FROM carriers",
    "SELECT name FROM carriers UNION SELECT contact_name FROM carriers",
    "SELECT (SELECT contact_phone FROM carriers c WHERE c.carrier_id = l.carrier_id) FROM loads l",
    "SELECT billing_email FROM shippers",
]


@pytest.mark.parametrize("sql", RAW_PII)
def test_raw_pii_selection_is_blocked(db, sql):
    d = review(sql, db)
    assert d.status == "blocked", d
    assert "PII" in " ".join(d.reasons)


@pytest.mark.parametrize("sql,col", [
    ("SELECT count(DISTINCT contact_email) FROM carriers", "carriers.contact_email"),
    ("SELECT count(*) FROM carriers WHERE contact_email LIKE '%gmail%'", "carriers.contact_email"),
    ("SELECT count(c.contact_phone) AS n FROM carriers c", "carriers.contact_phone"),
    ("SELECT name FROM carriers WHERE contact_name = 'Maria Garcia'", "carriers.contact_name"),
    ("SELECT count(*) FROM loads WHERE driver_phone IS NULL", "loads.driver_phone"),
])
def test_aggregate_or_filter_pii_use_needs_approval(db, sql, col):
    d = review(sql, db)
    assert d.status == "needs_approval", d
    assert col in " ".join(d.reasons)
    assert d.pii_touched


def test_select_star_on_non_pii_table_is_fine(db):
    assert review("SELECT * FROM claims", db).status == "ok"


def test_count_star_is_not_pii(db):
    assert review("SELECT count(*) FROM carriers", db).status == "ok"


def test_full_scan_of_big_table_needs_approval(db):
    d = review("SELECT status, count(*) FROM check_calls GROUP BY status", db)
    assert d.status == "needs_approval"
    assert "check_calls" in " ".join(d.reasons)


def test_indexed_lookup_on_big_table_is_fine(db):
    assert review("SELECT * FROM check_calls WHERE load_id = 10", db).status == "ok"


@pytest.mark.parametrize("sql", [
    "SELECT nosuch FROM loads",
    "SELECT * FROM nosuch_table",
    "SELECT load_id FROM main.loads",
    "SELECT load_id FROM temp.loads",
    "",
    "   ",
    "hello",
    "SELECT 1 /* unterminated",
    "(SELECT count(*) FROM loads)",  # SQLite rejects a parenthesised top-level select
])
def test_unknown_or_unparseable_fails_closed(db, sql):
    assert review(sql, db).status == "blocked"
