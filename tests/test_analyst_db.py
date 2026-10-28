"""Read-only connection, authorizer, timeout, row cap, PII masking/DLP, fixture determinism."""

import sqlite3

import pytest

from langchain_agents.analyst.db import MASK, QueryError, ReadOnlyDB
from langchain_agents.analyst.fixture_db import build_db
from langchain_agents.analyst.testing import shared_db


@pytest.fixture(scope="module")
def db():
    return shared_db()


@pytest.mark.parametrize("sql", [
    "DELETE FROM loads",
    "UPDATE loads SET status = 'x'",
    "INSERT INTO claims (claim_id, load_id, filed_at, claim_type, amount_claimed, amount_paid, status) "
    "VALUES (99999, 1, 'x', 'Loss', 1, 1, 'Open')",
    "DROP TABLE loads",
    "CREATE TABLE t (a)",
])
def test_connection_is_read_only_even_without_the_authorizer(db, sql):
    con = db.connect(enforce=False)  # mode=ro + query_only only
    with pytest.raises(sqlite3.Error):
        con.execute(sql)
    con.close()
    assert db.execute("SELECT count(*) FROM loads").rows[0][0] == 14400


@pytest.mark.parametrize("sql", [
    "DELETE FROM loads", "ATTACH DATABASE ':memory:' AS x", "PRAGMA query_only = OFF",
    "CREATE TEMP TABLE t (a)", "BEGIN",
])
def test_authorizer_denies_non_reads(db, sql):
    with pytest.raises(QueryError):
        db.execute(sql)


def test_execute_rejects_stacked_statements(db):
    with pytest.raises(QueryError):
        db.execute("SELECT 1; SELECT 2")


def test_authorizer_denies_pii_reads_unless_approved(db):
    with pytest.raises(QueryError, match="prohibited|not authorized"):
        db.execute("SELECT count(*) FROM carriers WHERE contact_email LIKE '%a%'")
    n = db.execute("SELECT count(*) FROM carriers WHERE contact_email LIKE '%a%'",
                   allowed_pii={("carriers", "contact_email")}).rows[0][0]
    assert n > 0


def test_timeout_stops_runaway_query(db):
    slow = ReadOnlyDB(db.path, timeout_s=0.2)
    with pytest.raises(QueryError, match="timeout"):
        slow.execute("WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) SELECT count(*) FROM r")


def test_row_cap_applies_even_if_sql_has_no_limit(db):
    res = db.execute("SELECT load_id FROM loads")
    assert len(res.rows) == db.max_rows and res.truncated


def test_dlp_masks_pii_values_that_slip_through(db):
    # Bypass the static guard: the authorizer allows the column, the DLP scan still masks it.
    res = db.execute("SELECT contact_phone, contact_email, name FROM carriers LIMIT 5",
                     allowed_pii={("carriers", "contact_phone"), ("carriers", "contact_email")})
    assert all(r[0] == MASK and r[1] == MASK for r in res.rows)
    assert all(r[2] != MASK for r in res.rows)
    assert res.redacted_cells == 10


def test_find_pii_in_text(db):
    phone = db.execute("SELECT contact_phone FROM carriers WHERE carrier_id = 1",
                       allowed_pii={("carriers", "contact_phone")}, enforce=False).rows[0][0]
    name = db.execute("SELECT contact_name FROM carriers WHERE carrier_id = 1", enforce=False).rows[0][0]
    assert db.find_pii_in_text(f"Call {name} at {phone}.")
    assert db.find_pii_in_text(f"digits only {''.join(c for c in phone if c.isdigit())}")
    assert not db.find_pii_in_text("Bluegrass Haulers LLC had 167 loads and a 51.5% late rate.")


def test_describe_table_masks_pii_and_flags_columns(db):
    d = db.describe_table("carriers")
    flags = {c["name"]: c["pii"] for c in d["columns"]}
    assert flags["contact_phone"] and flags["contact_email"] and not flags["name"]
    idx = [c["name"] for c in d["columns"]].index("contact_phone")
    assert all(r[idx] == MASK for r in d["sample_rows"])


def test_describe_unknown_table(db):
    with pytest.raises(QueryError):
        db.describe_table("users; DROP TABLE loads")


def test_fixture_is_deterministic_and_has_planted_patterns(tmp_path, db):
    other = ReadOnlyDB(build_db(tmp_path / "again.db"))
    q = "SELECT count(*), round(sum(shipper_rate - carrier_cost), 2) FROM loads"
    assert other.execute(q).rows == db.execute(q).rows
    sep = db.execute("SELECT round(sum(shipper_rate - carrier_cost), 2) FROM loads WHERE origin_city = 'Memphis' "
                     "AND dest_city = 'Chicago' AND status = 'Delivered' AND pickup_appt >= '2026-09'").rows[0][0]
    assert sep < 0
    worst = db.execute("SELECT c.name FROM loads l JOIN carriers c USING (carrier_id) WHERE l.status = 'Delivered' "
                       "GROUP BY c.carrier_id HAVING count(*) >= 50 "
                       "ORDER BY avg(l.delivery_actual > l.delivery_appt) DESC LIMIT 1").rows[0][0]
    assert worst == "Bluegrass Haulers LLC"
    counts = db.row_counts()
    assert counts["loads"] == 14400 and counts["shippers"] == 600 and counts["carriers"] == 1800
