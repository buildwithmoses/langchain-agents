"""Number check: answer numbers must come from the results (rounding allowed)."""

from langchain_agents.analyst.numbers import check_numbers, extract_mentions

Q = [{"sql": "SELECT ... HAVING count(*) >= 50 LIMIT 5",
      "rows": [["Bluegrass Haulers LLC", 167, 0.515], ["Delta Freight", 171, 0.1754]]},
     {"sql": "SELECT month, margin", "rows": [["2026-09", -7546.34], ["2026-08", 13455.52]]}]


def ok(text, queries=Q, question=""):
    return check_numbers(text, queries, question).passed


def test_exact_and_formatted_numbers_pass():
    assert ok("Bluegrass Haulers LLC ran 167 loads.")
    assert ok("The lane lost $7,546.34 in September 2026.")
    assert ok("The lane lost -$7,546.34.")
    assert ok("Margin was $13,455.52 in August.")


def test_rounding_passes():
    assert ok("The lane lost about $7,546.")
    assert ok("The lane lost about $7,500.")
    assert ok("The lane lost roughly $7.5k.")
    assert ok("August margin was $13.5k.")


def test_percent_of_ratio_passes():
    assert ok("51.5% of deliveries were late.")
    assert ok("About 52% of deliveries were late.")
    assert ok("Delta Freight was late 17.5% of the time.")


def test_invented_or_computed_numbers_fail():
    r = check_numbers("Bluegrass ran 185 loads and the lane swung by $21,001.86.", Q)
    assert not r.passed
    assert set(r.unsupported) == {"185", "$21,001.86"}
    assert not ok("The lane lost about $8,000.")  # rounding too coarse (6%)
    assert not ok("Late rate was 34%.")


def test_allowed_context_numbers():
    assert ok("At least 50 loads were required.")          # threshold in the SQL that ran
    assert ok("Here are the top 2 carriers.")               # row count of a result
    assert ok("Of the 3 carriers you asked about...", question="compare these 3 carriers")
    assert ok("Data covers 2026 only.")                     # calendar year


def test_list_markers_and_code_are_ignored():
    assert ok("1. Bluegrass Haulers LLC: 51.5%\n2. Delta Freight: 17.5%")
    assert ok("I ran `SELECT 999 FROM loads` for this.")


def test_no_numbers_passes():
    assert ok("I can't delete data; I'm read-only.", queries=[])


def test_no_queries_but_numbers_fails():
    assert not check_numbers("There were 412 cancelled loads.", []).passed


def test_extract_mentions_handles_suffixes():
    ms = extract_mentions("$1.2M and 40k loads and 12.5% and 3,400")
    assert [round(m.value) for m in ms] == [1200000, 40000, 12, 3400]
    assert ms[2].percent
