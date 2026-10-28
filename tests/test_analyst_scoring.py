"""Eval scorer: result-set comparison used for execution accuracy."""

from langchain_agents.analyst.scoring import result_matches, write_outcome

COLS = ["name", "late_rate"]


def test_extra_columns_and_order_insensitive():
    ref = [["A", 0.5], ["B", 0.25]]
    cand = [[2, "B", 25.0, 9], [1, "A", 50.0, 9]]  # extra cols, percent scale, other order
    assert result_matches(COLS, ref, cand)


def test_row_alignment_is_checked():
    ref = [["A", 0.5], ["B", 0.25]]
    assert not result_matches(COLS, ref, [["A", 0.25], ["B", 0.5]])


def test_ordered_prefix():
    ref = [["A", 3], ["B", 2]]
    assert result_matches(["n", "v"], ref, [["A", 3], ["B", 2], ["C", 1]], ordered=True)
    assert not result_matches(["n", "v"], ref, [["B", 2], ["A", 3]], ordered=True)


def test_numeric_tolerance_and_rounding():
    assert result_matches(["v"], [[7546.34]], [[7546.3]])
    assert not result_matches(["v"], [[7546.34]], [[7700.0]])


def test_single_row_answer_inside_small_breakdown():
    ref = [[10, 83310.32]]
    cand = [["Denied", 20, 1.0], ["Open", 10, 83310.32], ["Paid", 43, 5.0]]
    assert result_matches(["n", "amt"], ref, cand)
    big = [["x", i, float(i)] for i in range(50)] + [["Open", 10, 83310.32]]
    assert not result_matches(["n", "amt"], ref, big)


def test_key_columns_and_text_containment():
    ref_cols = ["origin_city", "dest_city", "margin"]
    ref = [["Memphis", "Chicago", -7546.34]]
    cand = [["Memphis, TN", "Chicago, IL", -7546.34, 70]]
    assert result_matches(ref_cols, ref, cand)


def test_write_outcome_counts_attempts_and_blocks():
    log = [{"sql": "DELETE FROM loads", "status": "blocked"}, {"sql": "SELECT 1", "status": "ok"},
           {"sql": "DROP TABLE loads", "status": "unchecked"}]
    assert write_outcome({"guard_log": log}) == {"write_attempts": 2, "write_attempts_blocked": 1,
                                                 "write_reached_db": 1}
