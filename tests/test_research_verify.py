"""Citation verifier: the gate. Offline, no keys."""

from langchain_agents.research.verify import (
    check_claim, closest_passage, filter_answer, finalize, normalize, quote_in_text, verify_claims,
)

SRC_TEXT = ("Reefer spot rates out of the Southeast averaged $2.87 per mile in Q3 2026, up 14% from $2.52 in Q2.\n"
            "Dry van rates on the same lanes rose only 4% over the same period.\n\n"
            "Brokers “may no longer” include waiver clauses — effective February 1, 2027.")
SOURCES = {"corpus:memo": {"text": SRC_TEXT, "title": "Memo"},
           "https://example.com/page": {"text": SRC_TEXT, "title": "Page", "final_url": "https://www.example.com/page"}}


def claim(quote, source_id="corpus:memo", text="Reefer spot rates rose."):
    return {"claim": text, "source_id": source_id, "quote": quote}


def test_exact_quote_verifies():
    c = check_claim(1, claim("Reefer spot rates out of the Southeast averaged $2.87 per mile in Q3 2026"), SOURCES)
    assert c.verified and c.reason == "ok"


def test_whitespace_case_and_quote_normalization():
    q = '"reefer   SPOT rates out of the\nSoutheast averaged $2.87 per mile in Q3 2026."'
    assert check_claim(1, claim(q), SOURCES).verified
    # curly quotes and em dash in the source match straight quotes / hyphen in the quote
    q2 = 'Brokers "may no longer" include waiver clauses - effective February 1, 2027'
    assert check_claim(1, claim(q2, text="Waiver clauses are banned from February 1, 2027."), SOURCES).verified


def test_quote_spanning_line_break_and_markdown_emphasis():
    assert quote_in_text("in Q2. Dry van rates on the same lanes", SRC_TEXT)
    assert quote_in_text("the **same** lanes rose only", SRC_TEXT)


def test_fabricated_quote_rejected_with_closest_passage():
    c = check_claim(1, claim("Reefer spot rates out of the Southeast averaged $2.78 per mile in Q3 2026"), SOURCES)
    assert not c.verified and c.reason == "quote_not_found"
    assert c.closest_ratio and c.closest_ratio > 0.9  # a near miss is still rejected: one digit is the whole point
    assert "$2.87" in c.closest_passage


def test_paraphrase_rejected():
    c = check_claim(1, claim("Southeast reefer rates averaged about $2.87 a mile last quarter"), SOURCES)
    assert not c.verified and c.reason == "quote_not_found"


def test_quote_from_unfetched_page_rejected():
    c = check_claim(1, claim("Reefer spot rates out of the Southeast averaged $2.87 per mile",
                             source_id="https://other.example.org/never-fetched"), SOURCES)
    assert not c.verified and c.reason == "source_not_fetched"
    # a source that failed to fetch (no text) also doesn't count
    failed = {"https://x.example.com/a": {"text": "", "error": "HTTP 403"}}
    assert check_claim(1, claim("anything at all goes here ok", source_id="https://x.example.com/a"),
                       failed).reason == "source_not_fetched"


def test_url_source_id_normalized_and_redirect_target_accepted():
    q = "Dry van rates on the same lanes rose only 4% over the same period"
    for sid in ("https://EXAMPLE.com/page/", "https://example.com/page#section", "https://www.example.com/page"):
        assert check_claim(1, claim(q, source_id=sid, text="Dry van rose 4%."), SOURCES).verified, sid


def test_too_short_quote_rejected():
    assert check_claim(1, claim("up 14%"), SOURCES).reason == "quote_too_short"


def test_ellipsis_segments_must_appear_in_order():
    ok = "Reefer spot rates out of the Southeast ... up 14% from $2.52 in Q2"
    assert check_claim(1, claim(ok, text="Rates rose 14% from $2.52."), SOURCES).verified
    backwards = "up 14% from $2.52 in Q2 ... Reefer spot rates out of the Southeast"
    assert not check_claim(1, claim(backwards), SOURCES).verified


def test_number_in_claim_must_be_in_quote():
    q = "Dry van rates on the same lanes rose only 4% over the same period"
    bad = check_claim(1, claim(q, text="Dry van rates rose 6% on the same lanes."), SOURCES)
    assert not bad.verified and bad.reason == "number_not_in_quote"
    # a year that appears in the source is allowed even if not in the quote
    assert check_claim(1, claim(q, text="In 2026, dry van rates rose 4%."), SOURCES).verified


def test_normalize_is_stable():
    assert normalize("A “B” – C\n\nD") == 'a "b" - c d'


def test_closest_passage_handles_empty():
    assert closest_passage("", "text") == (0.0, "")


def test_filter_answer_keeps_only_sentences_backed_by_verified_claims():
    answer = "Rates rose 14% [1]. Waivers are banned [2]. Fuel is up [1][3]. This sentence cites nothing."
    text, dropped = filter_answer(answer, verified={1, 2}, renumber={1: 1, 2: 2})
    assert text == "Rates rose 14% [1]. Waivers are banned [2]."
    assert dropped == 2


def test_final_brief_citation_precision_is_one_by_construction():
    """Mixed good/bad claims in, only verified claims out, renumbered, and every
    surviving quote re-verifies against the sources."""
    brief = {"answerable": True, "gaps": [], "answer": "Rates rose [1]. Made up [2]. Dry van rose [3].", "claims": [
        claim("Reefer spot rates out of the Southeast averaged $2.87 per mile in Q3 2026", text="Rates hit $2.87 in Q3 2026."),
        claim("Reefer rates tripled overnight according to every carrier we asked", text="Rates tripled."),
        claim("Dry van rates on the same lanes rose only 4% over the same period", text="Dry van rose 4%."),
    ]}
    final = finalize(brief, verify_claims(brief["claims"], SOURCES))
    assert [c["n"] for c in final["claims"]] == [1, 2]
    assert final["answer"] == "Rates rose [1]. Dry van rose [2]."
    assert len(final["dropped"]) == 1 and final["dropped"][0]["reason"] == "quote_not_found"
    recheck = verify_claims(final["claims"], SOURCES)
    assert all(c.verified for c in recheck)  # precision 1.0


def test_finalize_fails_closed_without_brief_or_without_verified_claims():
    assert finalize(None, [])["claims"] == [] and not finalize(None, [])["answerable"]
    brief = {"answerable": True, "gaps": [], "answer": "Big claim [1].",
             "claims": [claim("this quote was never in the source text at all")]}
    final = finalize(brief, verify_claims(brief["claims"], SOURCES))
    assert not final["answerable"] and final["claims"] == []
