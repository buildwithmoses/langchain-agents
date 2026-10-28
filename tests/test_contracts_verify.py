"""Quote verifier and deterministic sectioning. Offline."""

from pathlib import Path

from langchain_agents.contracts.fakes import compliant_extraction
from langchain_agents.contracts.playbook import VERIFY_SPEC
from langchain_agents.contracts.sections import locate, render_for_prompt, split_sections
from langchain_agents.contracts.verify import normalize, numbers_in, quote_in_text, verify_term

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
TEXT = (FIX / "01-bluegrass-haulers.md").read_text()


def test_normalize_handles_quotes_dashes_markup_whitespace():
    assert normalize("“Carrier’s”  **liability** –\n  full") == "\"carrier's\" liability - full"


def test_exact_and_normalized_quotes_verify():
    assert quote_in_text("Either Party may terminate this Agreement for any reason", TEXT)
    assert quote_in_text("either party   may terminate this agreement for any reason", TEXT)  # case + whitespace
    assert quote_in_text("Carrier’s liability shall be for the full actual value of the cargo", TEXT)  # curly apostrophe


def test_ellipsis_fragments_must_appear_in_order():
    assert quote_in_text("Either Party may terminate ... thirty (30) days' prior written notice", TEXT)
    # 11.2 comes after 5.2, so the reversed order must fail.
    assert not quote_in_text("for any reason upon thirty (30) days' prior written notice ... Ridgeline shall pay each undisputed invoice", TEXT)


def test_paraphrase_and_tiny_quotes_fail():
    assert not quote_in_text("Either party can end the agreement with 30 days notice", TEXT)
    assert not quote_in_text("Carrier", TEXT)  # too short to prove anything
    assert not quote_in_text("", TEXT)


def test_numbers_in_parses_digits_words_and_money():
    assert {30, 100000, 1000000} <= numbers_in("thirty (30) days, $100,000 and One Million Dollars ($1,000,000)")
    assert 45 in numbers_in("forty-five days")
    assert 125000 in numbers_in("One Hundred Twenty-Five Thousand Dollars")
    assert 1000000 in numbers_in("$1 million")


def test_verify_term_passes_correct_extraction():
    ext = compliant_extraction()
    for term, spec in VERIFY_SPEC.items():
        assert verify_term(ext[term], TEXT, **spec) == [], term


def test_verify_term_catches_fabricated_quote():
    t = compliant_extraction(termination={"quotes": ["Either party may terminate on ninety (90) days notice."],
                                          "termination_notice_days": 90})["termination"]
    problems = verify_term(t, TEXT, **VERIFY_SPEC["termination"])
    assert problems and "not found verbatim" in problems[0]


def test_verify_term_catches_value_not_in_quote():
    # Real quote, wrong value: the number must appear inside the quote.
    t = compliant_extraction(termination={"termination_notice_days": 90})["termination"]
    assert any("termination_notice_days=90" in p for p in verify_term(t, TEXT, **VERIFY_SPEC["termination"]))


def test_verify_term_requires_certificate_holder_phrase():
    ins = compliant_extraction()["insurance"]
    ins["quotes"] = ins["quotes"][:3]  # drop the certificate-holder quote
    assert any("certificate holder" in p for p in verify_term(ins, TEXT, **VERIFY_SPEC["insurance"]))


def test_found_without_quotes_fails_and_absent_passes():
    t = {"found": True, "quotes": [], "section_ref": "9"}
    assert verify_term(t, TEXT) == ["found=true but no quote given"]
    assert verify_term({"found": False, "quotes": []}, TEXT) == []


def test_split_sections_top_level_subsections_and_exhibits():
    text = (FIX / "06-ozark-ridge-trucking.md").read_text()
    secs = split_sections(text)
    tops = [s.ref for s in secs if s.level == 1]
    assert tops == ["Preamble", *map(str, range(1, 13)), "Exhibit A", "Exhibit B"]
    subs = {s.ref: s for s in secs if s.level == 2}
    assert "forty-five (45) days" in subs["1.4"].text
    assert subs["B.1"].parent == "Exhibit B"
    assert "Seven Hundred Fifty Thousand" in subs["B.1"].text


def test_split_sections_article_headings_and_preamble():
    secs = split_sections((FIX / "02-iron-horse-transport.md").read_text())
    tops = [s for s in secs if s.level == 1]
    assert tops[0].ref == "Preamble" and "MOTOR CARRIER TRANSPORTATION AGREEMENT" in tops[0].title
    assert [s.ref for s in tops[1:]] == [str(i) for i in range(1, 13)]


def test_sections_cover_whole_document_in_prompt():
    text = (FIX / "08-gulfline-carriers.md").read_text()
    raw = render_for_prompt(split_sections(text))
    assert "=== [Exhibit A | Exhibit A" in raw  # exhibits are labelled as part of the agreement
    assert "not to exceed fifty thousand dollars ($50,000) per shipment" in normalize(raw)


def test_locate_returns_most_specific_section():
    text = (FIX / "06-ozark-ridge-trucking.md").read_text()
    secs = split_sections(text)
    assert locate('"Notice Period" means forty-five (45) days.', secs, text) == "1.4"
    assert locate("Automobile liability: not less than Seven Hundred Fifty Thousand Dollars ($750,000)", secs, text) == "Exhibit B, B.1"
    assert locate("This sentence is not in the contract at all.", secs, text) is None


def test_sectioning_is_deterministic():
    text = (FIX / "05-red-river-carriers.md").read_text()
    assert [s.to_dict() for s in split_sections(text)] == [s.to_dict() for s in split_sections(text)]
