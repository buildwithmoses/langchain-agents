"""Corpus search/read, HTML-to-text, the fetcher's limits (mocked transport) and
the OpenRouter web search parser. Offline, no keys."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from langchain_agents.research.corpus import Corpus
from langchain_agents.research.fetch import Fetcher, normalize_url
from langchain_agents.research.text import html_to_text
from langchain_agents.research.tools import MAX_CHARS_TO_MODEL, make_tools
from langchain_agents.research.websearch import OpenRouterWebSearch

DATASET = Path(__file__).resolve().parents[1] / "evals" / "research" / "dataset.jsonl"


def runtime(state=None):
    return SimpleNamespace(state=state or {}, tool_call_id="call-1")


# ---------- corpus ----------

def test_corpus_loads_fictional_docs():
    c = Corpus()
    assert 8 <= len(c.docs) <= 12
    assert all("FICTIONAL" in d.body for d in c.docs.values())


@pytest.mark.parametrize("case", [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()],
                         ids=lambda c: c["id"])
def test_corpus_search_finds_expected_source_top3(case):
    hits = [h["source_id"] for h in Corpus().search(case["inputs"]["question"], k=3)]
    for sid in case["outputs"]["expected_sources"]:
        assert sid in hits, (case["inputs"]["question"], hits)


def test_corpus_get_accepts_id_forms():
    c = Corpus()
    for sid in ("corpus:ltl-rate-outlook-2027", "ltl-rate-outlook-2027", "ltl-rate-outlook-2027.md"):
        assert c.get(sid).doc_id == "ltl-rate-outlook-2027"
    assert c.get("corpus:nope") is None


def test_corpus_tools_record_search_and_source_in_state():
    search, read = make_tools("corpus")
    cmd = search.func("reefer spot rates Southeast", runtime())
    (sid, rec), = cmd.update["searches"].items()
    assert rec["result_ids"][0] == "corpus:ridgeline-market-memo-q3-2026"
    cmd = read.func("corpus:ridgeline-market-memo-q3-2026", runtime())
    src = cmd.update["sources"]["corpus:ridgeline-market-memo-q3-2026"]
    assert "$2.87 per mile" in src["text"] and src["kind"] == "corpus"
    assert "TEXT" in cmd.update["messages"][0].content


def test_read_unknown_doc_records_nothing():
    _, read = make_tools("corpus")
    cmd = read.func("corpus:made-up", runtime())
    assert "sources" not in cmd.update and "No document" in cmd.update["messages"][0].content


def test_search_and_source_budgets_enforced_from_state():
    search, read = make_tools("corpus", max_searches=2, max_sources=1)
    full = runtime({"searches": {"a": {}, "b": {}}})
    assert "budget" in search.func("reefer", full).update["messages"][0].content
    one = runtime({"sources": {"corpus:x": {"text": "t"}}})
    assert "budget" in read.func("corpus:ltl-rate-outlook-2027", one).update["messages"][0].content


# ---------- HTML to text ----------

def test_html_to_text_strips_chrome_and_keeps_sentences():
    html = """<html><head><title>FMCSA &amp; Rules</title><style>.x{}</style><script>var a=1;</script></head>
    <body><nav>Home | About</nav><h1>Broker rule</h1><p>Brokers must keep records
    for <b>three</b>&nbsp;years.</p><ul><li>One</li><li>Two</li></ul><footer>Copyright</footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "FMCSA & Rules"
    assert "Brokers must keep records for three years." in text
    for junk in ("var a", ".x{}", "Home | About", "Copyright"):
        assert junk not in text
    assert "One\n" in text and "Broker rule" in text


def test_html_to_text_survives_malformed_markup():
    _, text = html_to_text("<p>Unclosed <b>bold <div>and more")
    assert "Unclosed bold" in text and "and more" in text


# ---------- fetcher ----------

def fetcher(handler, **kw):
    kw.setdefault("sleep", lambda s: None)
    return Fetcher(transport=httpx.MockTransport(handler), **kw)


def page(request):
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                          html="<html><title>T</title><p>Hello world sentence.</p></html>")


def test_fetch_ok_sets_headers_and_text():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return page(request)

    r = fetcher(handler).fetch("https://example.com/a")
    assert r.ok and r.title == "T" and "Hello world sentence." in r.text
    assert "RidgelineResearchBot" in seen["ua"]


def test_fetch_size_cap_truncates():
    big = "<p>" + ("word " * 100_000) + "</p>"

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=big.encode())

    r = fetcher(handler, max_bytes=10_000).fetch("https://example.com/big")
    assert r.truncated and r.bytes_read >= 10_000 and len(r.text) <= 10_000


def test_fetch_timeout_is_an_error_not_an_exception():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    r = fetcher(handler, respect_robots=False).fetch("https://example.com/slow")
    assert not r.ok and r.error == "timed out"


def test_fetch_total_deadline():
    ticks = iter(range(0, 1000, 20))  # every clock read advances 20s

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"},
                              stream=httpx.ByteStream(b"x" * 50_000))

    r = fetcher(handler, respect_robots=False, deadline_s=30, clock=lambda: next(ticks)).fetch("https://example.com/drip")
    assert not r.ok and "longer than" in r.error


def test_fetch_respects_robots_disallow():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return page(request)

    f = fetcher(handler)
    assert f.fetch("https://example.com/private/x").error == "disallowed by robots.txt"
    assert f.fetch("https://example.com/public").ok


def test_fetch_rejects_pdf_errors_and_private_hosts():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/doc.pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
        return httpx.Response(500)

    f = fetcher(handler)
    assert "unsupported content type" in f.fetch("https://example.com/doc.pdf").error
    assert f.fetch("https://example.com/err").error == "HTTP 500"
    for url in ("http://127.0.0.1/x", "http://localhost:8000/", "http://10.0.0.5/admin", "file:///etc/passwd"):
        assert not f.fetch(url).ok, url


def test_fetch_politeness_waits_between_hits_to_same_host():
    waits = []
    now = [100.0]
    f = fetcher(page, respect_robots=False, min_interval_s=1.0, sleep=waits.append, clock=lambda: now[0])
    f.fetch("https://example.com/1")
    f.fetch("https://example.com/2")
    assert waits and waits[0] == pytest.approx(1.0)


def test_normalize_url():
    assert normalize_url("HTTPS://Example.COM/a/b/#frag") == "https://example.com/a/b"
    assert normalize_url("https://example.com/") == "https://example.com"


def test_web_read_tool_records_failure_without_text():
    def handler(request):
        return httpx.Response(404)

    _, read = make_tools("web", searcher=OpenRouterWebSearch(api_key="x"), fetcher=fetcher(handler))
    cmd = read.func("https://example.com/missing", runtime())
    rec = cmd.update["sources"]["https://example.com/missing"]
    assert rec["text"] == "" and rec["error"] == "HTTP 404"


def test_web_read_tool_caps_text_shown_to_model():
    long = "<p>" + "Sentence here. " * 5000 + "</p>"

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, html=long)

    _, read = make_tools("web", searcher=OpenRouterWebSearch(api_key="x"), fetcher=fetcher(handler))
    cmd = read.func("https://example.com/long", runtime())
    assert len(cmd.update["sources"]["https://example.com/long"]["text"]) > MAX_CHARS_TO_MODEL
    assert "more characters not shown" in cmd.update["messages"][0].content


# ---------- web search response parsing ----------

def test_openrouter_web_search_parses_url_citations_and_cost():
    body = {"choices": [{"message": {"role": "assistant", "content": "- https://a.example/x - A", "annotations": [
        {"type": "url_citation", "url_citation": {"url": "https://a.example/x", "title": "A", "content": "snip"}},
        {"type": "url_citation", "url_citation": {"url": "https://a.example/x", "title": "dup", "content": ""}},
        {"type": "url_citation", "url_citation": {"url": "https://b.example/y", "title": "B", "content": "s2"}},
    ]}}], "usage": {"cost": 0.0076, "server_tool_use_details": {"web_search_requests": 1}}}
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=body)

    res = OpenRouterWebSearch(api_key="k", model="openrouter:anthropic/claude-haiku-5.5",
                              transport=httpx.MockTransport(handler)).search("broker rule")
    assert [r["url"] for r in res["results"]] == ["https://a.example/x", "https://b.example/y"]
    assert res["cost_usd"] == pytest.approx(0.0076) and res["web_search_requests"] == 1
    assert sent["model"] == "anthropic/claude-haiku-5.5"
    assert sent["tools"][0]["type"] == "openrouter:web_search" and sent["tools"][0]["parameters"]["max_uses"] == 1


def test_bot_wall_page_is_a_failed_fetch_not_a_source():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"},
                              html="<title>Federal Register :: Request Access</title><p>Request Access. "
                                   "Due to aggressive automated scraping, programmatic access is limited.</p>")

    r = fetcher(handler).fetch("https://www.example.gov/doc")
    assert not r.ok and "bot check" in r.error and r.text == ""


def test_federal_register_urls_are_fetched_from_the_official_text_endpoint():
    from langchain_agents.research.fetch import official_api_url

    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /documents/current\n")
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="SUMMARY: The rule takes effect.")

    url = "https://www.federalregister.gov/documents/2024/11/04/2024-25517/broker-and-freight-forwarder"
    assert official_api_url(url) == "https://www.federalregister.gov/documents/full_text/text/2024/11/04/2024-25517.txt"
    assert official_api_url("https://www.ecfr.gov/current/title-49/part-387") is None  # robots disallows its API
    r = fetcher(handler).fetch(url)
    assert r.ok and "takes effect" in r.text and seen[-1].endswith("2024-25517.txt")
