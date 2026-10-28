"""Polite, bounded page fetcher for live mode.

- http(s) only; refuses localhost / private / link-local IP literals (no SSRF into
  the deployment's own network).
- Honest user agent, robots.txt respected (401/403 on robots.txt = disallow all,
  other errors = allow, the usual crawler convention), one request per host per
  `min_interval_s`.
- Hard limits: connect/read timeouts, a total wall-clock deadline (a server that
  drips bytes can't hold a run hostage), and a byte cap on the body.
- Only text/html, xhtml and text/plain are converted; PDFs and binaries are
  reported as unsupported rather than guessed at.

Fetch failures come back as a result with `error` set, never as an exception, so
the agent can try another source and the verifier simply has no text to match.
"""

from __future__ import annotations

import ipaddress
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .text import html_to_text

USER_AGENT = "RidgelineResearchBot/0.1 (+https://buildwithmoses.vercel.app; demo research agent)"
UA_TOKEN = "RidgelineResearchBot"
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")


_BOT_WALL = re.compile(r"request access|access denied|captcha|are you a robot|verify you are human|"
                       r"enable javascript|just a moment|unusual traffic|attention required", re.I)


def looks_like_bot_wall(title: str, text: str) -> bool:
    """A short page whose title/text is a challenge or block notice. These come back
    as HTTP 200, so without this check a 'Request Access' page would count as a
    fetched source (seen live on ecfr.gov and federalregister.gov)."""
    return len(text) < 3000 and bool(_BOT_WALL.search(f"{title}\n{text[:1500]}"))


_FR_DOC = re.compile(r"^/documents/(\d{4})/(\d{2})/(\d{2})/([\w-]+)")


def official_api_url(url: str) -> str | None:
    """federalregister.gov serves a bot wall on its HTML document pages but publishes
    each document's raw text (and its robots.txt allows /documents/full_text), so
    map the human URL to that. eCFR was considered and left out on purpose: its
    robots.txt disallows the content APIs. Anything else: None (fetch as-is)."""
    parts = urlsplit(url)
    host, path = parts.netloc.lower(), parts.path
    if host in ("www.federalregister.gov", "federalregister.gov") and (m := _FR_DOC.match(path)):
        y, mo, d, doc = m.groups()
        return f"https://www.federalregister.gov/documents/full_text/text/{y}/{mo}/{d}/{doc}.txt"
    return None


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int = 0
    content_type: str = ""
    title: str = ""
    text: str = ""
    bytes_read: int = 0
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def normalize_url(url: str) -> str:
    """Canonical form used as a source id: lowercase scheme/host, no fragment,
    no trailing slash on the path."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") if parts.path not in ("", "/") else ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _blocked_host(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if not host or host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


@dataclass
class Fetcher:
    transport: httpx.BaseTransport | None = None
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 15.0
    deadline_s: float = 30.0
    max_bytes: int = 2_000_000
    min_interval_s: float = 1.0
    respect_robots: bool = True
    user_agent: str = USER_AGENT
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _robots: dict[str, RobotFileParser | None] = field(default_factory=dict, repr=False)
    _last_hit: dict[str, float] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _client(self) -> httpx.Client:
        timeout = httpx.Timeout(self.read_timeout_s, connect=self.connect_timeout_s)
        return httpx.Client(transport=self.transport, timeout=timeout, follow_redirects=True, max_redirects=5,
                            headers={"User-Agent": self.user_agent, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"})

    def _wait_turn(self, host: str) -> None:
        with self._lock:
            last = self._last_hit.get(host)
            now = self.clock()
            wait = 0.0 if last is None else self.min_interval_s - (now - last)
            self._last_hit[host] = now + max(wait, 0.0)
        if wait > 0:
            self.sleep(wait)

    def _robots_allows(self, client: httpx.Client, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            rp: RobotFileParser | None = RobotFileParser()
            try:
                self._wait_turn(parts.netloc)
                r = client.get(origin + "/robots.txt", timeout=httpx.Timeout(5.0))
                if r.status_code in (401, 403):
                    rp.disallow_all = True
                elif r.status_code >= 400:
                    rp = None  # no robots.txt: allowed
                else:
                    rp.parse(r.text[:200_000].splitlines())
            except httpx.HTTPError:
                rp = None
            self._robots[origin] = rp
        rp = self._robots[origin]
        return True if rp is None else rp.can_fetch(UA_TOKEN, url)

    def fetch(self, url: str) -> FetchResult:
        start = self.clock()
        res = FetchResult(url=url)
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https"):
            res.error = "only http(s) URLs can be fetched"
            return res
        if _blocked_host(parts.hostname or ""):
            res.error = "refusing to fetch a local or private address"
            return res
        target = official_api_url(url) or url
        try:
            with self._client() as client:
                if not self._robots_allows(client, target):
                    res.error = "disallowed by robots.txt"
                    return res
                self._wait_turn(parts.netloc)
                with client.stream("GET", target) as r:
                    res.status = r.status_code
                    res.final_url = str(r.url)
                    res.content_type = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    if _blocked_host(r.url.host):
                        res.error = "redirected to a local or private address"
                        return res
                    if r.status_code >= 400:
                        res.error = f"HTTP {r.status_code}"
                        return res
                    if res.content_type and not res.content_type.startswith(TEXT_TYPES):
                        res.error = f"unsupported content type {res.content_type!r}"
                        return res
                    chunks: list[bytes] = []
                    for chunk in r.iter_bytes():
                        chunks.append(chunk)
                        res.bytes_read += len(chunk)
                        if res.bytes_read >= self.max_bytes:
                            res.truncated = True
                            break
                        if self.clock() - start > self.deadline_s:
                            res.error = f"took longer than {self.deadline_s:.0f}s"
                            return res
                    body = b"".join(chunks)[: self.max_bytes]
                    encoding = r.encoding or "utf-8"
        except httpx.TimeoutException:
            res.error = "timed out"
            return res
        except httpx.HTTPError as exc:
            res.error = f"fetch failed: {type(exc).__name__}"
            return res
        finally:
            res.elapsed_s = round(self.clock() - start, 3)
        try:
            raw = body.decode(encoding, errors="replace")
        except LookupError:
            raw = body.decode("utf-8", errors="replace")
        if res.content_type.startswith(("text/plain", "text/markdown")):
            res.text = raw.strip()
        else:
            res.title, res.text = html_to_text(raw)
        if not res.text:
            res.error = "page had no extractable text (JavaScript-only page?)"
        elif looks_like_bot_wall(res.title, res.text):
            res.error, res.text = "blocked by a bot check / access wall (HTTP 200 challenge page)", ""
        return res
