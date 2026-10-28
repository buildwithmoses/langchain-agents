"""HTML to plain text with the standard library (no extra dependency).

Good enough for quoting: drops script/style/nav chrome, keeps block structure as
line breaks, decodes entities. Quotes are verified against this text, so what
matters is that it is deterministic and keeps sentences intact.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "nav", "footer", "head", "form", "button"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "aside", "li", "ul", "ol", "table", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dd", "dt", "dl", "figcaption", "br", "hr",
}
VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed", "param", "track"}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.pre_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self.in_title = True
        if tag == "pre":
            self.pre_depth += 1
        if tag in SKIP_TAGS and tag not in VOID_TAGS:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in BLOCK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
        if tag in SKIP_TAGS and tag not in VOID_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in BLOCK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if not self.skip_depth:
            # Source newlines inside a paragraph are just whitespace in HTML.
            self.parts.append(data if self.pre_depth else re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text)."""
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep whatever was parsed
        pass
    text = "".join(parser.parts)
    lines = [re.sub(r"[ \t\r\f\v ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    return title, text
