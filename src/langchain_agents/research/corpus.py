"""Offline corpus mode: a folder of short fictional source documents with a
small BM25 search. Deterministic, no network, used by tests and evals."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_agents.core.config import data_dir

STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or our that the their this to was were what "
    "which who will with we you your does do did than then there these those about into over under out".split()
)
PREFIX = "corpus:"


def default_corpus_dir() -> Path:
    return data_dir("research", "corpus")


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Doc:
    doc_id: str
    title: str
    publisher: str
    date: str
    body: str

    @property
    def source_id(self) -> str:
        return PREFIX + self.doc_id

    @property
    def text(self) -> str:
        """The text that counts as 'fetched' for verification."""
        return f"{self.title}\n{self.publisher}, {self.date}\n\n{self.body}"


def _parse(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return Doc(path.stem, meta.get("title", path.stem), meta.get("publisher", ""), meta.get("date", ""), body.strip())


class Corpus:
    def __init__(self, folder: Path | str | None = None) -> None:
        self.folder = Path(folder or default_corpus_dir())
        self.docs = {d.doc_id: d for d in (_parse(p) for p in sorted(self.folder.glob("*.md")))}
        if not self.docs:
            raise FileNotFoundError(f"no .md documents in {self.folder}")
        self._tf = {i: Counter(tokens(d.title + " " + d.title + " " + d.body)) for i, d in self.docs.items()}
        self._len = {i: sum(tf.values()) for i, tf in self._tf.items()}
        self._avg = sum(self._len.values()) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def get(self, source_id: str) -> Doc | None:
        key = source_id.strip()
        key = key[len(PREFIX):] if key.startswith(PREFIX) else key
        return self.docs.get(key.removesuffix(".md"))

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        q = tokens(query)
        scored = []
        for i, tf in self._tf.items():
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self._idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / self._avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        out = []
        for s, i in scored[:k]:
            d = self.docs[i]
            out.append({"source_id": d.source_id, "title": d.title, "publisher": d.publisher, "date": d.date,
                        "score": round(s, 3), "snippet": _best_paragraph(d.body, q)[:240]})
        return out


def _best_paragraph(body: str, q: list[str]) -> str:
    paras = [p.strip() for p in body.split("\n\n") if p.strip() and not p.startswith("FICTIONAL")]
    qs = set(q)
    return max(paras, key=lambda p: len(qs & set(tokens(p))), default="")
