"""Structured outputs. Every field is required with no default: smaller models
silently omit optional fields (seen on the clinic build: citations went missing
until the field was made required)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    claim: str = Field(description="One factual statement, in plain words, that the quote supports.")
    source_id: str = Field(description="The exact source_id of a source the reader read (a URL or corpus:doc-id).")
    quote: str = Field(description="A passage copied character-for-character from that source's text that supports "
                                   "the claim. One to three full sentences. Never paraphrase. Never a search snippet.")


class Brief(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    answerable: bool = Field(description="True only if the sources read actually answer the question.")
    answer: str = Field(description="Answer first: 2-5 sentences for a VP of Operations. Every sentence must end "
                                    "with the numbers of the claims that back it, like [1] or [2][3]. If the "
                                    "sources don't answer the question, say so plainly.")
    claims: list[Claim] = Field(description="The findings, each tied to one quote. Empty if nothing relevant was found.")
    gaps: list[str] = Field(description="Parts of the question the sources did not answer. Empty list if none.")


class QuoteNote(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    quote: str = Field(description="Exact text copied from the source, 1-3 full sentences.")
    why: str = Field(description="What this quote shows, in a few words.")


class ReaderNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema (OpenAI requires it)
    source_id: str = Field(description="The source_id exactly as passed to read_source.")
    title: str = Field(description="The source's title.")
    relevant: bool = Field(description="Whether the source has anything relevant to the question.")
    quotes: list[QuoteNote] = Field(description="Up to 8 verbatim quotes relevant to the question. Empty if none.")
