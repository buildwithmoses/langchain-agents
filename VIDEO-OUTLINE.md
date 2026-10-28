# Demo video outline (about 3:30)

**Goal:** five POC shapes, one customer, and the moment each gate blocks a bad change.
**Setup:** `.env` filled in, `uv run langgraph dev` running with agent-chat-ui open on `support`, a terminal in the repo, and `examples/` open.

| Time | On screen | Say (roughly) |
|---|---|---|
| 0:00–0:15 | Face cam, README table | "These are the five proofs of concept a LangChain deployed engineer sees most. I built all five for one freight broker." |
| 0:15–0:55 | agent-chat-ui: load status, detention question, "send this detention to billing" → approval card → approve. New thread: late-Friday pickup → agent mentions the 3pm dock close | "Anything with consequences waits for a person. And it remembers the shipper across conversations: it was using that memory half the time until I fixed the prompt." |
| 0:55–1:25 | `research ask` brief, a finding with its quote | "Every claim has a quote, and code checks the quote is really in a page it read. The real catch was numbers the model added that weren't in the quote." |
| 1:25–1:55 | `contracts demo`: the indemnity reversal hidden in 16.4, memo, approve/reject redlines | "The model reads, code applies legal's playbook, and the lawyer approves each redline. When I let the model decide compliance, precision dropped and the gate blocked it." |
| 1:55–2:25 | `analyst ask`, the SQL, then the no-guard eval: 76 phone numbers | "Read-only three ways, PII blocked. With a naive prompt and no guard, 76 phone numbers came out. The gate caught it." |
| 2:25–3:15 | `evalkit mine` → candidates → review log → baseline → regression run, GATE FAILED on paused_when_required | "Evals before the POC ships: real traces become a reviewed dataset. The judge average didn't move when I removed an approval step. The code floor did." |
| 3:15–3:30 | Face cam | "Decisions in code, words from the model, a person approves anything with consequences. Code's linked below." |
