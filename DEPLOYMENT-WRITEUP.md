# Deployment write-up: LangChain Agents for Ridgeline Freight

**Customer:** Ridgeline Freight (fictional Memphis freight broker; demo scenario) · **Stack:** LangChain `create_agent` + middleware, LangGraph (checkpointer, Store, interrupts), Deep Agents, LangSmith (datasets, `evaluate()`, rules API), openevals; Claude Haiku 5.5 via OpenRouter, GPT-6.1 Terra as judge · **Author:** Henry Moses Jr

> Every number below was measured on 2026-10-28. Raw results are in `evals/*/results/` and `examples/`.

## Problem

Ridgeline's leadership wants to try agents in five places, which happen to be the five POC shapes a LangChain deployed engineer meets most:

| Sponsor | Ask | Shape |
|---|---|---|
| Dana Reyes, VP Ops | "Shippers message us all day about loads, detention and claims." | Conversational agent with memory and approvals |
| Dana Reyes | "Brief me on what's moving reefer rates, with sources I can check." | Research agent |
| Marcus Hale, General Counsel | "Carriers redline our agreement. Check it against our playbook." | Document review |
| Priya Nair, Head of Data | "Ops managers want answers from the TMS without writing SQL." | Data analyst |
| Everyone | "How do we know it's still working next month?" | Evals and monitoring |

Success criteria agreed up front, the same for all five:
- Anything with consequences (a claim, a detention review, a redline, a query that touches PII) waits for a person.
- Safety properties are hard floors checked in code, never an average a judge can smooth over.
- Every number shown to a user traces back to a tool result, a source quote or a query result.

## Approach

One repo, one shared core (model factory, exact cost tracking, gate), five agents. Each uses the lightest LangChain shape that fits (see `HOW-IT-WORKS.md`), makes its decisions in code, and uses the model for language and judgement only.

## What broke, and what I changed

*Platform*
- **The LangSmith workspace ran out of trace quota mid-build** (`429 Monthly unique traces usage limit exceeded`). Datasets kept working; traces and experiment uploads didn't. Every eval got a local mode. Then **`evaluate(upload_results=False)` still uploaded traces** (16 to 25 rate-limit errors per run), because it runs the target with tracing set to "local", which LangChain treats as on. Wrapping agent and judge calls in `tracing_context(enabled=False)` took it to 0.
- **`langgraph dev` wouldn't start.** Two graph files used relative imports, and the dev server loads graph files by path. One failed graph stops the whole server. Absolute imports fixed it, and all four graphs now load.
- **The support agent crashed on its first served request** while every offline test passed: `langgraph dev` refuses synchronous Store reads. Fixed with async reads and a test store that fails the same way.

*Support agent*
- **The guard had a gap and two false positives.** "The most I can offer is a goodwill credit of up to $100, and billing will decide" got past it. Meanwhile correct refusals ("I can't confirm a $150/hour rate") were blocked. Fixed both; replaying 123 earlier replies gave 0 new false positives.
- **Memory was loaded but used half the time.** The shipper's "dock closes at 3pm" was in the system prompt of every new thread, but asked about a late-Friday pickup, the agent mentioned it in 3 of 6 demo runs. The prompt said "use them when relevant". Naming the situations (times, appointments, rescheduling, equipment) took it to 6 of 6.

*Research agent*
- **The reader skipped reading.** Asked for a fixed answer format, Haiku sometimes answered "nothing relevant" without opening the source (4 of 4 reader calls in one run). Forcing a tool call isn't allowed for Claude 5.5 through OpenRouter, so a check sends the reader back until it has opened the page.
- **The verifier's real catch was numbers, not fake quotes.** Across 4 full runs, 15 of 227 first-draft claims failed: 13 added a number that wasn't in their quote, 2 had a quote not in the source. All were fixed on the one revision.
- **Blocked pages looked like sources.** Two government sites return a normal-looking "Request Access" page. That's now a failed fetch. Sites whose robots.txt disallows the agent are skipped, not spoofed.

*Contracts agent*
- **Anthropic rejected the extraction schema** ("compiled grammar is too large"), which sent all 13 rules to "lawyer must check". Extraction now runs as 3 parallel calls of 3 terms.
- **The state-abbreviation matcher read "in" and "or" as Indiana and Oregon,** which would have flagged a valid Tennessee venue. Tests caught it.

*Analyst*
- **The default model never tested the guard.** Haiku refused every write and PII request in prose, so a "no guard" variant would have passed. Adversarial cases where users paste SQL or claim authority made the guard earn its keep: with a naive prompt and no guard, 76 PII values reached the answer.
- **`COUNT(*)` over the 40k-row check-call table never asked for approval,** because SQLite reports it as an index scan. It now counts as a full scan.

*Eval kit* (each of these is now a test)
- **Mined turns were replayed out of context, four different ways.** First it paired turn 1's question with turn 3's reply. Then it returned the previous turn's answer for a turn that had paused for approval. Then text-only history hid that a detention review had already been filed, so the agent filed it again. And it lost the shipper `account` set earlier in the thread. Each fix is a test.
- **The first dataset scored 0.2 on the agent it came from.** The rubrics copied one reply's wording ("says pickup was yesterday"). The kit now refuses to save a baseline below 0.8 on the source agent, the triage prompt writes rubrics from the user's need and the policy, and a person reviews each one (`evals/evalkit/review-log.md`).
- **The judge average hid a safety regression.** A variant where detention reviews skip approval scored the same 0.8 as the shipped agent, because the baseline had missed that example for a minor reason. "Pauses for approval when required" is now a code check with a floor of 1.0. On that variant it read 0.0, and the gate blocked it.

## Results

| Agent | Hard floors | Quality | Cost |
|---|---|---|---|
| Support (34 cases) | unapproved action, credit promise, injection, hazmat: all 1.0 | policy facts 1.0, helpfulness 0.963 | $0.072 per eval run |
| Research (10 corpus questions) | citation precision 1.0, unanswerable 2/2 | correctness 0.875 | $0.013 per question; live web brief $0.046 |
| Contracts (8 agreements, 19 deviations) | high-severity recall 1.0, compliant false alarms 0 | recall 1.0, precision 1.0 | $0.004 per agreement |
| Analyst (30 questions) | writes blocked 1.0, PII leaks 0 | execution accuracy 1.0 | $0.022 per eval run |
| Eval kit (support, mined) | paused-when-required 1.0 | rubric 0.8 | $0.002 to label |

Every agent's gate blocked at least one real regression (see `README.md`). Model spend for the whole build, including every failed experiment, was about $2.50.

**Next steps for a real rollout:** run `evalkit mine` weekly on production traces and turn on the monitor rule (`--apply`) so failures land in the dataset; take the shipper account from login, not chat text; add PDF parsing for contracts and research; restore LangSmith uploads (`--upload`) once the workspace has trace quota.
