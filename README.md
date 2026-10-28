# LangChain Agents: five proof-of-concept shapes, one customer

The five POC shapes a LangChain deployed engineer sees most, built for one customer on **LangChain, LangGraph and Deep Agents**, each with offline tests, a live eval and a gate that blocks regressions. Models run through OpenRouter (Claude Haiku 5.5 for the agents, GPT-6.1 Terra as an independent judge).

| Agent | Shape | What it proves |
|---|---|---|
| **Support** | `create_agent` + middleware, short- and long-term memory | A human approves claims, detention reviews and escalations. A code guard blocks credit promises and wrong policy numbers. Works with `langgraph dev` and `agent-chat-ui`. |
| **Research** | Deep Agents: lead + searcher + reader | Every claim's quote is found word for word in a page this run actually read, or the claim is dropped. |
| **Contracts** | LangGraph pipeline | The model extracts; code applies legal's playbook. Quotes verified; the lawyer approves each redline. |
| **Analyst** | Custom LangGraph graph over SQLite | Read-only in three independent layers, PII blocked, approval for PII or full scans, and every number in the answer checked against the result. |
| **Eval kit** | LangSmith SDK + one labelling call | Mines real traces, replays them faithfully, a person reviews the rubric, experiments run, and a gate with hard floors blocks the change. Writes the LangSmith rule that feeds future failures back into the dataset. |

**The pattern in all five: decisions in code, words from the model, and a human approves anything with consequences.**

> **Fictional customer.** Ridgeline Freight (a Memphis freight broker), its shippers, loads, contracts, database and documents are made up (see `fixtures/ridgeline/COMPANY.md`). Every score, cost and latency below was measured on 2026-10-28. The research agent's live run used the real web.

## Results

| Agent | Eval | Hard floors | Quality | Cost | Regression the gate blocked |
|---|---|---|---|---|---|
| Support | 29 single-turn + 5 multi-turn cases | no unapproved action, no credit promise, injection resisted, hazmat refused: all **1.0** | policy facts 1.0, helpfulness 0.963 | $0.072 per full eval | a "goodwill credit" prompt with the guard off (credit floor 0.882); escalations without approval (0.824) |
| Research | 10 questions on a fixed corpus | final-brief citation precision **1.0**, unanswerable questions refused **2/2** | answer correctness 0.875 | $0.013 per question | a regressed result file (exit 1) |
| Contracts | 8 agreements, 19 planted deviations | high-severity recall **1.0** (6/6), false alarms on the compliant agreement **0** | recall 1.0, precision 1.0 | $0.004 per agreement | the model deciding compliance instead of code (precision 0.905, one unverifiable quote) |
| Analyst | 30 questions incl. adversarial | write attempts blocked **1.0**, PII leaks **0** | execution accuracy 1.0 (21/21) | $0.022 per full eval | a naive prompt with the SQL guard off (76 PII values shown, leak-free 0.625) |
| Eval kit | dataset mined from the support agent's own traces | pauses for approval when the rubric requires it **1.0** | rubric 0.8 | $0.002 to label 5 traces | detention reviews without approval: the judge average was unchanged at 0.8, the code floor dropped to 0.0 and blocked it |

The research agent's live web run answered a current FMCSA question with 9 verified claims from 2 Federal Register documents for $0.046 (`examples/research/live-fmcsa-financial-responsibility/`).

## Quickstart

```bash
uv sync
cp .env.example .env                  # OPENROUTER_API_KEY, LANGSMITH_API_KEY
uv run pytest                         # 439 tests, offline, no keys needed

uv run langchain-agents support demo                  # lookup, policy, approval, memory across threads
uv run langchain-agents support chat --account ACCT-1001
uv run langchain-agents research ask "..." --corpus   # or live web without --corpus
uv run langchain-agents contracts demo
uv run langchain-agents analyst ask "Which lane lost money last month?"
uv run langchain-agents evalkit mine --project langchain-agents --name ridgeline-support
uv run langgraph dev                                  # all four graphs; point agent-chat-ui at http://localhost:2024
```

Each agent has `evals/<agent>/run_evals.py` and `evals/<agent>/gate.py`. Evals run locally by default (no LangSmith uploads); pass `--upload` where supported once the workspace has trace quota.

## Repo layout

```
src/langchain_agents/
  core/        model factory (OpenRouter), exact cost per run, shared gate
  support/     create_agent + approval middleware + code guard + Store memory
  research/    Deep Agents lead/searcher/reader, fetcher, citation verifier
  contracts/   sectioning, extraction, quote verifier, playbook rules, lawyer approval
  analyst/     SQL guard, read-only DB, PII policy, approval, number check
  evalkit/     mine, triage, review, dataset, experiment, monitor, gate
fixtures/      Ridgeline company brief, TMS data, contracts, research corpus, DB generator
evals/         one runner, dataset, baseline and gate per agent
examples/      real output from each agent
tests/         439 offline tests
HOW-IT-WORKS.md        LangChain/LangSmith facts the agents rely on, verified
DEPLOYMENT-WRITEUP.md  problem, approach, what broke, results
```

Sibling build: [clinic-intake-agent](https://github.com/buildwithmoses/clinic-intake-agent) (Deep Agents + LangSmith evals for a clinic).
