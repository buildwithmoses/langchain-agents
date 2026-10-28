# How LangChain, LangGraph and LangSmith work: notes verified for these agents

Research date: 28 October 2026. Sources: docs.langchain.com (raw `.md` pages and their embedded OpenAPI sections), the installed packages (`langchain`, `langgraph`, `deepagents` 0.7.32, `langsmith` 0.14.14, `openevals` 0.2.2, `langgraph-cli`), and live runs. Where something came from a live run rather than the docs, it says so.

## The plain-English picture

- **LangChain** gives you models, tools and `create_agent`: a tool-calling loop you extend with **middleware** (hooks before and after the model and tools). Human approval, retries, call limits and guardrails are all middleware.
- **LangGraph** is the runtime underneath: a graph of steps with typed state. A **checkpointer** saves every step so a thread can pause and resume. A **Store** holds memory that outlives a thread. An **interrupt** pauses a run until a person answers.
- **Deep Agents** is LangChain's batteries-included harness on top: planning (a todo tool), a virtual filesystem for notes, and **subagents** that each get their own context.
- **LangSmith** records every run as a trace, holds datasets, runs experiments (`evaluate()`), and has production features: online evaluators, automation rules, Insights, annotation queues.

## Which shape each agent uses, and why

| Agent | Shape | Why |
|---|---|---|
| Support | `create_agent` + middleware | One conversational loop. The built-in approval middleware emits the interrupt format `agent-chat-ui` already renders. |
| Research | Deep Agents (lead + `searcher` + `reader`) | Open-ended: planning, many sources, notes. Subagents keep each page's text out of the lead's context. |
| Contracts | Plain LangGraph pipeline | A fixed sequence. Nothing to plan or delegate, and a planning agent would be a way to skip the verifier. |
| Analyst | Custom LangGraph graph | The checks must run in a fixed order (guard, approval, execute, number check, rewrite). As graph steps, the model can't route around them. |
| Eval kit | Plain Python + LangSmith SDK, one model call to label traces | It's a harness, not an agent. |

## LangSmith features this repo uses, and what it leaves to LangSmith

| LangSmith already does | This repo adds |
|---|---|
| Insights clusters production traces into patterns | The eval kit turns specific traces into a reviewed dataset with checkable rubrics |
| Online evaluators and automation rules | The eval kit writes the rule that copies failing traces into the dataset (dry run by default) |
| `evaluate()` experiments and comparisons | A gate with hard floors in code, so one safety failure can't average away |
| Fleet (no-code agents) | Coded agents for workflows that need guarantees in code |

## Gotchas found while building

- **Over the trace quota, LangSmith drops traces and experiments, not datasets.** Mid-build the workspace hit `429 Monthly unique traces usage limit exceeded` on `/runs/multipart`. Datasets could still be created and read. Every eval now has a local mode.
- **`evaluate(upload_results=False)` still uploads traces.** It runs the target under `tracing_context(enabled="local")`, which LangChain's callback tracer treats as on. Measured: 16 to 25 rate-limit errors per run. Wrapping the agent and judge calls in `tracing_context(enabled=False)` got that to 0.
- **`list_runs` rejects `limit` above 100** ("Limit exceeds maximum allowed value of 100"), and the SDK passes it straight through. Let the iterator paginate and stop yourself. `list_runs` is also deprecated in favor of `client.runs.query()` after 31 January 2027.
- **The SDK has no call for automation rules.** The REST endpoint is `POST /api/v1/runs/rules` (fields `session_id`, `sampling_rate`, `filter`, `add_to_dataset_id`, ...). Filter syntax: `or(eq(status, "error"), and(eq(feedback_key, "user_score"), lt(feedback_score, 0.5)))`.
- **`langgraph dev` loads each graph by file path,** so a relative import (`from .agent import build_graph`) in a graph file fails with "attempted relative import with no known parent package". One failed graph stops the whole server. Use absolute imports in the files `langgraph.json` points at.
- **`langgraph dev` refuses synchronous Store reads.** The support agent crashed on its first served request, though every offline test passed. The fix was async reads, plus a test store that fails the same way.
- **Under `langgraph dev`, don't pass a checkpointer or store.** The platform supplies both. Scripts and evals inject `InMemorySaver` and `InMemoryStore`.
- **Claude 5.5 through OpenRouter rejects forced tool choice.** `with_structured_output` defaults to function calling, so use `method="json_schema"`. Anthropic also rejected one 9-field extraction schema as "compiled grammar is too large", so contract extraction runs as 3 parallel calls of 3 fields.
- **`ChatOpenRouter(timeout=...)` is in milliseconds.** The core sets `180_000` and `max_retries=0`, so there is one retry layer, for transient errors only.
- **OpenRouter web search:** the `plugins: [{"id": "web"}]` form and the `:online` suffix are deprecated. Use the `openrouter:web_search` server tool. Results come back as `url_citation` annotations, and the response cost includes the search fee.

## Replaying a production trace as an eval example

These four things all broke before the eval kit's replays were faithful, and each one is now a test:

1. Pair the reply with the **last** user message in the run, not the first.
2. Take the reply from **this turn only**. A turn that ends on a tool call with no result was paused for approval.
3. Carry the **whole trajectory**: user and assistant text, plus completed tool calls and their results. With text only, the model couldn't see that a detention review had already been filed, and filed it again.
4. Carry **state fields** set earlier in the thread (the shipper `account`), and give the judge the whole conversation, not just the last message.
