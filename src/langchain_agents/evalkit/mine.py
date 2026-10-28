"""Trace mining: turn real LangSmith traces into eval-set candidates.

Everything here is code, no model: pull root runs, score each one for signals
that it is worth testing (errors, slow, empty answer, negative feedback, human
interrupts), drop near-duplicate inputs, and take a stratified sample so the
eval set isn't 90% "where is my load".
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

WORD = re.compile(r"[a-z0-9$]+")


@dataclass
class Candidate:
    run_id: str
    trace_id: str
    name: str
    input_text: str
    output_text: str
    inputs: dict
    outputs: dict
    latency_s: float | None
    error: str | None
    tool_evidence: str = ""
    history: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    feedback: dict = field(default_factory=dict)
    start_time: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _last_text(value: Any, roles: tuple = ("ai", "assistant")) -> str:
    """Best-effort text of the last message from `roles` in LangGraph-style
    state. For outputs that means the agent's reply only: an earlier version fell
    back to the user's message when the reply was blank, so empty answers were
    never flagged."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        msgs = value.get("messages")
        if isinstance(msgs, list) and msgs:
            for m in reversed(msgs):
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                kind = (m.get("type") or m.get("role")) if isinstance(m, dict) else getattr(m, "type", "")
                if content and kind in roles:
                    return content if isinstance(content, str) else json.dumps(content)[:2000]
            return ""
        for key in ("question", "input", "query", "text", "answer", "output", "brief", "memo"):
            if isinstance(value.get(key), str):
                return value[key]
        return json.dumps(value, default=str)[:2000]
    return str(value)[:2000]


def _last_human(inputs: dict) -> str:
    """The user turn this run answered. An earlier version took the FIRST human
    message, so on multi-turn threads it paired turn 1's question with turn 3's
    reply, and triage flagged correct answers as off-topic."""
    msgs = (inputs or {}).get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            kind = m.get("type") or m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if kind in ("human", "user") and content:
                return content if isinstance(content, str) else json.dumps(content)
    return _last_text(inputs, roles=("human", "user"))


def _kind(m) -> str:
    return (m.get("type") or m.get("role") or "") if isinstance(m, dict) else getattr(m, "type", "")


def _get(m, k):
    return m.get(k) if isinstance(m, dict) else getattr(m, k, None)


def current_turn(outputs: dict) -> list:
    """Messages after the last human message: this run's own turn."""
    msgs = (outputs or {}).get("messages") or []
    last_h = max((i for i, m in enumerate(msgs) if _kind(m) in ("human", "user")), default=-1)
    return msgs[last_h + 1:]


def turn_reply(outputs: dict) -> tuple[str, bool]:
    """(reply text, paused). The reply comes from THIS turn only. A turn that ends
    on tool calls with no results was paused for human approval; v1 instead
    reached back and returned the previous turn's answer, which made a correct
    pause look like an off-topic reply."""
    turn = current_turn(outputs)
    answered = {_get(m, "tool_call_id") for m in turn if _kind(m) == "tool"}
    for m in reversed(turn):
        if _kind(m) in ("ai", "assistant"):
            content = _get(m, "content")
            if content:
                return (content if isinstance(content, str) else json.dumps(content, default=str)), False
            pending = [(tc if isinstance(tc, dict) else vars(tc)) for tc in (_get(m, "tool_calls") or [])]
            pending = [tc for tc in pending if tc.get("id") not in answered]
            if pending:
                return "[paused for human approval: " + ", ".join(tc.get("name", "?") for tc in pending) + "]", True
    if "__interrupt__" in (outputs or {}):
        return "[paused for human approval]", True
    return "", False


def tool_evidence(outputs: dict, limit: int = 6000) -> str:
    """Tool calls and tool results from this turn, so triage can tell real data
    from made-up data. Without it, a correct status answer built from a TMS lookup
    looked 'fabricated' to the triage model (2 of 6 support traces)."""
    lines = []
    for m in current_turn(outputs):
        kind = _kind(m)
        if kind in ("ai", "assistant"):
            for tc in _get(m, "tool_calls") or []:
                tc = tc if isinstance(tc, dict) else vars(tc)
                lines.append(f"CALL {tc.get('name')}({json.dumps(tc.get('args'), default=str)[:300]})")
        elif kind == "tool":
            content = _get(m, "content")
            # 1,500 chars: at 500, a load record was cut before its ETA field and
            # triage called the agent's (correct) ETA unverifiable.
            lines.append(f"RESULT {_get(m, 'name')}: {(content if isinstance(content, str) else json.dumps(content, default=str))[:1500]}")
    return "\n".join(lines)[:limit]


def thread_history(outputs: dict, limit: int = 24) -> list[dict]:
    """The conversation up to and including this turn's user message, with the
    tool calls and tool results, so a replay sees what actually happened.

    Two earlier versions got this wrong, both caught by the baseline check:
    - no history: "Hi again, anything on RL-48410?" lost the account given earlier;
    - text only: the model couldn't see that a detention review had already been
      filed (the tool call and result were dropped), so it filed it again.
    A tool call left without a result paused for approval; it becomes a short
    assistant note, because a dangling tool call is invalid input."""
    msgs = (outputs or {}).get("messages") or []
    last_h = max((i for i, m in enumerate(msgs) if _kind(m) in ("human", "user")), default=-1)
    prior = msgs[: last_h + 1]
    answered = {_get(m, "tool_call_id") for m in prior if _kind(m) == "tool"}
    hist: list[dict] = []
    kept_calls: set = set()
    for m in prior:
        kind, content = _kind(m), _get(m, "content")
        text = content if isinstance(content, str) else (json.dumps(content) if content else "")
        if kind in ("human", "user"):
            if hist and hist[-1]["role"] == "user":
                hist.append({"role": "assistant", "content": PAUSE_NOTE})
            hist.append({"role": "user", "content": text})
        elif kind in ("ai", "assistant"):
            calls = [(tc if isinstance(tc, dict) else vars(tc)) for tc in (_get(m, "tool_calls") or [])]
            done = [{"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args") or {}}
                    for tc in calls if tc.get("id") in answered]
            if done:
                kept_calls.update(tc["id"] for tc in done)
                hist.append({"role": "assistant", "content": text, "tool_calls": done})
            elif text:
                hist.append({"role": "assistant", "content": text})
            elif calls:
                hist.append({"role": "assistant", "content": PAUSE_NOTE})
        elif kind == "tool" and _get(m, "tool_call_id") in kept_calls:  # orphan results are invalid input
            hist.append({"role": "tool", "content": text[:4000], "tool_call_id": _get(m, "tool_call_id"),
                         "name": _get(m, "name")})
    # Keep whole turns: cut at a user message so no tool result loses its call.
    if len(hist) > limit:
        cut = next((i for i in range(len(hist) - limit, len(hist)) if hist[i]["role"] == "user"), len(hist) - 1)
        hist = hist[cut:]
    return hist


PAUSE_NOTE = "That request has been sent to a team member for approval."


def state_extras(outputs: dict) -> dict:
    """Simple state fields set earlier in the thread (e.g. the shipper `account`
    the support agent receives on turn 1). Replaying a later turn without them
    changes the agent's behaviour, so they ride along with the example inputs."""
    return {k: v for k, v in (outputs or {}).items()
            if k not in ("messages", "__interrupt__") and isinstance(v, (str, int, float, bool))}


def to_candidate(run: Any, feedback: dict | None = None) -> Candidate:
    start, end = getattr(run, "start_time", None), getattr(run, "end_time", None)
    latency = (end - start).total_seconds() if start and end else None
    inputs = getattr(run, "inputs", None) or {}
    outputs = getattr(run, "outputs", None) or {}
    reply, paused = turn_reply(outputs) if isinstance(outputs, dict) and outputs.get("messages") else (_last_text(outputs), False)
    if paused:
        outputs = {**outputs, "__interrupt__": outputs.get("__interrupt__", True)}
    return Candidate(
        run_id=str(run.id), trace_id=str(getattr(run, "trace_id", run.id)), name=getattr(run, "name", ""),
        input_text=_last_human(inputs).strip(), output_text=reply.strip(),
        tool_evidence=tool_evidence(outputs),
        history=thread_history(outputs) if isinstance(outputs, dict) else [],
        extras=state_extras(outputs) if isinstance(outputs, dict) else {},
        inputs=inputs, outputs=outputs, latency_s=latency, error=getattr(run, "error", None),
        feedback=feedback or {}, start_time=start.isoformat() if start else None,
    )


def score_signals(cands: list[Candidate], slow_factor: float = 2.0) -> list[Candidate]:
    lats = [c.latency_s for c in cands if c.latency_s is not None]
    median = statistics.median(lats) if lats else None
    for c in cands:
        s = []
        if c.error:
            s.append("error")
        if not c.output_text:
            s.append("empty_output")
        if median and c.latency_s and c.latency_s > slow_factor * median:
            s.append("slow")
        if "__interrupt__" in json.dumps(c.outputs, default=str):
            s.append("human_interrupt")
        if any(isinstance(v, (int, float)) and v <= 0 for v in c.feedback.values()):
            s.append("negative_feedback")
        c.signals = s
    return cands


def _tokens(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def dedupe(cands: list[Candidate], threshold: float = 0.8) -> list[Candidate]:
    """Drop inputs that are near-copies (Jaccard over words) of one already kept.
    Prefer keeping the run with more signals."""
    kept: list[Candidate] = []
    seen_exact: set[str] = set()
    for c in sorted(cands, key=lambda x: -len(x.signals)):
        key = hashlib.sha1(" ".join(sorted(_tokens(c.input_text))).encode()).hexdigest()
        if not c.input_text or key in seen_exact:
            continue
        t = _tokens(c.input_text)
        if any(len(t & _tokens(k.input_text)) / max(1, len(t | _tokens(k.input_text))) >= threshold for k in kept):
            continue
        seen_exact.add(key)
        kept.append(c)
    return kept


def stratified_sample(cands: list[Candidate], n: int) -> list[Candidate]:
    """Every run with a signal first (they are why we mine), then fill with clean
    runs spread across time so the set also covers normal traffic."""
    flagged = [c for c in cands if c.signals]
    clean = sorted((c for c in cands if not c.signals), key=lambda c: c.start_time or "")
    out = flagged[:n]
    room = n - len(out)
    if room > 0 and clean:
        step = max(1, len(clean) // room)
        out += clean[::step][:room]
    return out


def fetch_runs(project: str, *, days: float = 7, name: str | None = None, filter: str | None = None,
               limit: int = 300, client=None) -> list[Candidate]:
    from langsmith import Client

    client = client or Client()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    # The API rejects limit > 100 per query ("Limit exceeds maximum allowed value of
    # 100"), and the SDK passes `limit` straight through. Let it paginate, stop ourselves.
    from itertools import islice

    runs = list(islice(client.list_runs(project_name=project, is_root=True, start_time=since, filter=filter), limit))
    if name:
        runs = [r for r in runs if r.name == name]
    fb: dict[str, dict] = {}
    ids = [r.id for r in runs]
    for i in range(0, len(ids), 100):
        for f in client.list_feedback(run_ids=ids[i:i + 100]):
            if f.score is not None:
                fb.setdefault(str(f.run_id), {})[f.key] = f.score
    return [to_candidate(r, fb.get(str(r.id))) for r in runs]


def is_resume(c: Candidate) -> bool:
    """Runs started by resuming an interrupt (approve / edit / reject) carry a
    Command, not a user message. They aren't eval inputs on their own."""
    inp = c.inputs or {}
    return "resume" in inp or (isinstance(inp.get("input"), dict) and "resume" in inp["input"])


def mine(project: str, *, n: int = 30, **kw) -> tuple[list[Candidate], dict]:
    pulled = fetch_runs(project, **kw)
    raw = [c for c in pulled if not is_resume(c)]
    scored = score_signals(raw)
    unique = dedupe(scored)
    sample = stratified_sample(unique, n)
    stats = {"runs_pulled": len(pulled), "resume_runs_skipped": len(pulled) - len(raw), "after_dedupe": len(unique), "sampled": len(sample),
             "signals": {s: sum(s in c.signals for c in sample) for s in
                         ("error", "empty_output", "slow", "human_interrupt", "negative_feedback")}}
    return sample, stats


def write_jsonl(cands: Iterable[Candidate], path) -> None:
    with open(path, "w") as f:
        for c in cands:
            f.write(json.dumps(c.to_json(), default=str) + "\n")
