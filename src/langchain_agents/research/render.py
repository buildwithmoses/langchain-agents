"""brief.md and claims.json."""

from __future__ import annotations

import json
from pathlib import Path


def _source_label(source_id: str, sources: dict) -> str:
    src = sources.get(source_id) or {}
    title = src.get("title") or ""
    if source_id.startswith("corpus:"):
        return f"{title} (`{source_id}`, fictional demo document)" if title else f"`{source_id}`"
    return f"[{title or source_id}]({source_id})"


def render_brief(question: str, final: dict, sources: dict, meta: dict) -> str:
    out = [f"# Research brief\n", f"**Question:** {question}\n",
           f"*Prepared for Dana Reyes, VP Operations, Ridgeline Freight. Mode: {meta.get('mode', '?')}. "
           f"Every finding below was checked in code: its quote appears verbatim in a source read during this run.*\n",
           "## Answer\n", final["answer"] + "\n", "## Findings\n"]
    if final["claims"]:
        for c in final["claims"]:
            out.append(f"{c['n']}. {c['claim']}\n   > \"{c['quote']}\"\n   > Source: {_source_label(c['source_id'], sources)}\n")
    else:
        out.append("_No verified findings._\n")
    out.append("## What we couldn't verify\n")
    lines = [f"- Gap: {g}" for g in final.get("gaps") or []]
    for d in final.get("dropped") or []:
        lines.append(f"- Dropped claim ({d['reason']}): {d['claim']} (cited `{d['source_id']}`)")
    if final.get("answer_sentences_dropped"):
        lines.append(f"- {final['answer_sentences_dropped']} answer sentence(s) removed because they cited no verified claim.")
    out.append("\n".join(lines) + "\n" if lines else "_Nothing was dropped and no gaps were reported._\n")
    stats = meta.get("stats", {})
    out.append("## Run\n")
    out.append(f"- Claims in the agent's first draft: {stats.get('raw_claims', 0)}, of which verified: "
               f"{stats.get('raw_verified', 0)}\n- Revision rounds: {stats.get('revisions', 0)}\n"
               f"- Final: {stats.get('final_claims', 0)} verified, {stats.get('final_dropped', 0)} dropped\n"
               f"- Sources read: {stats.get('sources_read', 0)} (failed: {stats.get('sources_failed', 0)}), searches: "
               f"{stats.get('searches', 0)}\n- Cost: ${meta.get('cost_usd', 0):.4f} · latency {meta.get('latency_s', 0):.1f}s · "
               f"model {meta.get('model', '?')}\n")
    return "\n".join(out)


def write_outputs(out_dir: Path, question: str, state: dict, meta: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    final, sources = state["final"], state.get("sources") or {}
    paths = {"brief": out_dir / "brief.md", "claims": out_dir / "claims.json", "run": out_dir / "run.json"}
    paths["brief"].write_text(render_brief(question, final, sources, meta), encoding="utf-8")
    paths["claims"].write_text(json.dumps({
        "question": question,
        "final": final,
        "raw_brief": state.get("raw_brief"),
        "rounds": state.get("rounds") or [],
    }, indent=2), encoding="utf-8")
    paths["run"].write_text(json.dumps({
        **meta,
        "sources": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in sources.items()},
        "searches": state.get("searches") or {},
    }, indent=2), encoding="utf-8")
    return paths
