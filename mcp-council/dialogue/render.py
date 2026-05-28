"""Render DialogueState to a markdown brief for the MCP consumer."""

from __future__ import annotations

from dialogue.state import DialogueState


def _format_setup_debate(state: DialogueState) -> list[str]:
    lines = []
    for i, p in enumerate(state.participants):
        letter = chr(ord("A") + i)
        pos = p.get("position") or "(no position)"
        lines.append(f"- Participant {letter} ({p['model']}): defending \"{pos}\"")
    if state.moderator:
        lines.append(f"- Moderator: {state.moderator['model']} ({state.moderator['id']})")
    lines.append(f"- Rounds: {state.total_rounds}")
    return lines


def _format_setup_panel(state: DialogueState) -> list[str]:
    lines = []
    for i, p in enumerate(state.participants):
        letter = chr(ord("A") + i)
        role = p.get("role") or "(no role)"
        lines.append(f"- Participant {letter} ({p['model']}): role={role}")
    if state.moderator:
        lines.append(f"- Monitor: {state.moderator['model']} ({state.moderator['id']})")
    lines.append(f"- Rounds: {state.total_rounds}")
    return lines


def _format_setup_socratic(state: DialogueState) -> list[str]:
    lines = []
    for p in state.participants:
        lines.append(f"- {p['role']}: {p['model']} ({p['id']})")
    if state.moderator:
        lines.append(f"- Moderator: {state.moderator['model']} ({state.moderator['id']})")
    lines.append(f"- Rounds: {state.total_rounds}")
    return lines


def _format_round(state: DialogueState, round_n: int) -> list[str]:
    lines = [f"## Round {round_n}"]
    round_items = [h for h in state.history if h["round"] == round_n]

    critiques = [h for h in round_items if h["phase"] == "critique"]
    responses = [h for h in round_items if h["phase"] == "response"]
    questions = [h for h in round_items if h["phase"] == "question"]
    answers = [h for h in round_items if h["phase"] == "answer"]
    notes = [h for h in round_items if h["phase"] == "moderator_note"]
    reprompts = [h for h in round_items if h["phase"] == "reprompt"]
    directives = [h for h in round_items if h["phase"] == "directive"]

    if directives:
        for d in directives:
            lines.append(f"### User directive (via dialogue_continue) — {d['id']}")
            lines.append(d["text"])
            lines.append("")
    if critiques:
        lines.append("### Critique")
        for c in critiques:
            lines.append(f"- {c['id']}: {c['text']}")
        lines.append("")
    if responses:
        lines.append("### Response")
        for r in responses:
            secs = (r.get("latency_ms") or 0) / 1000
            lines.append(f"#### {r['id']} — {secs:.0f}s")
            lines.append(r["text"])
            lines.append("")
    if reprompts:
        lines.append("### Re-prompt (diversity-monitor triggered)")
        for r in reprompts:
            lines.append(f"- {r['id']}: {r['text']}")
        lines.append("")
    if questions:
        for q in questions:
            secs = (q.get("latency_ms") or 0) / 1000
            lines.append(f"### Question — {q['id']} ({secs:.0f}s)")
            lines.append(q["text"])
            lines.append("")
    if answers:
        for a in answers:
            secs = (a.get("latency_ms") or 0) / 1000
            lines.append(f"### Answer — {a['id']} ({secs:.0f}s)")
            lines.append(a["text"])
            lines.append("")
    if notes:
        for n in notes:
            lines.append(f"### Moderator note — {n['id']}")
            lines.append(n["text"])
            lines.append("")
    return lines


def format_dialogue_markdown(state: DialogueState, topic: str) -> str:
    lines: list[str] = []
    lines.append(f"# Dialogue session {state.session_id} (mode: {state.mode})")
    lines.append("")
    lines.append("## Topic")
    lines.append(topic)
    lines.append("")
    lines.append("## Setup")
    if state.mode == "debate":
        lines.extend(_format_setup_debate(state))
    elif state.mode == "panel":
        lines.extend(_format_setup_panel(state))
    else:
        lines.extend(_format_setup_socratic(state))
    lines.append("")

    rounds_seen = sorted({h["round"] for h in state.history if h["phase"] != "summary"})
    for rn in rounds_seen:
        lines.extend(_format_round(state, rn))

    summary_entries = [h for h in state.history if h["phase"] == "summary"]
    if summary_entries:
        lines.append("## Final Summary")
        lines.append(summary_entries[-1]["text"])
        lines.append("")

    if state.diversity_scores or state.devils_advocates:
        lines.append("## Notes")
        if state.diversity_scores:
            lines.append(f"- diversity scores per round: {state.diversity_scores}")
        if state.devils_advocates:
            lines.append(f"- devil's advocate rotation: {state.devils_advocates}")
        lines.append("")

    lines.append("---")
    if state.dump_path:
        lines.append(f"Full dump: {state.dump_path}")
    return "\n".join(lines)
