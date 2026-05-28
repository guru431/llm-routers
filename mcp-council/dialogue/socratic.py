"""Socratic mode: questioner asks, respondent answers, optional moderator
adds a per-round note + final summary.

No critique phase (questions and answers are already asymmetric). No
diversity monitor (only 2 participants).
"""

from __future__ import annotations

from pathlib import Path

from dialogue.engine import _run_turn, write_dump
from dialogue.prompts import (
    render_socratic_questioner_prompt,
    render_socratic_respondent_prompt,
    render_summary_prompt,
)
from dialogue.state import DialogueState, mark_phase

DUMP_DIR = Path(__file__).parent.parent / "logs" / "dialogues"

MODERATOR_NOTE_PROMPT_TEMPLATE = (
    "You are the moderator for a Socratic dialogue on '{topic}'.\n"
    "Round {round_n} just finished. Below is the dialogue so far.\n"
    "Write a SHORT note (2-3 sentences max) covering:\n"
    "- What was clarified in this round.\n"
    "- What remains unclear or unaddressed.\n"
    "Output only the note, no preamble.\n\n"
    "=== TRANSCRIPT ===\n{transcript}\n"
)


def _format_transcript(history: list[dict]) -> str:
    lines: list[str] = []
    for h in history:
        marker = h["phase"]
        lines.append(f"[{h['id']}] ({marker}): {h['text']}")
    return "\n".join(lines)


async def run_socratic(
    *,
    state: DialogueState,
    topic: str,
    questioner_cfg: dict,
    respondent_cfg: dict,
    moderator_cfg: dict | None,
    rounds: int,
    max_tokens: int,
    web_search: bool,
    files_section: str | None,
) -> None:
    mark_phase(state, "starting")
    # Carry transport fields so engine helpers can reconstruct calls if needed.
    def _project(cfg: dict, role: str) -> dict:
        base = {"id": cfg["id"], "model": cfg["model"], "position": None, "role": role}
        for k in ("base_url", "env_key", "extra", "min_max_tokens"):
            if k in cfg:
                base[k] = cfg[k]
        return base

    state.participants = [
        _project(questioner_cfg, "questioner"),
        _project(respondent_cfg, "respondent"),
    ]
    state.moderator = (
        {"id": moderator_cfg["id"], "model": moderator_cfg["model"]} if moderator_cfg else None
    )

    start = state.current_round + 1
    for round_n in range(start, rounds + 1):
        # --- question phase ---
        mark_phase(state, f"round_{round_n}_question")
        q_prompt = render_socratic_questioner_prompt(
            topic=topic, history=state.history, round_n=round_n, files_section=files_section,
        )
        q_result = await _run_turn(
            cfg=questioner_cfg, prompt=q_prompt, max_tokens=max_tokens, web_search=web_search,
        )
        state.history.append({
            "round": round_n, "phase": "question", "id": questioner_cfg["id"],
            "text": q_result.text if q_result.status == "ok" else f"[error: {q_result.error}]",
            "latency_ms": q_result.latency_ms, "status": q_result.status,
        })

        # --- answer phase ---
        mark_phase(state, f"round_{round_n}_answer")
        a_prompt = render_socratic_respondent_prompt(
            topic=topic, history=state.history, round_n=round_n, files_section=files_section,
        )
        a_result = await _run_turn(
            cfg=respondent_cfg, prompt=a_prompt, max_tokens=max_tokens, web_search=web_search,
        )
        state.history.append({
            "round": round_n, "phase": "answer", "id": respondent_cfg["id"],
            "text": a_result.text if a_result.status == "ok" else f"[error: {a_result.error}]",
            "latency_ms": a_result.latency_ms, "status": a_result.status,
        })

        # --- optional moderator note ---
        if moderator_cfg:
            note_prompt = MODERATOR_NOTE_PROMPT_TEMPLATE.format(
                topic=topic, round_n=round_n, transcript=_format_transcript(state.history),
            )
            n_result = await _run_turn(
                cfg=moderator_cfg, prompt=note_prompt, max_tokens=max_tokens, web_search=False,
            )
            state.history.append({
                "round": round_n, "phase": "moderator_note", "id": moderator_cfg["id"],
                "text": n_result.text if n_result.status == "ok" else f"[note failed: {n_result.error}]",
                "latency_ms": n_result.latency_ms, "status": n_result.status,
            })

        state.current_round = round_n

    # --- final summary (only if moderator present) ---
    if moderator_cfg:
        mark_phase(state, "summarizing")
        summary_prompt = render_summary_prompt(topic=topic, history=state.history, mode="socratic")
        s_result = await _run_turn(
            cfg=moderator_cfg, prompt=summary_prompt, max_tokens=max_tokens, web_search=False,
        )
        state.history.append({
            "round": state.total_rounds, "phase": "summary", "id": moderator_cfg["id"],
            "text": (
                s_result.text if s_result.status == "ok"
                else f"[summary failed: {s_result.error}]"
            ),
            "latency_ms": s_result.latency_ms, "status": s_result.status,
        })

    from dialogue.render import format_dialogue_markdown
    state.result_markdown = format_dialogue_markdown(state, topic)
    mark_phase(state, "done")
    state.dump_path = str(write_dump(state, base_dir=DUMP_DIR))
