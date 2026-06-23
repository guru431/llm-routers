"""Socratic mode: questioner asks, respondent answers, optional moderator
adds a per-round note + final summary.

No critique phase (questions and answers are already asymmetric). No
diversity monitor (only 2 participants).
"""

from __future__ import annotations

from pathlib import Path

import asyncio

from dialogue.engine import _run_turn, write_dump, check_round_failures, maybe_dump
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
    resume: bool = False,
) -> None:
    mark_phase(state, "starting")
    # Single source of truth: the round loop and the summary tag both read
    # state.total_rounds, so sync the rounds param into it at entry.
    state.total_rounds = rounds
    # Carry transport fields so engine helpers can reconstruct calls if needed.
    def _project(cfg: dict, role: str) -> dict:
        base = {"id": cfg["id"], "model": cfg["model"], "position": None, "role": role}
        for k in ("base_url", "env_key", "extra", "min_max_tokens"):
            if k in cfg:
                base[k] = cfg[k]
        return base

    # resume=True (dialogue_continue): participants/moderator already on state.
    if not resume:
        state.participants = [
            _project(questioner_cfg, "questioner"),
            _project(respondent_cfg, "respondent"),
        ]
        state.moderator = (
            {"id": moderator_cfg["id"], "model": moderator_cfg["model"]} if moderator_cfg else None
        )

    # Invariant: moderator-note failures must NOT count toward the failure
    # threshold. check_round_failures only tallies ids present in
    # state.participants, so this holds *only* while the moderator id is
    # disjoint from the participant ids. Enforce it explicitly — a moderator
    # that overlaps a participant would let its note failures corrupt the
    # participants' failure accounting.
    if state.moderator is not None:
        participant_ids = {p["id"] for p in state.participants}
        if state.moderator["id"] in participant_ids:
            raise ValueError(
                f"socratic moderator id {state.moderator['id']!r} overlaps a "
                "participant id; moderator must be distinct so its note failures "
                "are not counted toward the failure threshold"
            )

    start = state.current_round + 1
    for round_n in range(start, state.total_rounds + 1):
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
        # Abort if a participant (questioner/respondent) failed this round — a
        # dead questioner otherwise has the respondent "answering" [error:...]
        # for every remaining round. Moderator-note failures don't count.
        check_round_failures(state, round_n)
        # Mid-run persistence — snapshot after each completed round.
        await maybe_dump(state, DUMP_DIR)

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
    state.dump_path = str(await asyncio.to_thread(write_dump, state, base_dir=DUMP_DIR))
