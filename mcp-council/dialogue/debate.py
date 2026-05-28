"""Debate mode: 2 (or more) participants defend opposing positions assigned
by a moderator.

Flow:
  1. Moderator splits the question into N opposing positions (JSON array).
  2. Positions are assigned to participants in declared order.
  3. run_dialogue executes N rounds (critique + response, critique auto-skipped
     in round 1 because there are no prior responses).
  4. Moderator writes a final summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from dialogue.engine import _run_turn, run_dialogue, write_dump
from dialogue.prompts import (
    render_position_split_prompt,
    render_summary_prompt,
)
from dialogue.state import DialogueState, mark_phase

DUMP_DIR = Path(__file__).parent.parent / "logs" / "dialogues"


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


async def _call_moderator(cfg: dict, prompt: str, max_tokens: int, web_search: bool) -> str:
    """Isolated wrapper for tests to monkeypatch."""
    from dialogue.engine import _call_model  # late import to allow monkeypatching
    return await _call_model(cfg, prompt, max_tokens, web_search)


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


async def generate_positions(
    *,
    moderator_cfg: dict,
    question: str,
    n: int,
) -> list[str]:
    """Ask the moderator to split the question into exactly n opposing positions.

    Raises RuntimeError on JSON parse failure or wrong array length.
    """
    prompt = render_position_split_prompt(question=question, n=n)
    raw = await _call_moderator(moderator_cfg, prompt, 512, False)
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"moderator returned non-JSON for position split: {e}; got: {cleaned[:200]}"
        ) from e
    if not isinstance(parsed, list):
        raise RuntimeError(
            f"moderator returned non-array for position split: {type(parsed).__name__}"
        )
    if len(parsed) != n:
        raise RuntimeError(f"expected {n} positions, got {len(parsed)}: {parsed}")
    if not all(isinstance(p, str) for p in parsed):
        raise RuntimeError(f"all positions must be strings, got: {parsed}")
    return parsed


def cfg_to_participant(cfg: dict) -> dict:
    """Project an engine cfg dict into a state.participants entry."""
    base = {"id": cfg["id"], "model": cfg["model"], "position": None, "role": None}
    # Carry transport fields so engine._participant_to_cfg can reconstruct calls.
    for k in ("base_url", "env_key", "extra", "min_max_tokens"):
        if k in cfg:
            base[k] = cfg[k]
    return base


async def run_debate(
    *,
    state: DialogueState,
    question: str,
    participant_cfgs: list[dict],
    moderator_cfg: dict,
    rounds: int,
    max_tokens: int,
    web_search: bool,
    files_section: str | None,
) -> None:
    """Orchestrate a full debate session and mark phase=done at the end."""
    mark_phase(state, "starting")
    positions = await generate_positions(
        moderator_cfg=moderator_cfg,
        question=question,
        n=len(participant_cfgs),
    )

    state.participants = [
        {**cfg_to_participant(c), "position": p}
        for c, p in zip(participant_cfgs, positions)
    ]
    state.moderator = {"id": moderator_cfg["id"], "model": moderator_cfg["model"]}

    role_descriptors = {
        p["id"]: (
            f"You are participant {p['id']}. You MUST defend this position "
            f"throughout the debate: \"{p['position']}\""
        )
        for p in state.participants
    }
    await run_dialogue(
        state=state,
        topic=question,
        role_descriptors=role_descriptors,
        max_tokens=max_tokens,
        web_search=web_search,
        files_section=files_section,
        do_critique=True,
        per_round_hook=None,
        start_round=state.current_round + 1,
    )

    mark_phase(state, "summarizing")
    summary_prompt = render_summary_prompt(topic=question, history=state.history, mode="debate")
    summary_result = await _run_turn(
        cfg=moderator_cfg,
        prompt=summary_prompt,
        max_tokens=max_tokens,
        web_search=False,
    )
    state.history.append({
        "round": state.total_rounds,
        "phase": "summary",
        "id": moderator_cfg["id"],
        "text": (
            summary_result.text if summary_result.status == "ok"
            else f"[summary failed: {summary_result.error}]"
        ),
        "latency_ms": summary_result.latency_ms,
        "status": summary_result.status,
    })

    from dialogue.render import format_dialogue_markdown
    state.result_markdown = format_dialogue_markdown(state, question)
    # Mark done BEFORE write_dump so the persisted JSON reflects the terminal
    # phase (otherwise dumps freeze at "summarizing" and post-mortem inspection
    # makes it look like the run never finished).
    mark_phase(state, "done")
    state.dump_path = str(write_dump(state, base_dir=DUMP_DIR))
