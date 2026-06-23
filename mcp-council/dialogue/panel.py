"""Panel mode: 4-6 participants in free discussion with anti-convergence.

Mechanisms:
- Devil's advocate rotation: each round, one participant gets a system rule
  to obligatorily disagree with the emerging consensus.
- Diversity monitor: after the response phase, a cheap model rates how
  similar the responses are (0-10) and lists 'agreers'. If score > threshold,
  the agreers receive a re-prompt asking them to break from the consensus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import asyncio

from dialogue.engine import (
    _run_turn, _run_phase, run_round, write_dump,
    check_round_failures, maybe_dump,
)
from dialogue.prompts import (
    render_diversity_monitor_prompt,
    render_summary_prompt,
    render_response_prompt,
)
from dialogue.state import DialogueState, mark_phase

DUMP_DIR = Path(__file__).parent.parent / "logs" / "dialogues"

DEVILS_ADVOCATE_RULE = (
    "You are the devil's advocate in this round. You MUST argue against the "
    "emerging consensus, even if you privately agree. Find at least one "
    "substantive objection nobody else raised, and defend it specifically."
)

REPROMPT_RULE = (
    "You agreed too closely with another participant. You MUST now state a "
    "specific point where you actually differ from {others}, even if it is "
    "minor. If you genuinely cannot find any difference, say so explicitly "
    "and explain why the consensus is unavoidable."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def devils_advocate_for_round(participants: list[dict], round_n: int) -> str:
    """Return the participant id assigned as devil's advocate for round_n
    (1-indexed). Rotates round-robin."""
    return participants[(round_n - 1) % len(participants)]["id"]


async def _call_monitor(cfg: dict, prompt: str, max_tokens: int, web_search: bool) -> str:
    """Isolated wrapper for tests to monkeypatch."""
    from dialogue.engine import _call_model
    return await _call_model(cfg, prompt, max_tokens, web_search)


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


async def run_diversity_check(
    *,
    monitor_cfg: dict,
    responses: dict[str, str],
) -> tuple[int, list[str]]:
    """Ask monitor to rate response similarity. Returns (score, agreers).

    On any parsing failure returns (0, []) — neutral, no re-prompt.
    """
    prompt = render_diversity_monitor_prompt(responses=responses)
    try:
        raw = await _call_monitor(monitor_cfg, prompt, 256, False)
    except Exception:
        return 0, []
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
        score = int(parsed.get("score", 0))
        agreers = parsed.get("agreers") or []
        if not isinstance(agreers, list):
            agreers = []
        agreers = [str(a) for a in agreers]
        return max(0, min(10, score)), agreers
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return 0, []


async def _maybe_reprompt(
    *,
    state: DialogueState,
    round_n: int,
    participant_cfgs: list[dict],
    score: int,
    agreers: list[str],
    threshold: int,
    topic: str,
    max_tokens: int,
    files_section: str | None,
) -> None:
    """If score > threshold and there are agreers, re-prompt them and append to history."""
    if score <= threshold or not agreers:
        return
    target_cfgs = [c for c in participant_cfgs if c["id"] in set(agreers)]
    if not target_cfgs:
        return

    def builder(cfg: dict) -> str:
        # Exclude the target's own id from `others` — telling them "you
        # agreed too closely with A, B, <yourself>" is confusing.
        others = ", ".join(a for a in agreers if a != cfg["id"])
        rule = REPROMPT_RULE.format(others=others) if others else REPROMPT_RULE.format(others="the other participants")
        return render_response_prompt(
            topic=topic,
            role_descriptor=f"You are participant {cfg['id']} in a panel discussion.",
            history=state.history,
            round_n=round_n,
            files_section=files_section,
            anti_agreement_rule=rule,
        )

    results = await _run_phase(
        participants=target_cfgs,
        prompt_builder=builder,
        max_tokens=max_tokens,
        web_search=False,
    )
    for r in results:
        state.history.append({
            "round": round_n,
            "phase": "reprompt",
            "id": r.id,
            "text": r.text if r.status == "ok" else f"[reprompt error: {r.error}]",
            "latency_ms": r.latency_ms,
            "status": r.status,
        })


def _cfg_to_participant(cfg: dict, role: str | None = None) -> dict:
    base = {"id": cfg["id"], "model": cfg["model"], "position": None, "role": role}
    for k in ("base_url", "env_key", "extra", "min_max_tokens"):
        if k in cfg:
            base[k] = cfg[k]
    return base


async def run_panel(
    *,
    state: DialogueState,
    question: str,
    participant_cfgs: list[dict],
    monitor_cfg: dict,
    rounds: int,
    max_tokens: int,
    web_search: bool,
    files_section: str | None,
    roles: list[str] | None,
    diversity_monitor: bool,
    diversity_threshold: int,
    devils_advocate_rotation: bool,
    resume: bool = False,
) -> None:
    """Orchestrate a panel session.

    resume=True (dialogue_continue): participants/moderator already live on
    state — skip seeding and pick up from state.current_round + 1.
    """
    mark_phase(state, "starting")
    # Single source of truth: the round loop and the summary tag both read
    # state.total_rounds, so sync the rounds param into it at entry.
    state.total_rounds = rounds
    if not resume:
        state.participants = [
            _cfg_to_participant(c, role=(roles[i] if roles else None))
            for i, c in enumerate(participant_cfgs)
        ]
        state.moderator = {"id": monitor_cfg["id"], "model": monitor_cfg["model"]}

    def role_descriptor(p: dict) -> str:
        if p.get("role"):
            return f"You are participant {p['id']} playing the role: {p['role']}. Stay in character."
        return (
            f"You are participant {p['id']} in a multi-model panel discussion. "
            "Bring your distinct perspective."
        )

    role_descriptors = {p["id"]: role_descriptor(p) for p in state.participants}

    start = state.current_round + 1
    for round_n in range(start, state.total_rounds + 1):
        da_id = (
            devils_advocate_for_round(state.participants, round_n)
            if devils_advocate_rotation else None
        )
        rules: dict[str, str] | None = None
        if devils_advocate_rotation:
            rules = {da_id: DEVILS_ADVOCATE_RULE}

        await run_round(
            state=state,
            round_n=round_n,
            topic=question,
            role_descriptors=role_descriptors,
            max_tokens=max_tokens,
            web_search=web_search,
            anti_agreement_rules=rules,
            files_section=files_section,
            do_critique=True,
        )

        # Same abort guard debate gets via run_dialogue: a dead provider must
        # not drag a 7-model panel through all rounds emitting [error:...].
        check_round_failures(state, round_n)

        if devils_advocate_rotation:
            state.devils_advocates.append(da_id)
        if diversity_monitor:
            responses_this_round = {
                h["id"]: h["text"]
                for h in state.history
                if h["round"] == round_n and h["phase"] == "response" and h.get("status") == "ok"
            }
            score, agreers = await run_diversity_check(
                monitor_cfg=monitor_cfg, responses=responses_this_round,
            )
            state.diversity_scores.append(score)
            await _maybe_reprompt(
                state=state,
                round_n=round_n,
                participant_cfgs=participant_cfgs,
                score=score,
                agreers=agreers,
                threshold=diversity_threshold,
                topic=question,
                max_tokens=max_tokens,
                files_section=files_section,
            )
            # Re-check after reprompt: a provider that only fails on the heavy
            # reprompt call adds an error entry to this same round that the
            # pre-reprompt check above could not have seen.
            check_round_failures(state, round_n)

        # Mid-run persistence — snapshot after each completed round.
        await maybe_dump(state, DUMP_DIR)

    mark_phase(state, "summarizing")
    summary_prompt = render_summary_prompt(topic=question, history=state.history, mode="panel")
    summary_result = await _run_turn(
        cfg=monitor_cfg, prompt=summary_prompt, max_tokens=max_tokens, web_search=False,
    )
    state.history.append({
        "round": state.total_rounds,
        "phase": "summary",
        "id": monitor_cfg["id"],
        "text": (
            summary_result.text if summary_result.status == "ok"
            else f"[summary failed: {summary_result.error}]"
        ),
        "latency_ms": summary_result.latency_ms,
        "status": summary_result.status,
    })

    from dialogue.render import format_dialogue_markdown
    state.result_markdown = format_dialogue_markdown(state, question)
    mark_phase(state, "done")
    state.dump_path = str(await asyncio.to_thread(write_dump, state, base_dir=DUMP_DIR))
