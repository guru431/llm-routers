"""Dialogue engine — generic round loop for all modes.

Contains:
- `_call_model`: thin wrapper over single_call.run_single, isolated for mocking.
- `_run_turn`: one participant, one LLM call, return TurnResult.
- `_run_phase`: parallel turns across participants.
- `run_round`: critique + response (Task 4).
- `run_dialogue`: full N-round loop with failure threshold and dump (Task 5).

Mode-specific orchestrators (debate.py, panel.py, socratic.py) build on top.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from single_call import run_single


@dataclass
class TurnResult:
    id: str
    model: str
    status: str  # "ok" | "error"
    text: str
    error: str | None
    latency_ms: int


async def _call_model(
    cfg: dict,
    prompt: str,
    max_tokens: int,
    web_search: bool,
) -> str:
    """Thin wrapper, isolated so tests can monkeypatch it."""
    return await run_single(cfg, prompt=prompt, max_tokens=max_tokens, web_search=web_search)


async def _run_turn(
    *,
    cfg: dict,
    prompt: str,
    max_tokens: int,
    web_search: bool,
) -> TurnResult:
    start = time.monotonic()
    try:
        text = await _call_model(cfg, prompt, max_tokens, web_search)
        latency_ms = int((time.monotonic() - start) * 1000)
        return TurnResult(
            id=cfg["id"],
            model=cfg["model"],
            status="ok",
            text=text,
            error=None,
            latency_ms=latency_ms,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return TurnResult(
            id=cfg["id"],
            model=cfg["model"],
            status="error",
            text="",
            error=f"{type(e).__name__}: {e}",
            latency_ms=latency_ms,
        )


async def _run_phase(
    *,
    participants: list[dict],
    prompt_builder: Callable[[dict], str],
    max_tokens: int,
    web_search: bool,
) -> list[TurnResult]:
    """Run one phase (e.g. critique or response) — all participants in parallel."""
    coros = [
        _run_turn(
            cfg=p,
            prompt=prompt_builder(p),
            max_tokens=max_tokens,
            web_search=web_search,
        )
        for p in participants
    ]
    return await asyncio.gather(*coros)


from dialogue.prompts import (  # noqa: E402
    render_critique_prompt,
    render_response_prompt,
)
from dialogue.state import DialogueState, mark_phase  # noqa: E402


def _append_turn_to_history(
    state: DialogueState,
    *,
    round_n: int,
    phase: str,
    result: TurnResult,
) -> None:
    state.history.append({
        "round": round_n,
        "phase": phase,
        "id": result.id,
        "text": result.text if result.status == "ok" else f"[error: {result.error}]",
        "latency_ms": result.latency_ms,
        "status": result.status,
    })


def _has_prior_responses(state: DialogueState, round_n: int) -> bool:
    return any(
        h["round"] < round_n and h["phase"] in {"response", "answer"}
        for h in state.history
    )


def _participant_to_cfg(p: dict) -> dict:
    """Project a state.participants entry into the engine cfg shape (id,
    model, base_url, env_key, optional `extra` / `min_max_tokens`)."""
    skip = {"position", "role"}
    return {k: v for k, v in p.items() if k not in skip}


async def run_round(
    *,
    state: DialogueState,
    round_n: int,
    topic: str,
    role_descriptors: dict[str, str],
    max_tokens: int,
    web_search: bool,
    anti_agreement_rules: dict[str, str] | None,
    files_section: str | None,
    do_critique: bool,
) -> None:
    """Execute one round (critique + response) for the given state.

    Critique is auto-skipped when there is no prior round to critique
    (regardless of do_critique flag) or when do_critique=False (socratic).

    Results are appended to state.history in place. state.phase is updated
    via mark_phase() at the start of each sub-phase.
    """
    participants = state.participants
    rules = anti_agreement_rules or {}
    cfgs = [_participant_to_cfg(p) for p in participants]

    # --- PHASE A: critique ---
    if do_critique and _has_prior_responses(state, round_n):
        mark_phase(state, f"round_{round_n}_critique")

        def critique_prompt_for(cfg: dict) -> str:
            return render_critique_prompt(
                topic=topic,
                role_descriptor=role_descriptors[cfg["id"]],
                history=state.history,
                round_n=round_n,
                files_section=files_section,
                anti_agreement_rule=rules.get(cfg["id"]),
            )

        critique_results = await _run_phase(
            participants=cfgs,
            prompt_builder=critique_prompt_for,
            max_tokens=max_tokens,
            web_search=False,
        )
        for r in critique_results:
            _append_turn_to_history(state, round_n=round_n, phase="critique", result=r)

    # --- PHASE B: response ---
    mark_phase(state, f"round_{round_n}_response")

    def response_prompt_for(cfg: dict) -> str:
        return render_response_prompt(
            topic=topic,
            role_descriptor=role_descriptors[cfg["id"]],
            history=state.history,
            round_n=round_n,
            files_section=files_section,
            anti_agreement_rule=rules.get(cfg["id"]),
        )

    response_results = await _run_phase(
        participants=cfgs,
        prompt_builder=response_prompt_for,
        max_tokens=max_tokens,
        web_search=web_search,
    )
    for r in response_results:
        _append_turn_to_history(state, round_n=round_n, phase="response", result=r)

    state.current_round = round_n


import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Awaitable  # noqa: E402

# A round is aborted if at least this many participants failed in any phase
# (ceil(participants * FAILURE_THRESHOLD_RATIO), with a floor of 2).
FAILURE_THRESHOLD_RATIO = 0.5

# Compatibility alias for tests that import FAILURE_THRESHOLD directly.
FAILURE_THRESHOLD = FAILURE_THRESHOLD_RATIO


def _count_failures_in_round(state: DialogueState, round_n: int) -> int:
    """Number of *distinct* participants who failed at least once in round_n.

    Counting entries instead would double-count: one bad participant fails both
    critique and response, but they're the same participant from a 'is the run
    still viable' perspective.
    """
    failed_ids = {
        h["id"] for h in state.history
        if h["round"] == round_n and h.get("status") == "error"
    }
    return len(failed_ids)


PerRoundHook = Callable[[DialogueState, int], Awaitable[None]]


async def run_dialogue(
    *,
    state: DialogueState,
    topic: str,
    role_descriptors: dict[str, str],
    max_tokens: int,
    web_search: bool,
    files_section: str | None,
    do_critique: bool,
    per_round_hook: PerRoundHook | None,
    start_round: int = 1,
) -> None:
    """Run rounds [start_round .. state.total_rounds] inclusive.

    After each round, calls per_round_hook(state, round_n) if provided —
    mode-specific orchestrators use this for diversity-monitor (panel) or
    other post-round logic. Hook may modify state.history (e.g. monitor
    appends its own entries) or trigger re-prompts.

    If at least ceil(participants * FAILURE_THRESHOLD_RATIO) participants
    failed in either phase of a round, the loop aborts with phase=error and
    raises RuntimeError.

    start_round > 1 supports dialogue_continue: caller bumps total_rounds and
    calls run_dialogue again with start_round=state.current_round+1.
    """
    n_participants = len(state.participants)
    threshold = max(2, int((n_participants * FAILURE_THRESHOLD_RATIO) + 0.99))

    for round_n in range(start_round, state.total_rounds + 1):
        await run_round(
            state=state,
            round_n=round_n,
            topic=topic,
            role_descriptors=role_descriptors,
            max_tokens=max_tokens,
            web_search=web_search,
            anti_agreement_rules=None,
            files_section=files_section,
            do_critique=do_critique,
        )

        failures = _count_failures_in_round(state, round_n)
        if failures >= threshold:
            state.error = (
                f"failure threshold exceeded in round {round_n}: "
                f"{failures}/{n_participants} participants failed"
            )
            mark_phase(state, "error")
            raise RuntimeError(state.error)

        if per_round_hook is not None:
            await per_round_hook(state, round_n)


def write_dump(state: DialogueState, *, base_dir: Path) -> Path:
    """Persist the full state snapshot to <base_dir>/<session_id>.json.

    base_dir is created if missing. Atomic-ish (write + rename).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    dump_path = base_dir / f"{state.session_id}.json"
    payload = {
        "session_id": state.session_id,
        "mode": state.mode,
        "question_preview": state.question_preview,
        "total_rounds": state.total_rounds,
        "current_round": state.current_round,
        "phase": state.phase,
        "participants": state.participants,
        "moderator": state.moderator,
        "history": state.history,
        "diversity_scores": state.diversity_scores,
        "devils_advocates": state.devils_advocates,
        "created_at": state.created_at,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "error": state.error,
    }
    tmp = dump_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # On Windows os.replace fails with PermissionError if dump_path is held
    # open by another process (antivirus scan, `tail -F`, IDE preview). Retry
    # a few times with a short sleep — almost always resolves within 200ms.
    last_err: Exception | None = None
    for _ in range(5):
        try:
            tmp.replace(dump_path)
            return dump_path
        except PermissionError as e:
            last_err = e
            time.sleep(0.05)
    # Last resort: overwrite directly. Loses atomicity but a stale dump_path
    # is worse than a successful (non-atomic) write of the latest snapshot.
    try:
        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return dump_path
    except OSError as e:
        raise last_err or e
