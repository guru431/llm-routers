"""In-memory session store for dialogue tools.

Separate from `state.py` (which manages council Karpathy jobs) because the
dialogue model has different shape: phases are round-keyed (round_N_critique,
round_N_response, round_N_diversity, etc.), not stage-keyed.

Lifetime is per-process; restart loses active sessions (matches council's
state.py). Hard cap MAX_ACTIVE_SESSIONS prevents memory leak; stale done/error/
cancelled sessions are pruned opportunistically when create_session is called.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

MAX_ACTIVE_SESSIONS = 20

INACTIVE_TIMEOUT_SECONDS = 2 * 3600

ACTIVE_PHASES = {"starting"}
TERMINAL_PHASES = {"done", "error", "cancelled"}


@dataclass
class DialogueState:
    session_id: str
    mode: Literal["debate", "panel", "socratic"]
    question_preview: str
    total_rounds: int
    created_at: float

    phase: str = "starting"
    current_round: int = 0
    participants: list[dict] = field(default_factory=list)
    moderator: dict | None = None
    history: list[dict] = field(default_factory=list)
    diversity_scores: list[int] = field(default_factory=list)
    devils_advocates: list[str] = field(default_factory=list)

    started_at: float | None = None
    finished_at: float | None = None
    last_activity: float = 0.0
    error: str | None = None
    result_markdown: str | None = None
    dump_path: str | None = None

    # Original session parameters, preserved so dialogue_continue can resume
    # with the same configuration instead of silently downgrading to defaults.
    web_search: bool = False
    max_tokens: int = 4096
    context_paths: list[str] = field(default_factory=list)

    # Panel-only anti-convergence settings; preserved so dialogue_continue
    # resumes a panel with the same config instead of the hardcoded defaults.
    diversity_monitor: bool = True
    diversity_threshold: int = 7
    devils_advocate_rotation: bool = True

    _task: asyncio.Task | None = field(default=None, repr=False)


_sessions: dict[str, DialogueState] = {}
_sessions_lock = asyncio.Lock()


def _new_session_id() -> str:
    return f"dlg-{uuid.uuid4().hex[:12]}"


def _gc_locked(now: float) -> None:
    """Caller holds the lock. Remove terminal sessions whose last_activity is
    older than the inactive timeout. Does not touch active sessions."""
    stale = [
        sid for sid, s in _sessions.items()
        if s.phase in TERMINAL_PHASES
        and (now - s.last_activity) > INACTIVE_TIMEOUT_SECONDS
    ]
    for sid in stale:
        del _sessions[sid]


async def create_session(
    *,
    mode: str,
    question_preview: str,
    total_rounds: int,
    web_search: bool = False,
    max_tokens: int = 4096,
    context_paths: list[str] | None = None,
) -> DialogueState:
    """Allocate a DialogueState and register it. Raises RuntimeError when the
    active-session cap is reached even after GC."""
    async with _sessions_lock:
        now = time.time()
        _gc_locked(now)
        if len(_sessions) >= MAX_ACTIVE_SESSIONS:
            raise RuntimeError(
                f"too many active sessions ({len(_sessions)}/{MAX_ACTIVE_SESSIONS}); "
                "wait for some to finish or call dialogue_cancel on stale ones"
            )
        sid = _new_session_id()
        s = DialogueState(
            session_id=sid,
            mode=mode,  # type: ignore[arg-type]
            question_preview=question_preview[:120],
            total_rounds=total_rounds,
            created_at=now,
            last_activity=now,
            web_search=web_search,
            max_tokens=max_tokens,
            context_paths=list(context_paths or []),
        )
        _sessions[sid] = s
        return s


async def get_session(session_id: str) -> DialogueState | None:
    async with _sessions_lock:
        return _sessions.get(session_id)


async def list_sessions(limit: int = 20) -> list[DialogueState]:
    async with _sessions_lock:
        items = sorted(_sessions.values(), key=lambda s: s.created_at, reverse=True)
        return items[:limit]


async def cancel_session(session_id: str) -> bool:
    """Request cancellation. Returns True if the session existed and was active.

    Phase transition is delegated to the runner's CancelledError handler so a
    task finishing in the small race window between our `task.cancel()` and
    the exception being delivered keeps its 'done' phase and result_markdown
    instead of being overwritten. Only the no-task fallback flips phase here
    (nothing else can).
    """
    async with _sessions_lock:
        s = _sessions.get(session_id)
        if s is None:
            return False
        if s.phase in TERMINAL_PHASES:
            return False
        task = s._task
        if task is None:
            # No background runner attached — no handler to delegate to, so
            # we transition synchronously here. Mirrors mark_phase('cancelled').
            now = time.time()
            s.phase = "cancelled"
            s.finished_at = now
            s.last_activity = now
            return True
    if not task.done():
        # Yield once so the task has a chance to begin executing before we
        # cancel it; otherwise cancel() on an unstarted coroutine fires
        # CancelledError before its body (and any try/except) is entered.
        await asyncio.sleep(0)
        if not task.done():
            task.cancel()
            return True
    return False


def attach_task(state: DialogueState, task: asyncio.Task) -> None:
    state._task = task


def mark_phase(state: DialogueState, phase: str) -> None:
    state.phase = phase
    now = time.time()
    state.last_activity = now
    if state.started_at is None and phase != "starting":
        state.started_at = now
    if phase in TERMINAL_PHASES:
        state.finished_at = now


def snapshot(state: DialogueState) -> dict:
    now = time.time()
    elapsed_ms = None
    if state.started_at is not None:
        end = state.finished_at if state.finished_at is not None else now
        elapsed_ms = int((end - state.started_at) * 1000)
    return {
        "session_id": state.session_id,
        "mode": state.mode,
        "phase": state.phase,
        "current_round": state.current_round,
        "total_rounds": state.total_rounds,
        "participants": list(state.participants),
        "moderator": state.moderator,
        "elapsed_ms": elapsed_ms,
        "error": state.error,
        "has_result": state.result_markdown is not None,
        "dump_path": state.dump_path,
        "diversity_scores": list(state.diversity_scores),
        "devils_advocates": list(state.devils_advocates),
    }


async def _reset_for_tests() -> None:
    """Test-only: clear the store and cancel any bound tasks."""
    async with _sessions_lock:
        for s in _sessions.values():
            if s._task is not None and not s._task.done():
                s._task.cancel()
        _sessions.clear()
