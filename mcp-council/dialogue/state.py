"""In-memory session store for dialogue tools.

Separate from `state.py` (which manages council Karpathy jobs) because the
dialogue model has different shape: phases are round-keyed (round_N_critique,
round_N_response, round_N_diversity, etc.), not stage-keyed.

Mid-run snapshots are persisted to logs/dialogues/<session_id>.json after every
round (see engine.write_dump); load_persisted_dialogues() restores them at
startup, marking still-running sessions as 'interrupted' (mirrors council's
state.py, which persists per-job). Hard cap MAX_ACTIVE_SESSIONS prevents memory
leak; stale terminal sessions are pruned opportunistically on create_session.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MAX_ACTIVE_SESSIONS = 20

INACTIVE_TIMEOUT_SECONDS = 2 * 3600

ACTIVE_PHASES = {"starting"}
TERMINAL_PHASES = {"done", "error", "cancelled", "interrupted"}

# Where engine.write_dump persists session snapshots. Override with
# COUNCIL_DIALOGUES_DIR (read at call time so tests can isolate it).
_DEFAULT_DUMP_DIR = Path(__file__).parent.parent / "logs" / "dialogues"


def resolve_dump_dir(default: Path) -> Path:
    """Single source of the COUNCIL_DIALOGUES_DIR precedence: the env override if
    set, else `default`. EVERY dialogue writer AND the loader/GC route through
    this, so an override can't send snapshots to one directory while recovery
    reads another (which silently lost sessions on restart)."""
    return Path(os.environ.get("COUNCIL_DIALOGUES_DIR") or default)


def _dump_dir() -> Path:
    return resolve_dump_dir(_DEFAULT_DUMP_DIR)


def _quarantine_dump(f: Path, reason: str) -> None:
    """Move an unloadable dialogue snapshot into `<dir>/corrupt/` and log why —
    same contract as state.quarantine_snapshot (kept local to avoid a dialogue →
    council-state import just for this)."""
    print(
        f"[mcp-council] quarantining unreadable dialogue snapshot {f.name}: {reason}",
        file=sys.stderr,
    )
    try:
        qdir = f.parent / "corrupt"
        qdir.mkdir(parents=True, exist_ok=True)
        f.replace(qdir / f.name)
    except OSError:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass


def _unlink_dump(session_id: str) -> None:
    """Delete a session's persisted snapshot (logs/dialogues/<id>.json).
    Best-effort — a missing file / perms error is ignored."""
    try:
        (_dump_dir() / f"{session_id}.json").unlink(missing_ok=True)
    except OSError:
        pass


@dataclass
class DialogueState:
    session_id: str
    mode: Literal["debate", "panel", "socratic"]
    question_preview: str
    total_rounds: int
    created_at: float

    # Full, untruncated topic. question_preview is only the first 120 chars for
    # listings; dialogue_continue and the runners must use this so a >120-char
    # question isn't silently resumed on a truncated task.
    question: str = ""

    phase: str = "starting"
    current_round: int = 0
    participants: list[dict] = field(default_factory=list)
    moderator: dict | None = None
    history: list[dict] = field(default_factory=list)
    diversity_scores: list[int] = field(default_factory=list)
    # Structured per-round diversity-monitor outcomes (diversity monitor 2.0):
    # each entry {round, status: ok|failed, score, uncertainty, agreers,
    # post_reprompt_score, delta}. `status="failed"` distinguishes a monitor
    # call/parse failure from a genuine score=0 (both used to look identical).
    diversity_monitor_status: list[dict] = field(default_factory=list)
    devils_advocates: list[str] = field(default_factory=list)

    started_at: float | None = None
    finished_at: float | None = None
    last_activity: float = 0.0
    error: str | None = None
    result_markdown: str | None = None
    dump_path: str | None = None
    # Non-fatal degradations that still let the run reach 'done' (a failed final
    # summary, a diversity-monitor call that errored). Surfaced by dialogue_result
    # so a 'done' with partial quality isn't indistinguishable from a clean run.
    warnings: list[str] = field(default_factory=list)

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
    # Set by cancel_session once a cancel has been requested, so a second
    # concurrent call (the runner hasn't reached its terminal phase yet, so the
    # phase pre-check still passes) reports False instead of claiming it
    # cancelled a run that was already cancelling. Non-persisted.
    _cancel_requested: bool = field(default=False, repr=False)
    # Optional append-only event-journal writer (event_log.EventWriter), attached
    # by the server so the dialogue emits per-round events a Monitor consumer can
    # tail. Non-persisted (rebound per run); None in tests / sync runs.
    event_writer: object | None = field(default=None, repr=False)
    # Serializes dialogue_continue on THIS session: two concurrent continues used
    # to both pass the terminal-phase gate and each spawn a runner (double LLM
    # spend, doubled total_rounds, corrupted history). Lazily binds to the loop on
    # first acquire, so constructing it in create_session / _state_from_dump before
    # the loop serves is safe.
    _continue_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


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
        # Delete the on-disk snapshot too — GC used to evict from memory only,
        # leaving logs/dialogues/<id>.json to accumulate forever.
        _unlink_dump(sid)


def _state_from_dump(data: dict) -> DialogueState:
    """Rebuild a DialogueState from a persisted snapshot. A non-terminal
    persisted phase becomes 'interrupted' (the run died with the previous
    process)."""
    s = DialogueState(
        session_id=data["session_id"],
        mode=data.get("mode", "panel"),  # type: ignore[arg-type]
        question_preview=data.get("question_preview", ""),
        total_rounds=data.get("total_rounds") or 1,
        created_at=data.get("created_at") or time.time(),
        question=data.get("question") or data.get("question_preview", ""),
    )
    s.current_round = data.get("current_round") or 0
    s.participants = data.get("participants") or []
    s.moderator = data.get("moderator")
    s.history = data.get("history") or []
    s.diversity_scores = data.get("diversity_scores") or []
    s.diversity_monitor_status = data.get("diversity_monitor_status") or []
    s.devils_advocates = data.get("devils_advocates") or []
    s.started_at = data.get("started_at")
    s.finished_at = data.get("finished_at")
    s.error = data.get("error")
    s.warnings = data.get("warnings") or []
    s.result_markdown = data.get("result_markdown")
    s.dump_path = data.get("dump_path")
    s.web_search = bool(data.get("web_search"))
    s.max_tokens = data.get("max_tokens") or 4096
    s.context_paths = data.get("context_paths") or []
    s.diversity_monitor = bool(data.get("diversity_monitor", True))
    # A restored threshold of 0 ("re-prompt on ANY agreement") is legitimate,
    # so `or 7` would corrupt it — guard explicitly on None instead.
    s.diversity_threshold = (
        data.get("diversity_threshold")
        if data.get("diversity_threshold") is not None
        else 7
    )
    s.devils_advocate_rotation = bool(data.get("devils_advocate_rotation", True))
    phase = data.get("phase") or "starting"
    now = time.time()
    if phase not in TERMINAL_PHASES:
        s.phase = "interrupted"
        s.error = s.error or "server restarted mid-run (partial history available)"
        s.finished_at = s.finished_at or now
    else:
        s.phase = phase
    # Prefer the persisted last_activity so a session that was active/finished
    # recently (but CREATED hours ago) isn't judged stale by the loader — matches
    # the runtime GC, which also keys on last_activity. Fall back through the
    # timing fields for snapshots written before last_activity was persisted.
    s.last_activity = (
        data.get("last_activity")
        or s.finished_at or s.started_at or s.created_at
    )
    return s


def _valid_dump(data: object) -> tuple[dict, str, float] | None:
    """Type-validate a parsed dialogue snapshot BEFORE any arithmetic on it.
    Returns (data, session_id, activity_timestamp) or None when unusable.

    The old loader did `now - activity` and `sid in _sessions` straight after
    json.loads, outside any guard: a string/list timestamp raised TypeError and
    an unhashable session_id raised TypeError, aborting the load of EVERY
    remaining snapshot — and, since the loader runs at startup, the server."""
    if not isinstance(data, dict):
        return None
    sid = data.get("session_id")
    if not isinstance(sid, str) or not sid:
        return None
    # First TRUTHY timestamp wins — a 0.0 means "never set" and falls through to
    # the next field (matches the original `or` chain), but a present non-numeric
    # value is a malformed snapshot rather than something to compute against.
    activity: float = 0.0
    for key in ("last_activity", "finished_at", "started_at", "created_at"):
        v = data.get(key)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        if v:
            activity = float(v)
            break
    return data, sid, activity


def load_persisted_dialogues() -> int:
    """Load persisted session snapshots into memory at startup, marking
    non-terminal sessions as 'interrupted'. Returns the number loaded.
    Synchronous — intended to run once before the event loop serves.

    Every per-file step is guarded: one malformed snapshot is quarantined (moved
    to `<dir>/corrupt/`) and the rest still load. Startup must never hinge on a
    single file's contents."""
    d = _dump_dir()
    if not d.exists():
        return 0
    now = time.time()
    loaded = 0
    for f in sorted(d.glob("*.json")):
        try:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except OSError:
                continue  # transient read problem — leave the file alone
            except ValueError as e:
                _quarantine_dump(f, f"invalid JSON: {e}")
                continue

            valid = _valid_dump(data)
            if valid is None:
                _quarantine_dump(f, "snapshot shape/type invalid")
                continue
            data, sid, activity = valid

            # Drop snapshots past the inactive timeout so a restart doesn't
            # resurrect ancient sessions; matches the in-memory GC horizon (which
            # keys on last_activity, NOT created_at — a long-lived but recently
            # active session must survive a restart). Unlink so the file isn't
            # rescanned every restart.
            if now - activity > INACTIVE_TIMEOUT_SECONDS:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            if sid in _sessions:
                continue
            _sessions[sid] = _state_from_dump(data)
            loaded += 1
        except Exception as e:  # noqa: BLE001 — one bad file must not stop startup
            _quarantine_dump(f, f"{type(e).__name__}: {e}")
            continue
    return loaded


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
        # Count only non-terminal sessions toward the cap. Terminal sessions
        # linger in _sessions until GC prunes them (2h), but they hold no
        # resources, so they must not block new work once finished.
        active = sum(1 for s in _sessions.values() if s.phase not in TERMINAL_PHASES)
        if active >= MAX_ACTIVE_SESSIONS:
            raise RuntimeError(
                f"too many active sessions ({active}/{MAX_ACTIVE_SESSIONS}); "
                "wait for some to finish or call dialogue_cancel on stale ones"
            )
        sid = _new_session_id()
        s = DialogueState(
            session_id=sid,
            mode=mode,  # type: ignore[arg-type]
            question=question_preview,
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


RESUMABLE_PHASES = ("done", "interrupted")


async def reserve_active_slot(state: DialogueState) -> str:
    """Atomically claim one active-session slot FOR `state` (dialogue_continue).

    Check-then-act was a real race: the old version only COUNTED active sessions
    under the global lock and released it before the caller flipped the phase to
    'starting'. Two continues on two different terminal sessions both saw the
    same single free slot and both activated, blowing past MAX_ACTIVE_SESSIONS
    and the spend ceiling tied to it. Here the count and the transition happen
    under ONE hold of the global lock, so the slot a caller sees free is the slot
    it takes.

    Returns the phase the session had, so the caller can roll back if the
    remaining pre-flight fails. Raises RuntimeError if the session is not
    resumable or the cap is reached.
    """
    async with _sessions_lock:
        now = time.time()
        _gc_locked(now)
        if state.phase not in RESUMABLE_PHASES:
            raise RuntimeError(
                f"dialogue_continue: session already resuming or active "
                f"(phase '{state.phase}')"
            )
        active = sum(1 for s in _sessions.values() if s.phase not in TERMINAL_PHASES)
        if active >= MAX_ACTIVE_SESSIONS:
            raise RuntimeError(
                f"too many active sessions ({active}/{MAX_ACTIVE_SESSIONS}); "
                "wait for some to finish or call dialogue_cancel on stale ones"
            )
        previous = state.phase
        # The transition IS the reservation — from here the session counts as
        # active for every concurrent caller.
        state.phase = "starting"
        state.last_activity = now
        return previous


async def release_active_slot(state: DialogueState, previous_phase: str) -> None:
    """Undo `reserve_active_slot` when the caller's remaining pre-flight fails,
    so a rejected continue doesn't leave a zombie session holding a slot."""
    async with _sessions_lock:
        if state.phase == "starting":
            state.phase = previous_phase
            state.last_activity = time.time()


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
        if s._cancel_requested:
            return False
        task = s._task
        s._cancel_requested = True
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
        "warnings": list(state.warnings),
        "has_result": state.result_markdown is not None,
        "dump_path": state.dump_path,
        "diversity_scores": list(state.diversity_scores),
        "diversity_monitor_status": list(state.diversity_monitor_status),
        "devils_advocates": list(state.devils_advocates),
    }


async def _reset_for_tests() -> None:
    """Test-only: clear the store, cancel any bound tasks, and remove persisted
    dump files so a later test's load_persisted_dialogues starts clean."""
    async with _sessions_lock:
        for s in _sessions.values():
            if s._task is not None and not s._task.done():
                s._task.cancel()
        _sessions.clear()
    try:
        for f in _dump_dir().glob("*.json*"):
            f.unlink(missing_ok=True)
    except OSError:
        pass
