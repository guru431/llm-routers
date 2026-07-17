"""Monitor-friendly event stream: one JSONL file per job.

When `council_ask_async` starts a job, every progress event (phase change,
member resolve, tool call, etc.) is appended as a single line of JSON to
`logs/events/<job_id>.jsonl`. The caller (e.g. Claude in a parent session)
can `tail -F` that file via the Monitor tool and react to events live without
polling `council_status`.

The file is line-flushed after every write so partial buffering doesn't hide
the latest event from a watcher.

Schema of one event (all events share the same envelope, payload varies)::

    {"ts": 1734567890.123, "event": "phase",
     "payload": {"phase": "stage1", ...}}

Event types currently emitted:
  - "phase"           → {"phase": <queued|stage1|stage2|stage3|done|error|cancelled>, ...}
  - "stage1_member"   → {"id": "glm", "model": "...", "status": "ok"|"error",
                          "latency_ms": int, "error": str|None,
                          "tool_calls_count": int}
  - "stage2_ranker"   → {"id": "...", "model": "...", "status": ..., ...}
  - "stage3"          → {"id": "...", "model": "...", "status": ..., ...}
  - "tool_call"       → {"member_id": ..., "name": "web_search",
                          "query": str, "status": "ok"|"error",
                          "num_results": int|None, "latency_ms": int|None}
  - "result_ready"    → {"status": "ok"|"error"|"cancelled", "error": str|None,
                          "members_ok_stage1": int, "members_ok_stage2": int,
                          "dump_path": str|None}  # terminal event — run is
                          # consumable / finished; Monitor consumers match on it
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Process-global registry of open file handles, one per job_id. The async
# orchestrator is single-thread per job so the inner write doesn't need a lock,
# but the dict-of-handles is touched from background tasks → guard with a lock.
_writers: dict[str, "EventWriter"] = {}
_lock = threading.Lock()

# Default logs base (mcp-council/logs). delete_log / prune_logs use it so callers
# that don't hold server.LOGS_DIR (e.g. state.py's job GC) can still reap files.
_DEFAULT_BASE_DIR = Path(__file__).parent / "logs"

# Cap on a single job's event log. A long multi-round web_search run emits many
# tool_call events; without a ceiling one .jsonl could grow very large and stall
# tail -F consumers. Past the cap we write one truncation notice, then stay silent
# for verbose (non-terminal) events — this is best-effort observability, not an
# audit trail. TERMINAL events (result_ready and the terminal `phase` markers) are
# ALWAYS written even past the cap, so a `tail -F --until-done` consumer can never
# hang forever waiting on a result_ready that got suppressed by the size guard.
MAX_EVENT_LOG_BYTES = 8 * 1024 * 1024  # 8 MB

# Phase values that mark the run as finished (mirrors state.TERMINAL_PHASES; kept
# local to avoid an import cycle event_log ⇄ state).
_TERMINAL_PHASES = frozenset({"done", "error", "cancelled", "interrupted"})


def _is_terminal_event(event_type: str, payload: dict[str, Any]) -> bool:
    """A terminal event tells watchers the run is consumable/finished. These must
    survive the size cap so `--until-done` consumers always see the end."""
    if event_type == "result_ready":
        return True
    return event_type == "phase" and payload.get("phase") in _TERMINAL_PHASES


class EventWriter:
    """Append-only JSONL writer for a single job's event stream.

    When `_fh` is None the writer is a no-op (the file couldn't be opened — see
    open_writer's fallback). The event log is best-effort observability, so a
    bad path must degrade to silence, not crash the council run.
    """

    def __init__(self, path: Path, *, fh=None) -> None:
        self._path = path
        self._fh = fh
        self._bytes_written = 0
        self._truncated = False

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        # Once the size cap is hit, emit a single truncation marker, then stay
        # silent for VERBOSE events so the file can't grow without bound. Terminal
        # events (result_ready / terminal phase) always fall through below so a
        # watcher's --until-done never blocks on a suppressed result_ready.
        terminal = _is_terminal_event(event_type, payload)
        if self._bytes_written >= MAX_EVENT_LOG_BYTES and not terminal:
            if not self._truncated:
                self._truncated = True
                notice = json.dumps(
                    {"ts": time.time(), "event": "log_truncated",
                     "payload": {"max_bytes": MAX_EVENT_LOG_BYTES}},
                    ensure_ascii=False,
                ) + "\n"
                self._fh.write(notice)
                self._fh.flush()
            return
        line = json.dumps(
            {"ts": time.time(), "event": event_type, "payload": payload},
            ensure_ascii=False,
        ) + "\n"
        # `print(..., file=self._fh)` would also work but `write` is clearer.
        self._fh.write(line)
        # Line-buffered mode flushes on the newline; the explicit flush is
        # belt-and-suspenders for cases where Python decides to consolidate.
        self._fh.flush()
        self._bytes_written += len(line.encode("utf-8"))

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        except Exception:  # pragma: no cover — close should not throw
            pass

    @property
    def path(self) -> Path:
        return self._path


def open_writer(job_id: str, base_dir: Path) -> EventWriter:
    """Open (or return existing) writer for `job_id`. Idempotent.

    If the file can't be created/opened (perms, bad path), returns a no-op
    writer instead of raising — the event log is best-effort and must never
    take down the background job that creates it.
    """
    with _lock:
        if job_id in _writers:
            return _writers[job_id]
        path = base_dir / "events" / f"{job_id}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Buffering=1 → line-buffered text mode, so each \n forces a flush.
            # encoding=utf-8 to keep cyrillic / emoji in event payloads readable.
            fh = path.open("a", encoding="utf-8", buffering=1)
        except OSError as e:
            print(
                f"[mcp-council] event log unavailable for {job_id} "
                f"({type(e).__name__}: {e}) — continuing without it",
                file=sys.stderr,
            )
            fh = None
        writer = EventWriter(path, fh=fh)
        _writers[job_id] = writer
        return writer


def close_writer(job_id: str) -> None:
    """Close and remove the writer for `job_id`. Safe to call multiple times."""
    with _lock:
        w = _writers.pop(job_id, None)
    if w is not None:
        w.close()


def delete_log(job_id: str, base_dir: "Path | None" = None) -> None:
    """Close (if open) and unlink a job's event-log file. Called when the job is
    GC'd so logs/events/ doesn't accumulate forever (state.py only reaped
    logs/jobs/). Best-effort — a missing file / perms error is ignored."""
    close_writer(job_id)
    base = base_dir or _DEFAULT_BASE_DIR
    try:
        (base / "events" / f"{job_id}.jsonl").unlink(missing_ok=True)
    except OSError:
        pass


def prune_logs(base_dir: "Path | None" = None, *, max_age_seconds: float) -> int:
    """Delete event-log files whose mtime is older than max_age_seconds. Returns
    the count removed. Startup reaper: a job whose in-memory state was already
    lost on a restart leaves its .jsonl behind, so sweep stale ones by age."""
    base = base_dir or _DEFAULT_BASE_DIR
    events_dir = base / "events"
    if not events_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for f in events_dir.glob("*.jsonl"):
        try:
            if now - f.stat().st_mtime > max_age_seconds:
                f.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def get_writer(job_id: str) -> EventWriter | None:
    with _lock:
        return _writers.get(job_id)
