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


class EventWriter:
    """Append-only JSONL writer for a single job's event stream.

    When `_fh` is None the writer is a no-op (the file couldn't be opened — see
    open_writer's fallback). The event log is best-effort observability, so a
    bad path must degrade to silence, not crash the council run.
    """

    def __init__(self, path: Path, *, fh=None) -> None:
        self._path = path
        self._fh = fh

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        line = json.dumps(
            {"ts": time.time(), "event": event_type, "payload": payload},
            ensure_ascii=False,
        )
        # `print(..., file=self._fh)` would also work but `write` is clearer.
        self._fh.write(line + "\n")
        # Line-buffered mode flushes on the newline; the explicit flush is
        # belt-and-suspenders for cases where Python decides to consolidate.
        self._fh.flush()

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


def get_writer(job_id: str) -> EventWriter | None:
    with _lock:
        return _writers.get(job_id)
