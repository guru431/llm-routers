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
"""

from __future__ import annotations

import json
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
    """Append-only JSONL writer for a single job's event stream."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Buffering=1 → line-buffered text mode, so each \n forces a flush.
        # encoding=utf-8 to keep cyrillic / emoji in event payloads readable.
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
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
        try:
            self._fh.close()
        except Exception:  # pragma: no cover — close should not throw
            pass

    @property
    def path(self) -> Path:
        return self._path


def open_writer(job_id: str, base_dir: Path) -> EventWriter:
    """Open (or return existing) writer for `job_id`. Idempotent."""
    with _lock:
        if job_id in _writers:
            return _writers[job_id]
        path = base_dir / "events" / f"{job_id}.jsonl"
        writer = EventWriter(path)
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
