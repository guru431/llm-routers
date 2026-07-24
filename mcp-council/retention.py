"""Privacy / retention controls for mcp-council on-disk artifacts.

The server writes, under `logs/`:

  * `calls/*.json`      — FULL per-call dumps (question, context excerpts,
                          every member's answer, provider bodies);
  * `council_*.log`     — the per-day JSONL summary journal at the logs ROOT;
  * `jobs/*.json`       — async job snapshots;
  * `dialogues/*.json`  — dialogue session dumps;
  * `events/*.jsonl`    — live event journals.

All of them can carry prompt text and context excerpts. This module bounds how
long they live:

  * **TTL purge** — delete artifacts older than a configurable age
    (COUNCIL_LOG_RETENTION_HOURS, default 168h/7d; 0 disables purge).
  * **size quota** — after the age sweep, if a directory still exceeds its byte
    quota, delete oldest-first until under it.
  * **redaction** — mask credential-shaped tokens (reuses the dlp secret
    patterns). `logger.py` applies it to the call dumps and the JSONL journal
    before writing, so a key pasted into a question isn't parked on disk in
    clear text. It is deliberately NOT applied to `jobs/` and `dialogues/`
    snapshots: those are working state that must round-trip verbatim (a
    recovered job returns its result to the client), and mangling them would
    corrupt the answer. They are bounded by the TTL/quota sweep instead.
  * **purge API** — `purge_all()` returns per-directory counts. Called at server
    startup (so the configured TTL is a real retention period, not a promise
    that only holds when someone runs a tool) and exposed on demand through the
    `council_purge_logs` MCP tool.

Dependency-light (only dlp for the redaction patterns) so it unit-tests without
the server.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dlp import _SECRET_PATTERNS

_RETENTION_HOURS_ENV = "COUNCIL_LOG_RETENTION_HOURS"
_DEFAULT_RETENTION_HOURS = 168.0  # 7 days
# Per-directory byte quota (best-effort). 0 disables the size sweep.
_DIR_BYTE_QUOTA_ENV = "COUNCIL_LOG_DIR_QUOTA_BYTES"
_DEFAULT_DIR_BYTE_QUOTA = 256 * 1024 * 1024  # 256 MB

# The artifact directories under a logs base, with their file globs. `calls` is
# where write_full_dump() actually puts the full per-call dumps — the old entry
# said `dumps`, a directory nothing ever writes, so the largest and most
# sensitive artifact set was silently exempt from every sweep.
_RETAINED_GLOBS = {
    "jobs": "*.json",
    "dialogues": "*.json",
    "events": "*.jsonl",
    "calls": "*.json",
}

# Files living directly in the logs root (the per-day JSONL journal). Swept with
# the same TTL/quota as the subdirectories — they were in no registry at all.
_ROOT_GLOBS = ("council_*.log",)


def retention_seconds() -> float:
    """Configured TTL in seconds. 0 (or negative) disables the age purge."""
    try:
        hours = float(os.environ.get(_RETENTION_HOURS_ENV, _DEFAULT_RETENTION_HOURS))
    except ValueError:
        hours = _DEFAULT_RETENTION_HOURS
    return max(0.0, hours) * 3600.0


def dir_byte_quota() -> int:
    try:
        q = int(os.environ.get(_DIR_BYTE_QUOTA_ENV, _DEFAULT_DIR_BYTE_QUOTA))
    except ValueError:
        q = _DEFAULT_DIR_BYTE_QUOTA
    return max(0, q)


def _purge_dir_by_age(d: Path, glob: str, max_age_seconds: float, now: float) -> int:
    removed = 0
    for f in d.glob(glob):
        try:
            if now - f.stat().st_mtime > max_age_seconds:
                f.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def _purge_dir_by_quota(d: Path, glob: str, quota_bytes: int) -> int:
    if quota_bytes <= 0:
        return 0
    files: list[tuple[float, int, Path]] = []
    total = 0
    for f in d.glob(glob):
        try:
            st = f.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, f))
        total += st.st_size
    if total <= quota_bytes:
        return 0
    files.sort(key=lambda t: t[0])  # oldest first
    removed = 0
    for _mtime, size, f in files:
        if total <= quota_bytes:
            break
        try:
            f.unlink(missing_ok=True)
            total -= size
            removed += 1
        except OSError:
            continue
    return removed


def purge_all(logs_base: Path, *, max_age_seconds: float | None = None,
              quota_bytes: int | None = None) -> dict:
    """Purge expired + over-quota artifacts under `logs_base`. Returns per-dir
    counts. Best-effort — a missing dir contributes 0 and never raises."""
    now = time.time()
    age = retention_seconds() if max_age_seconds is None else max_age_seconds
    quota = dir_byte_quota() if quota_bytes is None else quota_bytes
    result: dict[str, dict] = {}
    for sub, glob in _RETAINED_GLOBS.items():
        d = logs_base / sub
        by_age = _purge_dir_by_age(d, glob, age, now) if (age > 0 and d.exists()) else 0
        by_quota = _purge_dir_by_quota(d, glob, quota) if d.exists() else 0
        result[sub] = {"removed_by_age": by_age, "removed_by_quota": by_quota}
    if logs_base.exists():
        root_age = root_quota = 0
        for glob in _ROOT_GLOBS:
            root_age += _purge_dir_by_age(logs_base, glob, age, now) if age > 0 else 0
            root_quota += _purge_dir_by_quota(logs_base, glob, quota)
        result["root"] = {"removed_by_age": root_age, "removed_by_quota": root_quota}
    else:
        result["root"] = {"removed_by_age": 0, "removed_by_quota": 0}
    result["retention_hours"] = round(age / 3600.0, 2)
    result["dir_quota_bytes"] = quota
    return result


# --- Redaction --------------------------------------------------------------

_REDACT_PATTERNS = [pat for _label, pat in _SECRET_PATTERNS]


def redact(text: str) -> str:
    """Mask credential-shaped tokens in `text` (for privacy-mode logging). Each
    secret match becomes `‹redacted:LABEL›`. Idempotent-ish; best-effort."""
    if not text:
        return text
    out = text
    for label, pat in _SECRET_PATTERNS:
        out = pat.sub(f"‹redacted:{label}›", out)
    return out
