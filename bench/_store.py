"""Shared run-store helpers for the immutable benchmark manifests (idea 15).

A "run" is an immutable directory `results/runs/<run_id>/` holding:
  - manifest.json          (run metadata: run_id, git_sha, hashes, cli_args, ...)
  - <model_id>.jsonl       (per-model raw records for this run)
  - _judge.jsonl           (judge scores for this run, written by judge.py)

`results/runs/latest.txt` is a one-line pointer to the most recently STARTED
run_id; `results/runs/latest-complete.txt` points at the most recent run that
actually finished every expected cell. judge.py / report.py resolve
"the current run" through the COMPLETE pointer by default, because a run dir is
created (and latest.txt written) before the first cell executes: after a crash
or Ctrl-C the started-pointer names a partial run, and rankings built from it
looked exactly like rankings built from a full one.

The legacy flat `results/*.jsonl` layout is still readable as a fallback
(resolve_run_dir returns None → callers glob RESULTS directly), so old data
keeps working.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RUNS_DIR = RESULTS / "runs"
LATEST_TXT = RUNS_DIR / "latest.txt"
# Written ONLY after a run completes every expected cell.
LATEST_COMPLETE_TXT = RUNS_DIR / "latest-complete.txt"


def _pointer(path: Path) -> str | None:
    if path.exists():
        rid = path.read_text(encoding="utf-8").strip()
        if rid and (RUNS_DIR / rid).is_dir():
            return rid
    return None


def latest_run_id() -> str | None:
    """run_id recorded in latest.txt (most recently STARTED run; may be partial)."""
    return _pointer(LATEST_TXT)


def latest_complete_run_id() -> str | None:
    """run_id of the most recent run that finished every expected cell."""
    return _pointer(LATEST_COMPLETE_TXT)


def run_status(run_dir: Path | None) -> dict:
    """Lifecycle view of a run: {status, expected_cells, completed_cells, complete}.

    `status` ∈ started | completed | failed | unknown. `complete` is the single
    boolean callers gate on; it is True only when the manifest says completed AND
    the completed/expected counts agree."""
    mf = load_manifest(run_dir) or {}
    status = mf.get("status") or "unknown"
    expected = mf.get("expected_cells")
    completed = mf.get("completed_cells")
    complete = (
        status == "completed"
        and isinstance(expected, int) and isinstance(completed, int)
        and completed >= expected > 0
    )
    return {"status": status, "expected_cells": expected,
            "completed_cells": completed, "complete": complete}


def resolve_run_dir(run_id: str | None = None, *, prefer_complete: bool = True) -> Path | None:
    """Resolve which immutable run directory to read from.

    `run_id` may be a bare run id under results/runs/, a path to a run dir, or a
    path to a manifest.json. None → the latest COMPLETE run, falling back to the
    latest started one (callers warn when that fallback is partial). Returns None
    when neither exists — the caller then falls back to the legacy flat
    results/*.jsonl layout.
    """
    if run_id:
        p = Path(run_id)
        if p.is_file() and p.name == "manifest.json":
            return p.parent
        if p.is_dir():
            return p
        cand = RUNS_DIR / run_id
        if cand.is_dir():
            return cand
        return None
    rid = latest_complete_run_id() if prefer_complete else None
    rid = rid or latest_run_id()
    return RUNS_DIR / rid if rid else None


def result_files(run_dir: Path | None) -> list[Path]:
    """Per-model result jsonl files in a run dir (or the flat results/ when None)."""
    base = run_dir if run_dir is not None else RESULTS
    return sorted(p for p in base.glob("*.jsonl") if not p.name.startswith("_"))


def judge_file(run_dir: Path | None) -> Path:
    """Location of _judge.jsonl for a run dir (or the flat results/ when None)."""
    base = run_dir if run_dir is not None else RESULTS
    return base / "_judge.jsonl"


def update_manifest(run_dir: Path | None, **fields) -> None:
    """Merge `fields` into a run's manifest.json. No-op for the flat layout.

    judge.py records the ACTIVE judge panel through this, so report.py can name
    the judges that actually scored the run instead of printing the legacy
    default in the header of every report."""
    if run_dir is None:
        return
    data = load_manifest(run_dir) or {}
    data.update(fields)
    try:
        (run_dir / "manifest.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # a read-only/archived run dir must not fail the judging pass


def load_manifest(run_dir: Path | None) -> dict | None:
    """Parse manifest.json from a run dir, or None (flat layout / missing/broken)."""
    if run_dir is None:
        return None
    mf = run_dir / "manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
