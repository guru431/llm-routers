"""Shared run-store helpers for the immutable benchmark manifests (idea 15).

A "run" is an immutable directory `results/runs/<run_id>/` holding:
  - manifest.json          (run metadata: run_id, git_sha, hashes, cli_args, ...)
  - <model_id>.jsonl       (per-model raw records for this run)
  - _judge.jsonl           (judge scores for this run, written by judge.py)

`results/runs/latest.txt` is a one-line pointer to the most recent run_id, so
report.py / judge.py can find "the current run" without arguments. The legacy
flat `results/*.jsonl` layout is still readable as a fallback (resolve_run_dir
returns None → callers glob RESULTS directly), so old data keeps working.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RUNS_DIR = RESULTS / "runs"
LATEST_TXT = RUNS_DIR / "latest.txt"


def latest_run_id() -> str | None:
    """run_id recorded in latest.txt, or None if absent / dangling."""
    if LATEST_TXT.exists():
        rid = LATEST_TXT.read_text(encoding="utf-8").strip()
        if rid and (RUNS_DIR / rid).is_dir():
            return rid
    return None


def resolve_run_dir(run_id: str | None = None) -> Path | None:
    """Resolve which immutable run directory to read from.

    `run_id` may be a bare run id under results/runs/, a path to a run dir, or a
    path to a manifest.json. None → whatever latest.txt points at. Returns None
    when neither the requested run nor any latest run exists — the caller then
    falls back to the legacy flat results/*.jsonl layout.
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
    rid = latest_run_id()
    return RUNS_DIR / rid if rid else None


def result_files(run_dir: Path | None) -> list[Path]:
    """Per-model result jsonl files in a run dir (or the flat results/ when None)."""
    base = run_dir if run_dir is not None else RESULTS
    return sorted(p for p in base.glob("*.jsonl") if not p.name.startswith("_"))


def judge_file(run_dir: Path | None) -> Path:
    """Location of _judge.jsonl for a run dir (or the flat results/ when None)."""
    base = run_dir if run_dir is not None else RESULTS
    return base / "_judge.jsonl"


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
