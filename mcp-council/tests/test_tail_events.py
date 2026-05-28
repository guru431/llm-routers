"""Tests for tail_events.py — verify it tails JSONL, filters, and exits on done."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


TAIL_SCRIPT = Path(__file__).parent.parent / "tail_events.py"


def _write_lines_after_delay(path: Path, lines: list[str], delay: float) -> None:
    time.sleep(delay)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
            fh.flush()
            time.sleep(0.05)


def test_tail_emits_every_line(tmp_path: Path):
    log = tmp_path / "evt.jsonl"
    log.write_text(
        json.dumps({"ts": 1, "event": "phase", "payload": {"phase": "stage1"}}) + "\n"
        + json.dumps({"ts": 2, "event": "stage1_member", "payload": {"id": "glm"}}) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(TAIL_SCRIPT), str(log), "--until-done"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Append result_ready so the tail exits.
    threading.Thread(
        target=_write_lines_after_delay,
        args=(log, [json.dumps({"ts": 3, "event": "result_ready", "payload": {"status": "ok"}})], 0.3),
        daemon=True,
    ).start()
    out, _ = proc.communicate(timeout=10)
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert [e["event"] for e in parsed] == ["phase", "stage1_member", "result_ready"]


def test_tail_filter_keeps_only_wanted(tmp_path: Path):
    log = tmp_path / "evt.jsonl"
    log.write_text(
        json.dumps({"ts": 1, "event": "phase", "payload": {"phase": "stage1"}}) + "\n"
        + json.dumps({"ts": 2, "event": "stage1_member", "payload": {"id": "glm"}}) + "\n"
        + json.dumps({"ts": 3, "event": "tool_call", "payload": {"name": "web_search"}}) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(TAIL_SCRIPT), str(log),
         "--events", "phase,result_ready", "--until-done"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    threading.Thread(
        target=_write_lines_after_delay,
        args=(log, [json.dumps({"ts": 4, "event": "result_ready", "payload": {"status": "ok"}})], 0.3),
        daemon=True,
    ).start()
    out, _ = proc.communicate(timeout=10)
    parsed = [json.loads(l) for l in out.splitlines() if l.strip()]
    # Only phase + result_ready should pass the filter.
    events = [e["event"] for e in parsed]
    assert events == ["phase", "result_ready"]


def test_tail_waits_for_file_to_appear(tmp_path: Path):
    """Caller may start the tail before the producer creates the file."""
    log = tmp_path / "evt-late.jsonl"
    assert not log.exists()

    proc = subprocess.Popen(
        [sys.executable, str(TAIL_SCRIPT), str(log), "--until-done"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Create the file mid-tail.
    def create_and_write():
        time.sleep(0.4)
        log.write_text(
            json.dumps({"ts": 1, "event": "result_ready", "payload": {"status": "ok"}}) + "\n",
            encoding="utf-8",
        )
    threading.Thread(target=create_and_write, daemon=True).start()
    out, _ = proc.communicate(timeout=10)
    parsed = [json.loads(l) for l in out.splitlines() if l.strip()]
    assert len(parsed) == 1
    assert parsed[0]["event"] == "result_ready"
