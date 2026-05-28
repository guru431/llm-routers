"""Tests for the Monitor-friendly event-stream JSONL writer."""

import json
from pathlib import Path

import pytest

import event_log


@pytest.fixture(autouse=True)
def reset_writers():
    """Make sure registry is clean between tests."""
    event_log._writers.clear()
    yield
    for jid in list(event_log._writers.keys()):
        event_log.close_writer(jid)


def test_open_writer_creates_file(tmp_path: Path):
    w = event_log.open_writer("job-abc", tmp_path)
    assert w.path == tmp_path / "events" / "job-abc.jsonl"
    assert w.path.parent.exists()
    event_log.close_writer("job-abc")


def test_write_appends_jsonl_lines(tmp_path: Path):
    w = event_log.open_writer("job-1", tmp_path)
    w.write("phase", {"phase": "stage1", "members": ["a", "b"]})
    w.write("stage1_member", {"id": "a", "status": "ok", "latency_ms": 1234})
    event_log.close_writer("job-1")

    lines = (tmp_path / "events" / "job-1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    assert a["event"] == "phase"
    assert a["payload"]["phase"] == "stage1"
    assert a["payload"]["members"] == ["a", "b"]
    assert "ts" in a and isinstance(a["ts"], float)

    b = json.loads(lines[1])
    assert b["event"] == "stage1_member"
    assert b["payload"]["id"] == "a"
    assert b["payload"]["latency_ms"] == 1234


def test_write_flushes_immediately(tmp_path: Path):
    """The file should be readable from another handle right after write()
    (no buffering pitfalls for tail -F consumers)."""
    w = event_log.open_writer("job-flush", tmp_path)
    w.write("phase", {"phase": "stage1"})
    # Open second handle and read — must see the line.
    text = (tmp_path / "events" / "job-flush.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    parsed = json.loads(text.strip())
    assert parsed["payload"]["phase"] == "stage1"
    event_log.close_writer("job-flush")


def test_open_writer_is_idempotent(tmp_path: Path):
    a = event_log.open_writer("job-2", tmp_path)
    b = event_log.open_writer("job-2", tmp_path)
    assert a is b


def test_close_writer_safe_to_double_close(tmp_path: Path):
    event_log.open_writer("job-3", tmp_path)
    event_log.close_writer("job-3")
    # second close should be a no-op, not raise
    event_log.close_writer("job-3")


def test_get_writer_returns_none_after_close(tmp_path: Path):
    event_log.open_writer("job-4", tmp_path)
    assert event_log.get_writer("job-4") is not None
    event_log.close_writer("job-4")
    assert event_log.get_writer("job-4") is None


def test_unicode_payload_preserved(tmp_path: Path):
    """Cyrillic and arrows shouldn't be \\uXXXX-escaped (ensure_ascii=False)."""
    w = event_log.open_writer("job-utf", tmp_path)
    w.write("phase", {"phase": "stage1", "note": "проверка → готово"})
    event_log.close_writer("job-utf")
    text = (tmp_path / "events" / "job-utf.jsonl").read_text(encoding="utf-8")
    assert "проверка" in text
    assert "→" in text
