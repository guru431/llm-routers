"""Tests for logger module."""

import json
from pathlib import Path

import logger


def test_log_call_writes_record(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    logger.log_call(
        call_id="abc",
        members_total=6,
        members_ok_stage1=5,
        members_ok_stage2=5,
        prompt_size_bytes=1024,
        total_latency_ms=90000,
        status="ok",
        log_dump="logs/calls/abc.json",
    )
    files = list(tmp_path.glob("council_*.log"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["call_id"] == "abc"
    assert record["tool"] == "council_ask"
    assert record["members_ok_stage1"] == 5
    assert record["status"] == "ok"


def test_log_call_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    for i in range(3):
        logger.log_call(
            call_id=f"c{i}",
            members_total=6,
            members_ok_stage1=6,
            members_ok_stage2=6,
            prompt_size_bytes=100,
            total_latency_ms=1000,
            status="ok",
            log_dump=None,
        )
    files = list(tmp_path.glob("council_*.log"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_write_full_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "CALLS_DIR", tmp_path)
    dump = {"call_id": "abc", "question": "q", "stage1": [], "stage2": [], "aggregate": [], "notes": []}
    path = logger.write_full_dump("abc", dump)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == dump


def test_new_call_id_unique():
    a = logger._new_call_id()
    b = logger._new_call_id()
    assert a != b
    assert len(a) >= len("2026-05-20-120000-aaaa")


def test_full_dump_is_redacted(tmp_path, monkeypatch):
    """A key pasted into the question must not sit in the on-disk dump: the dump
    holds the whole prompt + provider bodies for the full retention window."""
    monkeypatch.setattr(logger, "CALLS_DIR", tmp_path)
    fake = "sk-" + "ABCDEF0123456789ghij"
    path = logger.write_full_dump("cid", {"question": "use " + fake + " please"})
    text = path.read_text(encoding="utf-8")
    assert fake not in text
    assert "redacted" in text


def test_log_call_record_is_redacted(tmp_path, monkeypatch):
    """`status` carries provider error text, which can echo a key-bearing URL."""
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    fake = "sk-" + "ABCDEF0123456789ghij"
    logger.log_call(
        call_id="cid", members_total=1, members_ok_stage1=0, members_ok_stage2=0,
        prompt_size_bytes=1, total_latency_ms=1,
        status="error: https://api.example/v1?key=" + fake, log_dump=None,
    )
    written = list(tmp_path.glob("council_*.log"))[0].read_text(encoding="utf-8")
    assert fake not in written


def test_redaction_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "CALLS_DIR", tmp_path)
    monkeypatch.setenv("COUNCIL_LOG_REDACT", "0")
    fake = "sk-" + "ABCDEF0123456789ghij"
    path = logger.write_full_dump("cid2", {"question": fake})
    assert fake in path.read_text(encoding="utf-8")
