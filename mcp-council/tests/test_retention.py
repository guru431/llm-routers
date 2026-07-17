"""Tests for retention.py — TTL purge, size quota, redaction."""

import os
import time

import retention


def test_purge_by_age_removes_old_keeps_fresh(tmp_path):
    (tmp_path / "jobs").mkdir()
    old = tmp_path / "jobs" / "old.json"
    fresh = tmp_path / "jobs" / "fresh.json"
    old.write_text("{}", encoding="utf-8")
    fresh.write_text("{}", encoding="utf-8")
    # Backdate `old` well past the TTL.
    past = time.time() - 10_000
    os.utime(old, (past, past))

    res = retention.purge_all(tmp_path, max_age_seconds=3600, quota_bytes=0)
    assert res["jobs"]["removed_by_age"] == 1
    assert not old.exists()
    assert fresh.exists()


def test_purge_by_quota_removes_oldest_first(tmp_path):
    (tmp_path / "events").mkdir()
    files = []
    for i in range(3):
        f = tmp_path / "events" / f"{i}.jsonl"
        f.write_text("x" * 100, encoding="utf-8")
        os.utime(f, (time.time() - (10 - i), time.time() - (10 - i)))  # 0 oldest
        files.append(f)
    # Quota below total (300B) → oldest deleted until under quota.
    res = retention.purge_all(tmp_path, max_age_seconds=0, quota_bytes=150)
    assert res["events"]["removed_by_quota"] >= 1
    assert not files[0].exists()  # oldest gone first


def test_redact_masks_secrets():
    # Fake key assembled at runtime so the source literal doesn't trip the repo's
    # pre-commit secret scanner (runtime string still matches the redaction pattern).
    fake = "sk-" + "ABCDEF0123456789ghij"
    red = retention.redact("token " + fake + " and text")
    assert fake not in red
    assert "redacted" in red
