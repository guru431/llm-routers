"""Tests for the per-provider circuit breaker."""
import pytest

import circuit_breaker as cb


@pytest.fixture(autouse=True)
def _reset():
    cb.reset()
    yield
    cb.reset()


def test_opens_after_threshold_consecutive_failures():
    host = "api.example.com"
    for _ in range(cb.FAILURE_THRESHOLD - 1):
        cb.record_failure(host)
    assert cb.open_for(host) == 0.0          # not yet
    cb.record_failure(host)                  # crosses threshold
    assert cb.open_for(host) > 0.0


def test_success_resets_failures():
    host = "api.example.com"
    for _ in range(cb.FAILURE_THRESHOLD - 1):
        cb.record_failure(host)
    cb.record_success(host)                  # clears the streak
    cb.record_failure(host)
    assert cb.open_for(host) == 0.0          # one failure after reset → still closed


def test_hosts_are_independent():
    cb.record_failure("a.com")
    for _ in range(cb.FAILURE_THRESHOLD):
        cb.record_failure("b.com")
    assert cb.open_for("a.com") == 0.0
    assert cb.open_for("b.com") > 0.0


def test_snapshot_reports_open_hosts():
    for _ in range(cb.FAILURE_THRESHOLD):
        cb.record_failure("down.com")
    snap = cb.snapshot()
    assert "down.com" in snap
    assert snap["down.com"]["fails"] >= cb.FAILURE_THRESHOLD
    assert snap["down.com"]["open_seconds_remaining"] > 0
