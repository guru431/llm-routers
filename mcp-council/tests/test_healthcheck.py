"""Tests for healthcheck.healthcheck_models — key/HTTP/latency triage."""
import pytest

from healthcheck import healthcheck_models, _classify_error
from models import CATALOG, UnknownModelError
from openai_client import CouncilHTTPError


@pytest.mark.parametrize("msg,expected", [
    ("http 402 insufficient_balance: ...", "insufficient_balance"),
    ("http 401: bad key", "auth"),
    ("overload after 3 attempts (last status 429)", "rate_limited"),
    ("timeout after 3 attempts: ReadTimeout", "timeout"),
    ("empty content (finish_reason=length)", "empty_response"),
    ("network error: ConnectError", "network"),
    ("http 400: weird", "error"),
])
def test_classify_error(msg, expected):
    assert _classify_error(msg) == expected


def test_disabled_model_reported_not_called(monkeypatch):
    async def boom(**kwargs):  # must never be called for disabled members
        raise AssertionError("disabled model should not be pinged")

    rows = _run(healthcheck_models(["minimax-direct"], call_fn=boom))
    assert rows[0]["status"] == "disabled"
    assert rows[0]["ok"] is False
    assert rows[0]["enabled"] is False


def test_missing_key_reported_no_call(monkeypatch):
    # glm uses OPENCODE_GO_KEY — ensure it's unset.
    monkeypatch.delenv(CATALOG["glm"]["env_key"], raising=False)

    async def boom(**kwargs):
        raise AssertionError("must not call when key missing")

    rows = _run(healthcheck_models(["glm"], call_fn=boom))
    assert rows[0]["status"] == "no_key"
    assert rows[0]["key_present"] is False


def test_ok_path(monkeypatch):
    monkeypatch.setenv(CATALOG["glm"]["env_key"], "k")

    async def fake(**kwargs):
        return {"content": "pong", "tokens_in": 3, "tokens_out": 1}

    rows = _run(healthcheck_models(["glm"], call_fn=fake))
    assert rows[0]["ok"] is True
    assert rows[0]["status"] == "ok"
    assert rows[0]["latency_ms"] is not None


def test_http_error_classified(monkeypatch):
    monkeypatch.setenv(CATALOG["glm"]["env_key"], "k")

    async def fake(**kwargs):
        raise CouncilHTTPError("http 402 insufficient_balance: no funds")

    rows = _run(healthcheck_models(["glm"], call_fn=fake))
    assert rows[0]["ok"] is False
    assert rows[0]["status"] == "insufficient_balance"


def test_unknown_id_raises():
    async def fake(**kwargs):
        return {"content": "pong", "tokens_out": 1}

    with pytest.raises(UnknownModelError):
        _run(healthcheck_models(["nope"], call_fn=fake))


def test_per_model_healthcheck_timeout_overrides_default(monkeypatch):
    # codex (local agent-server) carries healthcheck_timeout in CATALOG; glm does
    # not. The catalog override must win over the passed default for codex while
    # glm still uses the default.
    monkeypatch.setenv(CATALOG["codex"]["env_key"], "k")
    monkeypatch.setenv(CATALOG["glm"]["env_key"], "k")
    assert "healthcheck_timeout" not in CATALOG["glm"]
    seen: dict[str, float] = {}

    async def fake(**kwargs):
        seen[kwargs["model"]] = kwargs["timeout"]
        return {"content": "pong", "tokens_out": 1}

    _run(healthcheck_models(["codex", "glm"], call_fn=fake, timeout=12.0))
    assert seen[CATALOG["codex"]["model"]] == CATALOG["codex"]["healthcheck_timeout"]
    assert seen[CATALOG["glm"]["model"]] == 12.0


# --- helper ---
import asyncio


def _run(coro):
    return asyncio.run(coro)
