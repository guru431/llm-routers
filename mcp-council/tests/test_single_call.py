"""Tests for single_call.run_single — the engine behind model_ask."""
import asyncio

import pytest

from single_call import run_single


def _fake_call_factory(content: str = "hello", tool_calls=None):
    """Build a fake call_openai_compat that returns the given content."""
    async def fake(**kwargs):
        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning_content": None,
            "finish_reason": "stop",
            "tokens_in": 10,
            "tokens_out": 5,
        }
    return fake


def test_run_single_happy_path(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_KEY", "fake-key")
    monkeypatch.setattr("single_call.call_openai_compat", _fake_call_factory("the answer"))
    cfg = {"id": "deepseek-flash", "model": "deepseek-v4-flash",
           "base_url": "https://x", "env_key": "DEEPSEEK_KEY"}
    out = asyncio.run(run_single(cfg, prompt="hi", max_tokens=1024))
    assert out == "the answer"


def test_run_single_respects_min_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_KEY", "fake")
    captured = {}

    async def fake(**kwargs):
        captured["max_tokens"] = kwargs["max_tokens"]
        return {
            "content": "x", "tool_calls": None, "reasoning_content": None,
            "finish_reason": "stop", "tokens_in": 1, "tokens_out": 1,
        }

    monkeypatch.setattr("single_call.call_openai_compat", fake)
    cfg = {"id": "kimi", "model": "kimi-k2.6", "base_url": "https://x",
           "env_key": "OPENCODE_GO_KEY", "min_max_tokens": 30000}
    asyncio.run(run_single(cfg, prompt="hi", max_tokens=1024))
    assert captured["max_tokens"] == 30000


def test_run_single_missing_env_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
    cfg = {"id": "deepseek-flash", "model": "deepseek-v4-flash",
           "base_url": "https://x", "env_key": "DEEPSEEK_KEY"}
    with pytest.raises(RuntimeError, match="env var DEEPSEEK_KEY"):
        asyncio.run(run_single(cfg, prompt="hi", max_tokens=1024))


def test_run_single_empty_content_returns_empty_string(monkeypatch):
    """If the model returns None content (degenerate), we return ''."""
    monkeypatch.setenv("DEEPSEEK_KEY", "fake")
    monkeypatch.setattr("single_call.call_openai_compat", _fake_call_factory(content=None))
    cfg = {"id": "deepseek-flash", "model": "deepseek-v4-flash",
           "base_url": "https://x", "env_key": "DEEPSEEK_KEY"}
    out = asyncio.run(run_single(cfg, prompt="hi", max_tokens=1024))
    assert out == ""


def test_run_single_web_search_routes_through_tool_loop(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_KEY", "fake")
    captured = {}

    async def fake_loop(**kwargs):
        captured["called"] = True
        return ({
            "content": "web answer", "tool_calls": None,
            "reasoning_content": None, "finish_reason": "stop",
            "tokens_in": 1, "tokens_out": 1,
        }, [])

    monkeypatch.setattr("single_call.run_with_tool_loop", fake_loop)
    cfg = {"id": "deepseek-flash", "model": "deepseek-v4-flash",
           "base_url": "https://x", "env_key": "DEEPSEEK_KEY"}
    out = asyncio.run(run_single(cfg, prompt="hi", max_tokens=1024, web_search=True))
    assert captured["called"] is True
    assert out == "web answer"
