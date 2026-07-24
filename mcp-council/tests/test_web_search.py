"""Tests for web_search.py (Exa client) and the tool-loop integration."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import web_search
from web_search import (
    WEB_SEARCH_TOOL_SPEC,
    WebSearchError,
    format_error_for_llm,
    format_results_for_llm,
    web_search_exa,
)


@pytest.fixture(autouse=True)
def _reset_module_client():
    # web_search now reuses a module-level AsyncClient bound to the event loop it
    # was created in; drop it around each test so a client from a prior test's
    # (now-closed) loop isn't reused. Mirrors test_openai_client.
    web_search._CLIENT = None
    yield
    web_search._CLIENT = None


# Apply asyncio mark only to coroutine tests below; sync ones don't need it.


def _make_exa_response(results: list[dict], cost: float = 0.01) -> httpx.Response:
    body = {"results": results, "costDollars": cost}
    return httpx.Response(200, json=body, request=httpx.Request("POST", "https://api.exa.ai/search"))


async def test_web_search_empty_query_raises():
    with pytest.raises(WebSearchError, match="empty query"):
        await web_search_exa("", api_key="k")
    with pytest.raises(WebSearchError, match="empty query"):
        await web_search_exa("   ", api_key="k")


async def test_web_search_missing_key_raises(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(WebSearchError, match="EXA_API_KEY not set"):
        await web_search_exa("test query")


async def test_web_search_returns_parsed_results():
    async def fake_post(self, url, headers=None, json=None, timeout=None):
        return _make_exa_response([
            {
                "title": "test title",
                "url": "https://example.com",
                "summary": "test summary",
                "highlights": ["highlight one", "highlight two"],
            },
        ], cost=0.005)

    with patch("httpx.AsyncClient.post", new=fake_post):
        result = await web_search_exa("Windows Server 2025 update", api_key="k")
    assert result["query"] == "Windows Server 2025 update"
    assert len(result["results"]) == 1
    r = result["results"][0]
    assert r["title"] == "test title"
    assert r["url"] == "https://example.com"
    assert r["summary"] == "test summary"
    assert r["highlights"] == ["highlight one", "highlight two"]
    assert result["cost_dollars"] == 0.005
    assert result["latency_ms"] >= 0


async def test_web_search_http_error_raises():
    async def fake_post(self, url, headers=None, json=None, timeout=None):
        return httpx.Response(
            429, text="rate limited",
            request=httpx.Request("POST", "https://api.exa.ai/search"),
        )

    with patch("httpx.AsyncClient.post", new=fake_post):
        with pytest.raises(WebSearchError, match="http 429"):
            await web_search_exa("q", api_key="k")


async def test_web_search_network_error_raises():
    async def fake_post(self, url, headers=None, json=None, timeout=None):
        raise httpx.ConnectError("conn refused")

    with patch("httpx.AsyncClient.post", new=fake_post):
        with pytest.raises(WebSearchError, match="network error"):
            await web_search_exa("q", api_key="k")


def test_format_results_for_llm_includes_query_and_items():
    result = {
        "query": "Hyper-V TSO bug",
        "results": [
            {
                "title": "Article 1",
                "url": "https://example.com/1",
                "summary": "S1",
                "highlights": ["H1", "H2"],
            },
            {
                "title": "Article 2",
                "url": "https://example.com/2",
                "summary": "",
                "highlights": [],
            },
        ],
        "latency_ms": 850,
    }
    out = format_results_for_llm(result)
    assert "Hyper-V TSO bug" in out
    assert "Article 1" in out
    assert "Article 2" in out
    assert "https://example.com/1" in out
    assert "H1" in out
    # Latency block is present.
    assert "850" in out


def test_format_results_for_llm_empty():
    out = format_results_for_llm({"query": "q", "results": [], "latency_ms": 100})
    assert "no results" in out.lower()


def test_format_error_for_llm():
    out = format_error_for_llm("rate limited", "test query")
    assert "rate limited" in out
    assert "test query" in out
    assert "try a different query" in out.lower() or "proceed without" in out.lower()


def test_tool_spec_shape():
    """Sanity-check the tool spec we'll inject into council payloads."""
    assert WEB_SEARCH_TOOL_SPEC["type"] == "function"
    fn = WEB_SEARCH_TOOL_SPEC["function"]
    assert fn["name"] == "web_search"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["required"] == ["query"]


# --- Malformed provider tool_calls must stay fail-soft -----------------------


@pytest.mark.asyncio
async def test_tool_loop_survives_malformed_tool_calls(monkeypatch):
    """`tool_calls=[None]` / a scalar / a non-object `function` used to raise
    AttributeError in the loop right after the (safe) executor, killing the whole
    member instead of degrading to a tool-call error."""
    from web_search_tool import run_with_tool_loop

    member = {"id": "m1", "model": "M1", "base_url": "u"}
    turns = {"n": 0}

    async def fake_call(**kwargs):
        turns["n"] += 1
        if turns["n"] == 1:
            return {"content": None, "tool_calls": [None, "oops", {"function": 5}],
                    "tokens_in": 1, "tokens_out": 1, "attempts": 1}
        return {"content": "final answer", "tool_calls": None,
                "tokens_in": 1, "tokens_out": 1, "attempts": 1}

    # No exception: the malformed payload is logged and the turn ends (the caller
    # then marks the member "no final content"), instead of an AttributeError
    # propagating out of the fan-out.
    result, log = await run_with_tool_loop(
        member=member, api_key="k", messages=[{"role": "user", "content": "q"}],
        max_tokens=100, call_fn=fake_call,
    )
    assert result["content"] is None
    assert len(log) == 3 and all(e["ok"] is False for e in log)
    assert all("malformed" in e["error"] for e in log)


@pytest.mark.asyncio
async def test_tool_loop_accepts_non_list_tool_calls(monkeypatch):
    """A single tool_call object (not wrapped in a list) is normalized, executed
    and answered instead of iterating the dict's keys."""
    import web_search_tool
    from web_search_tool import run_with_tool_loop

    async def fake_search(query):
        return {"query": query, "cost_dollars": 0.0, "latency_ms": 1,
                "results": [{"url": "https://a.example", "title": "t",
                             "summary": "x", "highlights": []}]}

    monkeypatch.setattr(web_search_tool, "web_search_exa", fake_search)
    member = {"id": "m1", "model": "M1", "base_url": "u"}
    turns = {"n": 0}

    async def fake_call(**kwargs):
        turns["n"] += 1
        if turns["n"] == 1:
            return {"content": None, "tokens_in": 1, "tokens_out": 1, "attempts": 1,
                    "tool_calls": {"id": "1", "function": {
                        "name": "web_search", "arguments": '{"query": "q"}'}}}
        return {"content": "done", "tool_calls": None,
                "tokens_in": 1, "tokens_out": 1, "attempts": 1}

    result, log = await run_with_tool_loop(
        member=member, api_key="k", messages=[{"role": "user", "content": "q"}],
        max_tokens=100, call_fn=fake_call,
    )
    assert result["content"] == "done"
    assert log[0]["ok"] is True and log[0]["name"] == "web_search"
