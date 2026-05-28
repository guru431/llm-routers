"""Tests for web_search.py (Exa client) and the tool-loop integration."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from web_search import (
    WEB_SEARCH_TOOL_SPEC,
    WebSearchError,
    format_error_for_llm,
    format_results_for_llm,
    web_search_exa,
)


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
    async def fake_post(self, url, headers=None, json=None):
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
    async def fake_post(self, url, headers=None, json=None):
        return httpx.Response(
            429, text="rate limited",
            request=httpx.Request("POST", "https://api.exa.ai/search"),
        )

    with patch("httpx.AsyncClient.post", new=fake_post):
        with pytest.raises(WebSearchError, match="http 429"):
            await web_search_exa("q", api_key="k")


async def test_web_search_network_error_raises():
    async def fake_post(self, url, headers=None, json=None):
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
