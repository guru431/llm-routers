"""Tests for RunSearchCache — per-run web_search dedup by normalized query."""
import asyncio

import pytest

from web_search_tool import RunSearchCache


def test_dedup_and_normalization():
    calls = []

    async def fake_search(query):
        calls.append(query)
        return {"query": query, "results": [], "latency_ms": 1}

    async def run():
        cache = RunSearchCache(search_fn=fake_search)
        await cache.search("Python  GIL")
        await cache.search("python gil")        # normalized → same key
        await cache.search("  PYTHON   gil ")    # normalized → same key
        await cache.search("rust async")         # distinct
        return cache

    cache = asyncio.run(run())
    assert len(calls) == 2          # only two distinct underlying searches
    assert cache.misses == 2
    assert cache.hits == 2


def test_concurrent_identical_queries_collapse_to_one_call():
    started = 0

    async def slow_search(query):
        nonlocal started
        started += 1
        await asyncio.sleep(0.02)
        return {"query": query, "results": [], "latency_ms": 1}

    async def run():
        cache = RunSearchCache(search_fn=slow_search)
        # 5 concurrent identical queries must trigger exactly one search.
        results = await asyncio.gather(*(cache.search("same q") for _ in range(5)))
        return cache, results

    cache, results = asyncio.run(run())
    assert started == 1
    assert cache.misses == 1
    assert cache.hits == 4
    assert all(r["query"] == "same q" for r in results)


def test_run_budget_exhausted_raises():
    # F13: a run-wide cap on distinct (billed) searches. New distinct queries past
    # the cap raise WebSearchError (caught upstream and surfaced to the model);
    # cached repeats stay free.
    from web_search import WebSearchError

    async def fake_search(query):
        return {"query": query, "results": [], "latency_ms": 1}

    async def run():
        cache = RunSearchCache(search_fn=fake_search, max_searches=2)
        await cache.search("q1")
        await cache.search("q2")
        await cache.search("q1")  # cache hit — still allowed past the cap
        with pytest.raises(WebSearchError, match="budget exhausted"):
            await cache.search("q3")  # new distinct query → over budget

    asyncio.run(run())


def test_execute_tool_call_rejects_malformed_shapes():
    # F13: a malformed tool_calls array (non-dict entry, non-object args) must
    # return a tool-error message, not raise AttributeError and kill the member.
    from web_search_tool import execute_tool_call

    def noop(*a, **k):
        return None

    async def run():
        # tc is a bare scalar
        msg, log = await execute_tool_call(1, noop, "m")
        assert "Malformed tool_call" in msg and log["ok"] is False
        # arguments parse to a JSON array, not an object
        tc = {"function": {"name": "web_search", "arguments": "[1, 2, 3]"}}
        msg, log = await execute_tool_call(tc, noop, "m")
        assert "must be a JSON object" in msg and log["ok"] is False

    asyncio.run(run())
