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
