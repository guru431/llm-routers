"""Regressions for the response-cache findings resolved on 2026-07-25:
owner cancellation poisoning the singleflight key, and a fingerprint that didn't
describe the actual council execution spec.
"""

import asyncio

import pytest

import response_cache
from budget import RunBudget


@pytest.mark.asyncio
async def test_owner_cancellation_does_not_poison_key():
    """CancelledError inherits from BaseException, so the old `except Exception`
    cleanup never ran: the key stayed in _inflight holding a future nobody would
    complete, and every later request for it hung until process restart."""
    cache = response_cache.ResponseCache()
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(60)
        return "never"

    owner = asyncio.create_task(cache.get_or_compute("k", slow))
    await started.wait()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    assert "k" not in cache._inflight

    async def fast():
        return "fresh"

    value, prov = await asyncio.wait_for(cache.get_or_compute("k", fast), timeout=2)
    assert value == "fresh" and prov is None


@pytest.mark.asyncio
async def test_waiter_recomputes_when_owner_is_cancelled():
    """A waiter must not inherit the owner's cancellation as its own failure."""
    cache = response_cache.ResponseCache()
    owner_started = asyncio.Event()
    calls = {"n": 0}

    async def slow():
        calls["n"] += 1
        owner_started.set()
        await asyncio.sleep(60)
        return "never"

    async def compute():
        # The waiter retries with the same callable; second invocation is fast.
        if calls["n"] >= 1:
            calls["n"] += 1
            return "recomputed"
        return await slow()

    owner = asyncio.create_task(cache.get_or_compute("k", slow))
    await owner_started.wait()
    waiter = asyncio.create_task(cache.get_or_compute("k", compute))
    await asyncio.sleep(0)  # let the waiter register on the in-flight future
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    value, _prov = await asyncio.wait_for(waiter, timeout=2)
    assert value == "recomputed"


def _base_kwargs():
    return dict(
        question="q",
        model_configs=[
            {"id": "glm", "model": "glm-5.2", "base_url": "u", "provider": "ocg"},
            {"id": "gemini", "model": "gemini", "base_url": "v", "provider": "hel"},
        ],
        synthesis=False,
        rounds=1,
        web_search=False,
        max_tokens=8192,
    )


@pytest.mark.parametrize("field,value", [
    ("context_in_stage2", False),
    ("adaptive", True),
    ("budget", {"max_llm_calls": 3}),
])
def test_fingerprint_changes_for_each_semantic_flag(field, value):
    """Each of these changes what the council actually did — a cached answer for
    one is not a valid answer for the other."""
    base = _base_kwargs()
    assert response_cache.fingerprint(**base) != response_cache.fingerprint(
        **{**base, field: value}
    )


def test_fingerprint_changes_with_member_order():
    """Stage-1 order decides pseudonym assignment and the order answers are
    presented for ranking, so it can change the winner. The old key sorted
    members and served one run's answer for the other."""
    base = _base_kwargs()
    reversed_members = list(reversed(base["model_configs"]))
    assert response_cache.fingerprint(**base) != response_cache.fingerprint(
        **{**base, "model_configs": reversed_members}
    )


def test_fingerprint_changes_with_effective_min_max_tokens():
    base = _base_kwargs()
    bumped = [dict(base["model_configs"][0], min_max_tokens=32768),
              base["model_configs"][1]]
    assert response_cache.fingerprint(**base) != response_cache.fingerprint(
        **{**base, "model_configs": bumped}
    )


def test_budget_cache_key_excludes_the_clock():
    """as_dict() carries elapsed/remaining seconds; hashing those would make every
    key unique and silently disable the cache."""
    b = RunBudget(max_llm_calls=5, deadline_seconds=60)
    key = b.as_dict_key()
    assert key == {"deadline_seconds": 60, "max_llm_calls": 5,
                   "max_web_searches": None, "max_cost_usd": None}
    assert "elapsed_seconds" not in key
