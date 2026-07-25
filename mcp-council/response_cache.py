"""Opt-in council response cache with provenance and singleflight.

A council run is expensive (2-8 min, many LLM calls). When the SAME question is
asked with the SAME council configuration, recomputing is pure waste. This is an
OPT-IN (``cache=True``) LRU+TTL cache for the final markdown brief, with:

  * **fingerprint key** over everything that changes the answer — question,
    sorted model ids + their catalog config, synthesis, rounds, web_search,
    max_tokens, and a context-files fingerprint — so a changed catalog/config
    never serves a stale answer;
  * **per-entry byte cap** — an oversized brief is not cached (mirrors the
    agent-server cache guard);
  * **singleflight** — concurrent identical misses collapse to ONE run instead
    of firing N parallel 2-8 min councils;
  * **provenance** — every hit carries cached_at / fingerprint / age so the
    consumer can see the answer was reused, not recomputed;
  * **privacy mode** — in-memory only, never touches disk; a process restart
    drops it (deliberate — a council brief can contain sensitive context).

Off by default: correctness of a fresh deliberation always wins unless the
caller explicitly asks to trade freshness for speed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

DEFAULT_TTL_SECONDS = 3600.0
DEFAULT_MAX_ENTRIES = 64
# A single brief past this size is not cached (a huge multi-round web_search brief
# would otherwise pin megabytes). Mirrors the agent-server cache's oversized guard.
DEFAULT_MAX_ENTRY_BYTES = 512 * 1024


def fingerprint(
    *,
    question: str,
    model_configs: list[dict],
    synthesis: bool,
    rounds: int,
    web_search: bool,
    max_tokens: int,
    context_fingerprint: str = "",
    context_in_stage2: bool = True,
    adaptive: bool = False,
    budget: dict | None = None,
) -> str:
    """Stable sha256 over the RESOLVED execution spec — everything that changes
    the council answer, after defaults are applied.

    `model_configs` are the resolved member dicts; we hash id + model + base_url
    + provider + extra + the effective min_max_tokens (a catalog edit, a changed
    quirk or a raised per-model floor must invalidate the entry rather than serve
    a stale brief).

    Member ORDER is part of the key. It is NOT sorted away: stage-1 order decides
    each ranker's pseudonym assignment and the presentation order of the answers
    it ranks, so two runs with the same members in a different order are two
    different deliberations that can pick different winners.

    `context_in_stage2`, `adaptive` and `budget` (deadline / cost / search /
    call ceilings) are hashed too — they were previously absent, so a
    budget-stopped 2-model adaptive run could be served for a full, unbounded,
    context-carrying one.
    """
    members = [
        {
            "id": m.get("id"),
            "model": m.get("model"),
            "base_url": m.get("base_url"),
            "provider": m.get("provider"),
            "extra": m.get("extra"),
            "min_max_tokens": m.get("min_max_tokens"),
        }
        for m in model_configs
    ]
    material = json.dumps(
        {
            "q": question,
            "members": members,
            "synthesis": synthesis,
            "rounds": rounds,
            "web_search": web_search,
            "max_tokens": max_tokens,
            "ctx": context_fingerprint,
            "context_in_stage2": context_in_stage2,
            "adaptive": adaptive,
            "budget": budget or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass
class _Entry:
    value: str
    cached_at: float
    fingerprint: str


@dataclass
class ResponseCache:
    """In-memory LRU+TTL cache with singleflight. One per process (opt-in)."""

    ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES
    _store: "OrderedDict[str, _Entry]" = field(default_factory=OrderedDict, repr=False)
    _inflight: dict[str, asyncio.Future] = field(default_factory=dict, repr=False)
    hits: int = 0
    misses: int = 0

    def _get_fresh(self, key: str) -> _Entry | None:
        e = self._store.get(key)
        if e is None:
            return None
        if time.time() - e.cached_at > self.ttl_seconds:
            # Expired — drop it so a later put re-inserts cleanly.
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)  # LRU touch
        return e

    def provenance(self, key: str) -> dict | None:
        """Provenance metadata for a currently-cached key (None if absent)."""
        e = self._get_fresh(key)
        if e is None:
            return None
        return {
            "cached": True,
            "cached_at": e.cached_at,
            "age_seconds": round(time.time() - e.cached_at, 1),
            "fingerprint": e.fingerprint,
        }

    def _put(self, key: str, value: str) -> None:
        if len(value.encode("utf-8", "surrogatepass")) > self.max_entry_bytes:
            # Oversized — never cache; also drop any stale entry under this key.
            self._store.pop(key, None)
            return
        self._store[key] = _Entry(value=value, cached_at=time.time(), fingerprint=key)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # evict LRU

    async def get_or_compute(self, key: str, compute):
        """Return the cached value for `key`, or run `compute()` (an async
        zero-arg callable) exactly once even under concurrent identical misses.

        Returns (value, provenance_dict_or_None). provenance is non-None on a hit.
        A compute() exception propagates to every waiter and is NOT cached.
        """
        while True:
            prov = self.provenance(key)
            if prov is not None:
                self.hits += 1
                return self._store[key].value, prov

            # Singleflight: if an identical run is already in flight, await it.
            inflight = self._inflight.get(key)
            if inflight is None:
                break
            try:
                value = await inflight
            except asyncio.CancelledError:
                if not inflight.cancelled():
                    raise  # THIS task was cancelled — propagate our own.
                # The owner was cancelled; a waiter must not inherit that. The
                # owner already removed the key, so looping round takes the owner
                # path (or picks up a newer in-flight run) and computes for real.
                # A loop, not recursion: a key whose owners keep getting
                # cancelled would otherwise grow the stack one frame per retry.
                continue
            # Served off a concurrent computation — count as a hit and attach
            # provenance if the winner cached it.
            self.hits += 1
            return value, (self.provenance(key) or {"cached": True, "coalesced": True})

        self.misses += 1
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._inflight[key] = fut
        try:
            value = await compute()
            self._put(key, value)
            # Retire the in-flight entry BEFORE waking the waiters. Waking first
            # would let a request that arrives between set_result and the pop see
            # a completed-but-still-registered future (or, on the failure path, a
            # future that already carries the exception) and either await a dead
            # entry or start a redundant compute. Popping first means a newcomer
            # sees either the fresh _store entry or a clean miss.
            self._inflight.pop(key, None)
            if not fut.done():
                fut.set_result(value)
            return value, None
        except BaseException as e:
            # BaseException, not Exception: asyncio.CancelledError inherits from
            # BaseException, so an `except Exception` cleanup NEVER ran when the
            # OWNER of the singleflight was cancelled (client disconnect, timeout,
            # task cancellation). The key then stayed in `_inflight` holding a
            # future nobody would ever complete, and every later request for that
            # question awaited it forever — a permanently poisoned key, curable
            # only by restarting the process.
            self._inflight.pop(key, None)
            if not fut.done():
                if isinstance(e, asyncio.CancelledError):
                    # Waiters must not inherit the owner's cancellation as their
                    # own — cancel the shared future so each waiter raises
                    # CancelledError at its own await and can retry cleanly.
                    fut.cancel()
                else:
                    fut.set_exception(e)
                    # Mark it retrieved: with no concurrent waiter nobody ever
                    # awaits this future, and asyncio would log a spurious
                    # "Future exception was never retrieved" traceback for an
                    # error the caller already received by `raise` below. Real
                    # waiters still get it raised.
                    fut.exception()
            raise

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }

    def clear(self) -> None:
        self._store.clear()
