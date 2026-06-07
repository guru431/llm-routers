"""Provider/model healthcheck for mcp-council.

Pings each catalog model with a trivial prompt so key / HTTP / latency /
empty-response problems surface explicitly via a tool call instead of only
showing up mid-council. Pure-logic core (call_fn injectable) — unit-testable
without network.
"""
from __future__ import annotations

import asyncio
import os
import time
from urllib.parse import urlparse

from models import CATALOG, UnknownModelError
from openai_client import CouncilHTTPError, call_openai_compat

# A trivial prompt — we only care that the provider answers, not what it says.
_PING_MESSAGES = [{"role": "user", "content": "Reply with the single word: pong"}]

# Per-check ceiling. Healthcheck should be fast; a real model that needs minutes
# is itself a signal worth surfacing as a timeout.
DEFAULT_TIMEOUT = 25.0


def _classify_error(msg: str) -> str:
    """Map a CouncilHTTPError message to a coarse status for at-a-glance triage."""
    m = msg.lower()
    if "circuit_open" in m:
        return "circuit_open"
    if "402" in m or "insufficient_balance" in m:
        return "insufficient_balance"
    if "401" in m or "403" in m or " auth" in m:
        return "auth"
    if "429" in m or "overload" in m or "rate" in m:
        return "rate_limited"
    if "timeout" in m:
        return "timeout"
    if "empty content" in m:
        return "empty_response"
    if "network error" in m:
        return "network"
    return "error"


def _provider(base_url: str) -> str:
    return urlparse(base_url).netloc or base_url


async def _check_one(mid: str, cfg: dict, call_fn, timeout: float) -> dict:
    model = cfg["model"]
    base = {
        "id": mid,
        "model": model,
        "provider": _provider(cfg.get("base_url", "")),
    }
    if cfg.get("enabled") is False:
        return {**base, "enabled": False, "key_present": None, "ok": False,
                "status": "disabled", "latency_ms": None, "error": None}

    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        return {**base, "enabled": True, "key_present": False, "ok": False,
                "status": "no_key", "latency_ms": None,
                "error": f"env var {cfg['env_key']} not set"}

    start = time.monotonic()
    try:
        result = await call_fn(
            base_url=cfg["base_url"],
            api_key=api_key,
            model=model,
            messages=_PING_MESSAGES,
            max_tokens=max(64, cfg.get("min_max_tokens", 0)),
            extra_payload=cfg.get("extra"),
            timeout=timeout,
        )
    except CouncilHTTPError as e:
        return {**base, "enabled": True, "key_present": True, "ok": False,
                "status": _classify_error(str(e)),
                "latency_ms": int((time.monotonic() - start) * 1000),
                "error": str(e)}

    return {**base, "enabled": True, "key_present": True, "ok": True,
            "status": "ok", "latency_ms": int((time.monotonic() - start) * 1000),
            "error": None, "tokens_out": result.get("tokens_out")}


async def healthcheck_models(
    ids: list[str] | None = None,
    *,
    call_fn=None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Ping each model id (None → every CATALOG entry, incl. disabled).

    Runs all checks concurrently. Raises UnknownModelError on an unknown id.
    Disabled models are reported with status="disabled" rather than skipped.
    """
    call_fn = call_fn or call_openai_compat
    if ids is None:
        ids = list(CATALOG.keys())
    for i in ids:
        if i not in CATALOG:
            raise UnknownModelError(
                f"unknown model_id: '{i}'. Available: {sorted(CATALOG.keys())}"
            )
    checks = [_check_one(i, CATALOG[i], call_fn, timeout) for i in ids]
    return list(await asyncio.gather(*checks))
