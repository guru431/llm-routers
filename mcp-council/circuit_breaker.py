"""Per-provider circuit breaker for mcp-council.

When a provider host (e.g. the OCG gateway that serves 4 council members) starts
throwing infra errors — 5xx, timeouts, exhausted-retry overloads — there's no
point hammering it with the rest of a fan-out. After N consecutive infra
failures the host is marked "open" for a cooldown window; calls to it then
short-circuit immediately instead of each spending the full retry/timeout budget.

Process-global, in-memory (the MCP server is one process). State is keyed by
host. Single-threaded asyncio means the simple int counters need no lock —
there is no await between read and write in record_*().

Scope: ONLY clear infra-outage signals trip the breaker. Request/account errors
(402 balance, 401 auth, 400 bad request) are not the provider being down, so
they never open it.
"""
from __future__ import annotations

import time

# Consecutive infra failures before a host is considered down. OCG serves 4
# default members, so a single outage that fails all 4 reaches this quickly.
FAILURE_THRESHOLD = 4
# How long a host stays open before the next call is allowed through to probe it.
COOLDOWN_SECONDS = 120.0

_state: dict[str, dict] = {}  # host -> {"fails": int, "open_until": float}


def open_for(host: str) -> float:
    """Remaining cooldown seconds if the host's breaker is open, else 0.0."""
    st = _state.get(host)
    if not st:
        return 0.0
    remaining = st["open_until"] - time.monotonic()
    return remaining if remaining > 0 else 0.0


def record_success(host: str) -> None:
    """A successful call clears any accumulated failure state for the host."""
    _state.pop(host, None)


def record_failure(host: str) -> None:
    """Count one infra failure; open the breaker once the threshold is reached."""
    st = _state.setdefault(host, {"fails": 0, "open_until": 0.0})
    st["fails"] += 1
    if st["fails"] >= FAILURE_THRESHOLD:
        st["open_until"] = time.monotonic() + COOLDOWN_SECONDS


def snapshot() -> dict[str, dict]:
    """Diagnostic view: {host: {fails, open_seconds_remaining}}."""
    out: dict[str, dict] = {}
    for host in _state:
        out[host] = {"fails": _state[host]["fails"], "open_seconds_remaining": round(open_for(host), 1)}
    return out


def reset() -> None:
    """Clear all breaker state (test hook / manual recovery)."""
    _state.clear()
