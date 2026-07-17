"""Shared budget / deadline manager for Council and Dialogue runs.

Before this module the codebase had many SCATTERED point limits (MAX_ROUNDS,
MAX_TOOL_ITERATIONS, MAX_RUN_SEARCHES, token clamps, connection semaphore) and
only POST-HOC usage accounting — nothing bounded a run's wall-time or dollar
spend, and there was no up-front estimate. `RunBudget` unifies the run-scoped
ceilings a caller actually wants to cap:

  * wall-time deadline (graceful — checked between rounds, never mid-fan-out so
    an in-flight round always finishes and its answers are kept);
  * max billable web searches (feeds RunSearchCache);
  * max LLM calls and max reference-PAYG dollars (checked between rounds).

`estimate_run` gives a dry-run estimate (calls / tokens / dollars / minutes)
BEFORE committing to a 2-8 minute run, so the caller sees the cost first.

Deliberately NOT a per-provider scheduler (that was reviewed and rejected as
over-engineering for this pet-project scale) — it's a run-scoped ceiling object
shared by both orchestrators, checked at round boundaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Rough per-call wall-time for a thinking-style member on OCG (cold reasoning
# models hold the connection 30-120s). Used only for the dry-run TIME estimate,
# not for enforcement — enforcement is the real monotonic deadline.
_EST_SECONDS_PER_CALL = 45.0
# Rough tokens per call (in+out) for a substantive council answer — estimate only.
_EST_TOKENS_IN_PER_CALL = 1500
_EST_TOKENS_OUT_PER_CALL = 1200


class BudgetExceeded(RuntimeError):
    """Raised/collected when a run crosses a configured ceiling. `reason` names
    which ceiling and `partial` signals the run kept its work so far."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class RunBudget:
    """Run-scoped ceilings. All fields optional — None disables that ceiling.

    Created at run start (records a monotonic clock). Orchestrators call
    `check_deadline()` / `check(usage)` at round boundaries; a returned reason
    string means "stop gracefully, keep what you have".
    """

    deadline_seconds: float | None = None
    max_llm_calls: int | None = None
    max_web_searches: int | None = None
    max_cost_usd: float | None = None
    # Set on construction; overridable for deterministic tests.
    started_monotonic: float | None = None

    def __post_init__(self) -> None:
        if self.started_monotonic is None:
            self.started_monotonic = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - (self.started_monotonic or time.monotonic())

    def remaining_seconds(self) -> float | None:
        if self.deadline_seconds is None:
            return None
        return self.deadline_seconds - self.elapsed_seconds()

    def deadline_expired(self) -> bool:
        rem = self.remaining_seconds()
        return rem is not None and rem <= 0

    def check(self, usage: dict | None = None) -> str | None:
        """Return a stop-reason string if any ceiling is crossed, else None.

        `usage` is the council usage dict so far (llm_calls / web_search_calls /
        reference_payg_cost_usd). Cheap to call at each round boundary.
        """
        if self.deadline_expired():
            return f"wall-time deadline reached ({self.deadline_seconds:.0f}s)"
        if usage:
            if self.max_llm_calls is not None and (usage.get("llm_calls") or 0) >= self.max_llm_calls:
                return f"max LLM calls reached ({self.max_llm_calls})"
            if self.max_web_searches is not None and (usage.get("web_search_calls") or 0) >= self.max_web_searches:
                return f"max web searches reached ({self.max_web_searches})"
            if self.max_cost_usd is not None:
                cost = usage.get("reference_payg_cost_usd") or 0.0
                if cost >= self.max_cost_usd:
                    return f"max reference-PAYG cost reached (${self.max_cost_usd:.4f})"
        return None

    def as_dict(self) -> dict:
        return {
            "deadline_seconds": self.deadline_seconds,
            "max_llm_calls": self.max_llm_calls,
            "max_web_searches": self.max_web_searches,
            "max_cost_usd": self.max_cost_usd,
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "remaining_seconds": (
                round(self.remaining_seconds(), 1)
                if self.remaining_seconds() is not None else None
            ),
        }


def estimate_run(
    *,
    n_members: int,
    rounds: int = 1,
    synthesis: bool = False,
    web_search: bool = False,
    price_in: float | None = None,
    price_out: float | None = None,
) -> dict:
    """Dry-run estimate for a council run, BEFORE it starts.

    Per round: n stage-1 + n stage-2 calls; ×rounds; +1 synthesis. web_search can
    add up to a few extra stage-1 turns per member (estimated at +2 per member per
    round). Returns expected calls / tokens / minutes and, when a per-token
    reference price is given, a reference-PAYG dollar estimate (NOT billed spend —
    council members are flat-rate; this mirrors usage.reference_payg_cost_usd).
    """
    per_round = 2 * n_members
    web_extra = (2 * n_members * rounds) if web_search else 0
    calls = per_round * rounds + (1 if synthesis else 0) + web_extra
    tokens_in = calls * _EST_TOKENS_IN_PER_CALL
    tokens_out = calls * _EST_TOKENS_OUT_PER_CALL
    # Stage 1 and stage 2 within a round run concurrently, so wall-time ≈ the
    # sequential sum of round latencies, not the per-call sum.
    est_minutes = round((_EST_SECONDS_PER_CALL * 2 * rounds + (_EST_SECONDS_PER_CALL if synthesis else 0)) / 60.0, 1)
    ref_cost = None
    if price_in is not None or price_out is not None:
        ref_cost = round(
            tokens_in * (price_in or 0.0) / 1_000_000
            + tokens_out * (price_out or 0.0) / 1_000_000,
            6,
        )
    return {
        "expected_llm_calls": calls,
        "expected_tokens_in": tokens_in,
        "expected_tokens_out": tokens_out,
        "expected_minutes": est_minutes,
        "reference_payg_cost_usd": ref_cost,
        "note": (
            "Rough dry-run estimate. Council members bill flat-rate — "
            "reference_payg_cost_usd is a yardstick, not billed spend."
        ),
    }
