"""Adaptive provider-aware council selection (pure helpers).

The default council is a static 7-member list. `adaptive.py` adds three pure
decisions the orchestrator (council.run_adaptive_council) composes into a
health-aware, start-small-escalate flow:

  * `filter_healthy` — drop members a pre-flight healthcheck flags as broken
    (no_key / auth / insufficient_balance / circuit_open / …), keeping the ones
    that answered. A model that is down should not silently take a fan-out slot.
  * `pick_starting_subset` — choose a SMALL diverse starting council that spans
    ≥2 provider (failure) domains, so the first pass is cheap yet corroboratable.
  * `should_escalate` — decide, from the first pass's summary, whether to add the
    held-back members (low quorum / low agreement / high disagreement) or stop.

Pure and dependency-light so it unit-tests without network. The health status
strings mirror healthcheck.py's classifier.
"""

from __future__ import annotations

from models import provider_domain

# Health statuses that mean "usable now". Everything else (no_key, auth,
# insufficient_balance, rate_limited, timeout, network, circuit_open, disabled,
# empty_response, error) drops the member from the adaptive council.
_HEALTHY_STATUSES = frozenset({"ok"})


def filter_healthy(
    member_ids: list[str], health_rows: list[dict]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split member_ids into (kept, dropped) using healthcheck rows.

    `health_rows` = healthcheck_models output ({"id","status",...}). A member with
    no row (health unknown) is KEPT — absence of evidence isn't evidence of a
    fault, and dropping it would over-shrink the council on a partial probe.
    Returns (kept_ids, [(dropped_id, status), ...]); kept preserves input order.
    """
    status_by_id = {r["id"]: r.get("status") for r in health_rows if "id" in r}
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for mid in member_ids:
        st = status_by_id.get(mid)
        if st is None or st in _HEALTHY_STATUSES:
            kept.append(mid)
        else:
            dropped.append((mid, st))
    return kept, dropped


def pick_starting_subset(
    members: list[dict], *, min_size: int = 3, min_domains: int = 2
) -> list[str]:
    """Pick a small starting council that maximizes provider-domain diversity.

    Greedy: walk members in order, always preferring an id from a not-yet-covered
    provider domain, until we have ≥min_size members AND ≥min_domains domains (or
    we run out). Returns member ids. With <min_size members total, returns them
    all. Order-stable given the input order (deterministic)."""
    if len(members) <= min_size:
        return [m["id"] for m in members]
    chosen: list[str] = []
    domains: set[str] = set()
    # First pass: one member per fresh domain (diversity first).
    for m in members:
        d = provider_domain(m["id"])
        if d not in domains:
            chosen.append(m["id"])
            domains.add(d)
        if len(chosen) >= min_size and len(domains) >= min_domains:
            return chosen
    # Second pass: top up to min_size with remaining members (any domain).
    if len(chosen) < min_size:
        for m in members:
            if m["id"] not in chosen:
                chosen.append(m["id"])
                if len(chosen) >= min_size:
                    break
    return chosen


def should_escalate(summary: dict | None) -> tuple[bool, str]:
    """Decide whether to add the held-back members after the first pass.

    Escalate when the cheap first pass did NOT reach a corroborated, confident
    verdict: no quorum, low agreement, an unresolved high-risk topic needing more
    independent eyes, or a live top disagreement. Returns (escalate, reason)."""
    if not summary:
        return True, "no summary from first pass"
    if not summary.get("quorum_ok"):
        return True, "first pass not independently corroborated (no quorum)"
    if summary.get("agreement_confidence") == "low":
        return True, "low agreement in first pass"
    if summary.get("human_review_required"):
        return True, "first pass flagged human_review_required"
    if summary.get("top_disagreements"):
        return True, "unresolved disagreement in first pass"
    return False, "first pass reached a corroborated, confident verdict"
