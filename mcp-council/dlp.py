"""Outbound DLP for model-generated web_search queries + claim→source ledger.

Two independent concerns, both about the web_search side channel:

1. **Outbound query DLP** (`scrub_outbound_query`). A council member composes its
   own Exa query. Context files are already gated on the INPUT side (sandbox.py),
   but a model that saw a secret (in context, in a peer answer, in its own
   reasoning) could echo it into a search query — shipping it to a third-party
   search API. `scrub_outbound_query` inspects the query BEFORE it leaves the
   process and blocks it when it carries a credential-shaped token, a private-key
   header, or an obvious sensitive absolute path. Blocked queries never hit Exa;
   the model receives a refusal string and can retry with a clean query.

2. **Claim→source ledger** (`build_claim_ledger`). The web-search transparency
   section lists WHAT was searched; the ledger goes one step further and records,
   per search, the SOURCES (result URLs) that back it, plus provenance (which
   member issued it, when, cost). It is not a per-sentence citation graph — it
   maps each executed query to the sources returned, so a reader can trace which
   external pages fed the deliberation.

Dependency-free of council/orchestrator code so it unit-tests on its own.
"""

from __future__ import annotations

import re

# --- Outbound query DLP ----------------------------------------------------

# Credential-shaped tokens. These mirror the formats the pre-commit secret guard
# and the vault care about; the goal is to catch a secret a model pasted into a
# query, not to be a universal secret scanner. Ordered rough-specific → generic.
_SECRET_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("openai/anthropic key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("openrouter user id", re.compile(r"\buser_[A-Za-z0-9]{24,}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    ("bearer header", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}\b")),
    ("private key header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("generic api_key=…", re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\s*[=:]\s*\S{8,}")),
)

# Sensitive absolute paths a model should never be searching the web for — these
# are local credential stores, not public topics. Substring match on a normalized
# (forward-slash, lower) query.
_SENSITIVE_PATH_SEGMENTS = (
    "/.ssh/", "/.aws/", "/.gcp/", "/.azure/", "/.kube/",
    "/.config/gcloud/", "/.docker/config", "secrets/vault", "vault.env",
    "id_rsa", "id_ed25519", ".git-credentials", ".pgpass", ".netrc",
)


class OutboundBlocked(Exception):
    """Raised (or signalled via return) when a query must not leave the process."""


def scrub_outbound_query(query: str) -> tuple[str | None, str | None]:
    """Inspect a model-generated search query before it hits Exa.

    Returns ``(safe_query, None)`` when the query is clean, or ``(None, reason)``
    when it must be blocked. The reason is a short human string (also fed back to
    the model) naming WHAT tripped the guard, never echoing the secret itself.
    """
    if not query or not query.strip():
        return query, None
    for label, pat in _SECRET_PATTERNS:
        if pat.search(query):
            return None, f"blocked: query contains a {label}-shaped secret"
    norm = query.replace("\\", "/").lower()
    for seg in _SENSITIVE_PATH_SEGMENTS:
        if seg in norm:
            return None, f"blocked: query references a sensitive local path ({seg})"
    return query, None


# --- Claim→source ledger ---------------------------------------------------


def build_claim_ledger(records: list[dict]) -> list[dict]:
    """Build a query→sources ledger from council member records' tool_calls_log.

    Each record is a stage-1 member / chairman dict carrying `tool_calls_log`
    (see web_search_tool.execute_tool_call). Only successful web_search entries
    that captured result URLs contribute. Returns a list of::

        {"member": str, "query": str, "num_results": int,
         "sources": [str, ...], "cost_dollars": float | None}

    De-dups identical (member, query) pairs, keeping the first. Empty when no
    search captured sources (e.g. web_search disabled, or an older log without
    the `sources` field).
    """
    ledger: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        member = rec.get("model") or rec.get("chairman_model") or rec.get("id") or "?"
        for entry in rec.get("tool_calls_log") or []:
            if not isinstance(entry, dict) or entry.get("name") != "web_search":
                continue
            if not entry.get("ok"):
                continue
            query = (entry.get("query") or "").strip()
            sources = [s for s in (entry.get("sources") or []) if s]
            if not query or not sources:
                continue
            key = (str(member), query.lower())
            if key in seen:
                continue
            seen.add(key)
            ledger.append({
                "member": member,
                "query": query,
                "num_results": entry.get("num_results"),
                "sources": sources,
                "cost_dollars": entry.get("cost_dollars"),
            })
    return ledger
