"""Council orchestrator: stage 1 (independent) → stage 2 (anonymized peer-ranking).

Stage 3 (synthesis) is delegated to the main Claude agent — this module returns
all materials, the caller formats them into markdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import string
import time
from typing import Any, Awaitable, Callable

from healthcheck import _classify_error
from models import CATALOG, COUNCIL_DEFAULT, resolve_members
from openai_client import call_openai_compat
from prompts import (
    STAGE1_ROUND_N_SYSTEM,
    STAGE1_SYSTEM,
    STAGE2_SYSTEM,
    STAGE3_SYSTEM,
    STAGE3_WEB_SEARCH_NOTE,
    build_stage1_round_n_user,
    build_stage1_user,
    build_stage2_user,
    build_stage3_user,
)
from web_search import WEB_SEARCH_TOOL_SPEC
from web_search_tool import MAX_TOOL_ITERATIONS, RunSearchCache, run_with_tool_loop

# Hard cap on debate rounds. Pet-project scale; each round adds 2-8 minutes of
# wall-time so we don't want hidden cost blow-ups.
MAX_ROUNDS = 3

# Fallback chairman id when peer-rankings can't pick one. DeepSeek-direct is the
# most stable provider in this stack (api.deepseek.com vs OCG outages).
DEFAULT_CHAIRMAN_FALLBACK_ID = "deepseek-pro"

# Type for the HTTP client function — overridable in tests via dependency injection.
CallFn = Callable[..., Awaitable[dict]]

# Type for the progress callback. Called as `progress(event_type, payload)`.
# event_type ∈ {"phase", "stage1_member", "stage2_ranker", "stage3"}.
# Synchronous (cheap state writes only) — orchestrator does NOT await it.
ProgressFn = Callable[[str, dict[str, Any]], None]


def _noop_progress(event_type: str, payload: dict[str, Any]) -> None:  # noqa: ARG001
    """Default progress sink — silently discards events."""
    return None


def _loop_usage(result: dict) -> dict:
    """Extract web_search tool-loop usage aggregates from a call result.

    run_with_tool_loop attaches loop_* keys summing tokens/calls/attempts across
    every loop iteration. Non-web calls don't have them — return {} so the
    record keeps its single-call tokens_in/out/attempts (backward compatible)."""
    keys = ("loop_calls", "loop_tokens_in", "loop_tokens_out", "loop_attempts")
    return {k: result[k] for k in keys if k in result}


async def _run_member_stage1(
    member: dict,
    question: str,
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
    *,
    web_search: bool = False,
    on_progress: ProgressFn | None = None,
    search_cache: RunSearchCache | None = None,
) -> dict:
    """Run one council member through stage 1. Always returns a dict — exceptions
    are captured into status="error" so asyncio.gather sees no raise."""
    start = time.monotonic()
    api_key = os.environ.get(member["env_key"])
    if not api_key:
        return {
            "id": member["id"],
            "model": member["model"],
            "status": "error",
            "error": f"env var {member['env_key']} not set",
            "answer": None,
            "latency_ms": 0,
            "tokens_in": None,
            "tokens_out": None,
            "tool_calls_log": [],
        }

    max_tokens = max(max_response_tokens, member.get("min_max_tokens", 0))
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": build_stage1_user(question, files_section)},
    ]
    tools = [WEB_SEARCH_TOOL_SPEC] if web_search else None
    progress = on_progress or _noop_progress
    try:
        if tools:
            result, tool_log = await run_with_tool_loop(
                member=member, api_key=api_key, messages=messages,
                max_tokens=max_tokens, call_fn=call_fn, tools=tools,
                on_progress=progress, search_cache=search_cache,
            )
        else:
            result = await call_fn(
                base_url=member["base_url"],
                api_key=api_key,
                model=member["model"],
                messages=messages,
                max_tokens=max_tokens,
                extra_payload=member.get("extra"),
            )
            tool_log = []
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Always return a dict so asyncio.gather (no return_exceptions) never
        # sees a raise: any non-cancellation failure (HTTP, KeyError on a
        # malformed result, …) becomes this member's status="error" and the
        # rest of the fan-out keeps its answers.
        return {
            "id": member["id"],
            "model": member["model"],
            "status": "error",
            "error": str(e),
            "answer": None,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": None,
            "tokens_out": None,
            "tool_calls_log": [],
        }

    # If the loop ran out of iterations and the model still wanted to call
    # tools, treat as error so the rest of the council doesn't try to rank
    # an empty answer.
    if not result.get("content"):
        return {
            "id": member["id"],
            "model": member["model"],
            "status": "error",
            "error": (
                f"no final content after {MAX_TOOL_ITERATIONS} tool iterations "
                f"(finish_reason={result.get('finish_reason')})"
            ),
            "answer": None,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": result.get("tokens_in"),
            "tokens_out": result.get("tokens_out"),
            "attempts": result.get("attempts"),
            "tool_calls_log": tool_log,
            **_loop_usage(result),
        }

    return {
        "id": member["id"],
        "model": member["model"],
        "status": "ok",
        "error": None,
        "answer": result["content"],
        "latency_ms": int((time.monotonic() - start) * 1000),
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "attempts": result.get("attempts"),
        "tool_calls_log": tool_log,
        **_loop_usage(result),
    }


async def _run_member_stage1_round_n(
    member: dict,
    question: str,
    own_previous_answer: str,
    other_answers: list[tuple[str, str]],
    rankings_digest: str,
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
) -> dict:
    """Stage 1 for round 2+: same member, new prompt that includes its prior
    answer + other answers + critique digest."""
    start = time.monotonic()
    api_key = os.environ.get(member["env_key"])
    if not api_key:
        return {
            "id": member["id"], "model": member["model"], "status": "error",
            "error": f"env var {member['env_key']} not set",
            "answer": None, "latency_ms": 0, "tokens_in": None, "tokens_out": None,
            "tool_calls_log": [],
        }

    max_tokens = max(max_response_tokens, member.get("min_max_tokens", 0))
    messages = [
        {"role": "system", "content": STAGE1_ROUND_N_SYSTEM},
        {
            "role": "user",
            "content": build_stage1_round_n_user(
                question, own_previous_answer, other_answers, rankings_digest, files_section
            ),
        },
    ]
    try:
        result = await call_fn(
            base_url=member["base_url"],
            api_key=api_key,
            model=member["model"],
            messages=messages,
            max_tokens=max_tokens,
            extra_payload=member.get("extra"),
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Always return a dict (see _run_member_stage1): any non-cancellation
        # failure becomes status="error" so gather doesn't abort the fan-out.
        return {
            "id": member["id"], "model": member["model"], "status": "error",
            "error": str(e), "answer": None,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": None, "tokens_out": None,
            "tool_calls_log": [],
        }

    return {
        "id": member["id"], "model": member["model"], "status": "ok",
        "error": None, "answer": result["content"],
        "latency_ms": int((time.monotonic() - start) * 1000),
        "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"],
        "attempts": result.get("attempts"),
        "tool_calls_log": [],
    }


def _assign_pseudonyms(member_ids: list[str], seed: int | None = None) -> dict[str, str]:
    """Assign random capital letters to members. Same set of letters every time
    (A, B, C, …) but the mapping member_id → letter varies. Caller passes a seed
    to make this deterministic per-ranker — anti-positional-bias for stage 2."""
    rng = random.Random(seed)
    letters = list(string.ascii_uppercase[: len(member_ids)])
    rng.shuffle(letters)
    return dict(zip(member_ids, letters))


def _first_json_object(text: str) -> str | None:
    """Return the first complete top-level {...} object as a substring, or None.

    A balanced-brace scanner that respects JSON string literals (and their
    escapes) so braces inside string values don't skew the depth count. Unlike a
    greedy `\\{[\\s\\S]*\\}` regex, it stops at the matching close brace instead
    of swallowing trailing prose with extra braces and mangling valid JSON."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text. Tries json.loads first; on failure tries
    to find the first {...} block. Returns the parsed dict or raises ValueError."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidate = _first_json_object(text)
    if candidate is not None:
        return json.loads(candidate)
    raise ValueError("no JSON object found in response")


# The chairman appends its structured analysis after this exact line, then a
# single fenced ```json block. Anchoring on the sentinel means a ```json sample
# inside the prose answer can't be mistaken for the analysis block.
ANALYSIS_SENTINEL = "=== ANALYSIS (JSON) ==="
_ANALYSIS_KEYS = (
    "consensus", "contradictions", "partial_coverage", "unique_insights", "blind_spots",
)


def _normalize_analysis(parsed: object) -> dict | None:
    """Coerce a parsed analysis object to the fixed 5-key taxonomy: each key
    becomes a list ([] when missing or the wrong type). Returns None if `parsed`
    is not a dict or every category is empty (no signal worth surfacing)."""
    if not isinstance(parsed, dict):
        return None
    out = {
        k: (parsed.get(k) if isinstance(parsed.get(k), list) else [])
        for k in _ANALYSIS_KEYS
    }
    if not any(out[k] for k in _ANALYSIS_KEYS):
        return None
    return out


def _split_synthesis_and_analysis(content: str) -> tuple[str, dict | None]:
    """Split the chairman's output into (prose_synthesis, analysis | None).

    The chairman is told to append `=== ANALYSIS (JSON) ===` then a single fenced
    ```json block. We parse ONLY the sentinel-anchored block. Any failure (no
    sentinel, no fence, bad JSON, wrong shape, all-empty) degrades to
    (prose, None): the prose synthesis is the primary deliverable and is never
    lost — only the optional structured analysis is dropped. The sentinel and
    everything after it are stripped from the returned prose."""
    idx = content.rfind(ANALYSIS_SENTINEL)
    if idx == -1:
        return content, None
    prose = content[:idx].rstrip()
    tail = content[idx + len(ANALYSIS_SENTINEL):]
    # Anchor to the fence that immediately follows the sentinel (only whitespace/
    # newlines allowed in between). A foreign ```code``` block placed between the
    # sentinel and the real JSON must NOT be captured as the analysis — re.match
    # anchors at position 0 so the leading \s* can't skip past intervening prose.
    m = re.match(r"\s*```(?:json)?\s*\n?(.*?)```", tail, re.DOTALL)
    if not m:
        return prose, None
    try:
        parsed = json.loads(m.group(1).strip())
    except (ValueError, TypeError):
        return prose, None
    return prose, _normalize_analysis(parsed)


async def _run_member_stage2(
    ranker: dict,
    question: str,
    others: list[dict],
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
) -> dict:
    """Run one ranker through stage 2 with anonymized peer answers.

    `others` = list of stage1 results (status=ok) excluding the ranker itself.
    Each call gets a fresh pseudonym mapping (deterministic per ranker_id+seed).
    """
    start = time.monotonic()
    api_key = os.environ.get(ranker["env_key"])
    if not api_key:
        return {
            "ranker_id": ranker["id"],
            "status": "error",
            "error": f"env var {ranker['env_key']} not set",
            "rankings": [],
            "pseudonyms": {},
            "latency_ms": 0,
        }

    # Deterministic across processes (built-in hash() randomizes via PYTHONHASHSEED).
    pseudonym_seed = int(hashlib.sha256(ranker["id"].encode("utf-8")).hexdigest()[:8], 16)
    pseudonyms = _assign_pseudonyms([o["id"] for o in others], seed=pseudonym_seed)

    other_answers = [(pseudonyms[o["id"]], o["answer"]) for o in others]
    # Sort answers by pseudonym letter for stable presentation order.
    other_answers.sort(key=lambda x: x[0])

    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {
            "role": "user",
            "content": build_stage2_user(question, other_answers, files_section),
        },
    ]

    # Request larger output for Kimi-style rankers; min_max_tokens still applies.
    max_tokens = max(max_response_tokens, ranker.get("min_max_tokens", 0))
    extra = dict(ranker.get("extra") or {})

    try:
        result = await call_fn(
            base_url=ranker["base_url"],
            api_key=api_key,
            model=ranker["model"],
            messages=messages,
            max_tokens=max_tokens,
            extra_payload=extra,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Always return a dict (see _run_member_stage1): any non-cancellation
        # failure becomes status="error" so gather doesn't abort the fan-out.
        return {
            "ranker_id": ranker["id"],
            "status": "error",
            "error": str(e),
            "rankings": [],
            "pseudonyms": pseudonyms,
            "latency_ms": int((time.monotonic() - start) * 1000),
        }

    try:
        parsed = _extract_json(result["content"])
        rankings = parsed.get("rankings", [])
        if not isinstance(rankings, list):
            raise ValueError("rankings is not a list")
        # An empty rankings list almost always means the model returned
        # something malformed (e.g. bare `{}`) that we accepted as "ok JSON".
        # Treat it as an error so the operator sees the failed ranker in
        # council_status instead of a silent degradation of council quality.
        if not rankings:
            raise ValueError("rankings is empty")
    except (ValueError, KeyError) as e:
        return {
            "ranker_id": ranker["id"],
            "status": "error",
            "error": f"invalid_json: {e}",
            "rankings": [],
            "confidence": None,
            "pseudonyms": pseudonyms,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": result.get("tokens_in"),
            "tokens_out": result.get("tokens_out"),
            "attempts": result.get("attempts"),
        }

    # Confidence is optional; clamp to [1, 10] if present, else None.
    raw_conf = parsed.get("confidence")
    confidence: int | None = None
    try:
        if raw_conf is not None:
            cval = int(raw_conf)
            if 1 <= cval <= 10:
                confidence = cval
    except (TypeError, ValueError):
        confidence = None

    # Normalize: keep only entries with valid pseudonym letter and integer score.
    letter_to_id = {v: k for k, v in pseudonyms.items()}
    clean: list[dict] = []
    for r in rankings:
        if not isinstance(r, dict):
            continue
        letter = str(r.get("member", "")).strip().upper()
        if letter not in letter_to_id:
            continue
        try:
            score = int(r.get("score"))
        except (TypeError, ValueError):
            continue
        if not (1 <= score <= 10):
            continue
        clean.append(
            {
                "ranked_id": letter_to_id[letter],
                "pseudonym": letter,
                "score": score,
                # 1200 chars ≈ 5-6 sentences with comparative reasoning headroom.
                "reasoning": str(r.get("reasoning", ""))[:1200],
            }
        )

    # The raw `rankings` list was non-empty, but normalization may have dropped
    # every entry (unknown pseudonym letters, non-integer/out-of-range scores).
    # An all-invalid ranking carries no signal — surface it as a failed ranker
    # instead of silently degrading the aggregate with an empty contribution.
    if not clean:
        return {
            "ranker_id": ranker["id"],
            "status": "error",
            "error": "invalid_json: no valid rankings after normalization",
            "rankings": [],
            "confidence": None,
            "pseudonyms": pseudonyms,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": result.get("tokens_in"),
            "tokens_out": result.get("tokens_out"),
            "attempts": result.get("attempts"),
        }

    return {
        "ranker_id": ranker["id"],
        "status": "ok",
        "error": None,
        "rankings": clean,
        "confidence": confidence,
        "pseudonyms": pseudonyms,
        "latency_ms": int((time.monotonic() - start) * 1000),
        "tokens_in": result.get("tokens_in"),
        "tokens_out": result.get("tokens_out"),
        "attempts": result.get("attempts"),
    }


def _pick_chairman(
    aggregate: list[tuple[str, float, int]],
    ok_stage1_ids: list[str],
    members_by_id: dict[str, dict],
    preferred_fallback: str = DEFAULT_CHAIRMAN_FALLBACK_ID,
) -> dict | None:
    """Choose the chairman for stage 3.

    Order of preference:
      1. Highest-ranked surviving member from `aggregate`.
      2. `preferred_fallback` (default deepseek) if it survived stage 1.
      3. First survivor in `ok_stage1_ids`.
    Returns the member config dict, or None if no survivors at all.
    """
    survivors = set(ok_stage1_ids)
    if not survivors:
        return None
    for mid, _mean, _n in aggregate:
        if mid in survivors:
            return members_by_id[mid]
    if preferred_fallback in survivors:
        return members_by_id[preferred_fallback]
    return members_by_id[ok_stage1_ids[0]]


def _build_rankings_digest(
    stage1: list[dict],
    aggregate: list[tuple[str, float, int]],
    stage2: list[dict],
) -> str:
    """One short text block summarising peer review for the chairman prompt."""
    lines: list[str] = []
    model_by_id = {s["id"]: s["model"] for s in stage1}
    if aggregate:
        lines.append("Aggregate (mean score across rankers, excluding self):")
        for i, (mid, mean, n) in enumerate(aggregate, 1):
            model = model_by_id.get(mid, mid)
            lines.append(f"  {i}. {model} ({mid}) — mean {mean:.2f} (n={n})")
    else:
        lines.append("Aggregate: (no peer-rankings available — only one survivor)")
    lines.append("")
    if stage2:
        lines.append("Per-ranker reasoning (verbatim, anonymized pseudonyms inside):")
        for s in stage2:
            if s["status"] != "ok":
                continue
            ranker_model = model_by_id.get(s["ranker_id"], s["ranker_id"])
            conf = s.get("confidence")
            conf_str = f" (self-conf {conf}/10)" if conf is not None else ""
            lines.append(f"  Ranker {ranker_model}{conf_str}:")
            for r in sorted(s["rankings"], key=lambda x: -x["score"]):
                ranked_model = model_by_id.get(r["ranked_id"], r["ranked_id"])
                reasoning = (r.get("reasoning") or "").strip()
                lines.append(
                    f"    - {ranked_model}: {r['score']}/10 — {reasoning}"
                )
    return "\n".join(lines)


async def _run_stage3_synthesis(
    chairman: dict,
    question: str,
    stage1: list[dict],
    aggregate: list[tuple[str, float, int]],
    stage2: list[dict],
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
    *,
    web_search: bool = False,
    search_cache: RunSearchCache | None = None,
) -> dict:
    """Single LLM call to synthesise a final answer from stage1 + stage2 inputs.

    `web_search=True` routes the chairman through the same tool-loop stage-1
    members use, so it can fact-check a disputed claim before adopting it. The
    per-run `search_cache` is shared with stage 1 (duplicate queries are free).

    Returns:
        {"chairman_id": str, "chairman_model": str, "status": "ok"|"error",
         "synthesis": str|None, "error": str|None, "latency_ms": int,
         "tool_calls_log": [...]}
    """
    start = time.monotonic()
    api_key = os.environ.get(chairman["env_key"])
    if not api_key:
        return {
            "chairman_id": chairman["id"],
            "chairman_model": chairman["model"],
            "status": "error",
            "synthesis": None,
            "error": f"env var {chairman['env_key']} not set",
            "latency_ms": 0,
        }

    answers = [(s["model"], s["answer"]) for s in stage1 if s["status"] == "ok"]
    rankings_digest = _build_rankings_digest(stage1, aggregate, stage2)
    system_content = STAGE3_SYSTEM
    if web_search:
        system_content = STAGE3_SYSTEM + "\n\n" + STAGE3_WEB_SEARCH_NOTE
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": build_stage3_user(
                question, answers, rankings_digest, files_section
            ),
        },
    ]
    max_tokens = max(max_response_tokens, chairman.get("min_max_tokens", 0))
    tools = [WEB_SEARCH_TOOL_SPEC] if web_search else None

    try:
        if tools:
            result, tool_log = await run_with_tool_loop(
                member=chairman, api_key=api_key, messages=messages,
                max_tokens=max_tokens, call_fn=call_fn, tools=tools,
                search_cache=search_cache,
            )
        else:
            result = await call_fn(
                base_url=chairman["base_url"],
                api_key=api_key,
                model=chairman["model"],
                messages=messages,
                max_tokens=max_tokens,
                extra_payload=chairman.get("extra"),
            )
            tool_log = []
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Always return a dict (see _run_member_stage1): any non-cancellation
        # failure becomes status="error" so the caller falls back to the
        # stage 1/2 materials instead of the whole call aborting.
        return {
            "chairman_id": chairman["id"],
            "chairman_model": chairman["model"],
            "status": "error",
            "synthesis": None,
            "error": str(e),
            "latency_ms": int((time.monotonic() - start) * 1000),
        }

    # A tool-looping chairman can exhaust its iterations still wanting to search
    # — mirror the stage-1 contract: mark error so the caller falls back to the
    # stage 1/2 materials rather than relaying an empty synthesis.
    if not result.get("content"):
        return {
            "chairman_id": chairman["id"],
            "chairman_model": chairman["model"],
            "status": "error",
            "synthesis": None,
            "error": (
                f"no final content after {MAX_TOOL_ITERATIONS} tool iterations "
                f"(finish_reason={result.get('finish_reason')})"
            ),
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": result.get("tokens_in"),
            "tokens_out": result.get("tokens_out"),
            "attempts": result.get("attempts"),
            "tool_calls_log": tool_log,
            **_loop_usage(result),
        }

    prose, analysis = _split_synthesis_and_analysis(result["content"])
    return {
        "chairman_id": chairman["id"],
        "chairman_model": chairman["model"],
        "status": "ok",
        "synthesis": prose,
        "analysis": analysis,
        "error": None,
        "latency_ms": int((time.monotonic() - start) * 1000),
        "tokens_in": result.get("tokens_in"),
        "tokens_out": result.get("tokens_out"),
        "attempts": result.get("attempts"),
        "tool_calls_log": tool_log,
        **_loop_usage(result),
    }


def _aggregate(stage2_results: list[dict]) -> list[tuple[str, float, int]]:
    """Aggregate stage 2 rankings into a (member_id, weighted_mean, vote_count)
    list sorted by weighted_mean desc.

    Weight per ranker = confidence/10 (1.0 if unspecified). A high-confidence
    ranker counts more than a low-confidence one. Vote_count is the raw number
    of ranks received (not weight-adjusted) — useful sanity check.

    Self-ranks are not present by construction (stage 2 didn't include self).
    """
    weighted_sums: dict[str, float] = {}
    weight_sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for s in stage2_results:
        if s["status"] != "ok":
            continue
        conf = s.get("confidence")
        weight = (conf / 10.0) if isinstance(conf, int) and 1 <= conf <= 10 else 1.0
        for r in s["rankings"]:
            mid = r["ranked_id"]
            weighted_sums[mid] = weighted_sums.get(mid, 0.0) + r["score"] * weight
            weight_sums[mid] = weight_sums.get(mid, 0.0) + weight
            counts[mid] = counts.get(mid, 0) + 1
    agg: list[tuple[str, float, int]] = []
    for mid in weighted_sums:
        # Explicit positivity check so a real zero weight isn't silently
        # replaced by 1.0 via the `or` operator. With current logic weights
        # are always ≥ 0.1, but the explicit form survives future refactors.
        w = weight_sums[mid] if weight_sums[mid] > 0 else 1.0
        agg.append((mid, weighted_sums[mid] / w, counts[mid]))
    agg.sort(key=lambda x: x[1], reverse=True)
    return agg


def _compute_usage(
    rounds_detail: list[dict],
    stage3: dict | None,
    search_cache=None,
) -> dict:
    """Aggregate cost/usage signals across every round and stage 3.

    `llm_calls` counts only invocations that actually reached a provider (a
    member that failed on a missing env var made zero calls and is excluded).
    `retries` is the number of HTTP retries on the SUCCESS path; calls that
    exhausted their retries and failed appear in summary.failed_models instead.
    `web_search_cache_hits` = duplicate queries served from the per-run cache.
    `estimated_cost_usd` = Σ tokens × per-model price (models.CATALOG price_in/
    price_out, USD per 1M). None only if NO record had a priced model (all
    members on flat-rate subscriptions with no per-token price).

    For web_search members the per-turn result only carries its last turn's
    tokens; the loop_* aggregates (summed across every tool-loop iteration) are
    used instead when present, so multi-turn members aren't undercounted.
    """
    calls = tin = tout = retries = web = 0
    cost = 0.0
    any_priced = False

    def _model_id(rec: dict) -> str | None:
        return rec.get("id") or rec.get("ranker_id") or rec.get("chairman_id")

    def _acc(rec: dict) -> None:
        nonlocal calls, tin, tout, retries, web, cost, any_priced
        # Prefer the tool-loop aggregates for web_search members (calls/tokens/
        # attempts summed across every iteration); fall back to the single-call
        # figures for plain (non-web) members.
        if "loop_calls" in rec:
            rec_calls = rec.get("loop_calls") or 0
            rec_in = rec.get("loop_tokens_in") or 0
            rec_out = rec.get("loop_tokens_out") or 0
            rec_attempts = rec.get("loop_attempts") or 0
            calls += rec_calls
            retries += max(0, rec_attempts - rec_calls)
        else:
            rec_in = rec.get("tokens_in") or 0
            rec_out = rec.get("tokens_out") or 0
            attempts = rec.get("attempts")
            if attempts is not None:
                calls += 1
                retries += max(0, attempts - 1)
        tin += rec_in
        tout += rec_out
        web += len(rec.get("tool_calls_log") or [])

        cfg = CATALOG.get(_model_id(rec) or "")
        if cfg:
            pin, pout = cfg.get("price_in"), cfg.get("price_out")
            if pin is not None:
                cost += rec_in * pin / 1_000_000
                any_priced = True
            if pout is not None:
                cost += rec_out * pout / 1_000_000
                any_priced = True

    # A failed round-1 member is carried forward by identity (the SAME dict
    # object) into every later round's stage1, so dedup by id() to avoid
    # double-counting its tokens/web_search/cost across rounds.
    seen: set[int] = set()

    def _acc_once(rec: dict) -> None:
        if id(rec) in seen:
            return
        seen.add(id(rec))
        _acc(rec)

    for rd in rounds_detail:
        for s in rd["stage1"]:
            _acc_once(s)
        for s in rd["stage2"]:
            _acc_once(s)
    if stage3 is not None:
        _acc_once(stage3)

    return {
        "llm_calls": calls,
        "tokens_in": tin,
        "tokens_out": tout,
        "web_search_calls": web,
        "web_search_cache_hits": search_cache.hits if search_cache is not None else 0,
        "retries": retries,
        "estimated_cost_usd": round(cost, 6) if any_priced else None,
    }


def _council_failure_reason(error: str | None) -> str:
    """Classify a council failed-member error string into the same coarse enum
    healthcheck uses, so automation can branch on a stable `failure_reason` code
    instead of substring-matching the human-readable `error`.

    Reuses healthcheck._classify_error for provider-level failures (402/401/5xx/
    timeout/…). Council-only failure modes the healthcheck classifier never sees:
    a missing env var maps to `no_key` (healthcheck's dedicated status for that
    case); malformed-ranking (`invalid_json: …`) and tool-loop-exhaustion strings
    have no dedicated code and fall through to the generic `error`."""
    if not error:
        return "error"
    low = error.lower()
    if "env var" in low and "not set" in low:
        return "no_key"
    return _classify_error(error)


def _build_summary(
    stage1: list[dict],
    stage2: list[dict],
    aggregate: list[tuple[str, float, int]],
    stage3: dict | None,
) -> dict:
    """Machine-readable verdict for automation (n8n etc.): winner, confidence,
    failed models, top disagreements, recommended next action."""
    model_by_id = {s["id"]: s["model"] for s in stage1}

    failed_models: list[dict] = []
    for s in stage1:
        if s["status"] != "ok":
            failed_models.append({"id": s["id"], "model": s["model"],
                                  "stage": "stage1", "error": s.get("error"),
                                  "failure_reason": _council_failure_reason(s.get("error"))})
    for s in stage2:
        if s["status"] != "ok":
            rid = s["ranker_id"]
            failed_models.append({"id": rid, "model": model_by_id.get(rid, rid),
                                  "stage": "stage2", "error": s.get("error"),
                                  "failure_reason": _council_failure_reason(s.get("error"))})

    # Winner = top of aggregate, else the lone surviving member.
    winner_id = winner_model = None
    winner_mean = None
    if aggregate:
        winner_id, winner_mean, _ = aggregate[0]
        winner_model = model_by_id.get(winner_id, winner_id)
        winner_mean = round(winner_mean, 2)
    else:
        survivors = [s for s in stage1 if s["status"] == "ok"]
        if survivors:
            winner_id = survivors[0]["id"]
            winner_model = survivors[0]["model"]

    # Confidence from the margin between the top two aggregate means.
    if not aggregate:
        confidence = "low"
    else:
        top = aggregate[0][1]
        margin = top - aggregate[1][1] if len(aggregate) >= 2 else top
        if margin >= 1.5 and top >= 7:
            confidence = "high"
        elif margin >= 0.7:
            confidence = "medium"
        else:
            confidence = "low"

    # Disagreements: per-member score spread across rankers (1-10 scale).
    scores_by_id: dict[str, list[int]] = {}
    for s in stage2:
        if s["status"] != "ok":
            continue
        for r in s["rankings"]:
            scores_by_id.setdefault(r["ranked_id"], []).append(r["score"])
    top_disagreements: list[dict] = []
    for mid, scores in scores_by_id.items():
        if len(scores) < 2:
            continue
        spread = max(scores) - min(scores)
        if spread >= 3:
            top_disagreements.append({
                "id": mid, "model": model_by_id.get(mid, mid),
                "spread": spread, "scores": sorted(scores),
            })
    top_disagreements.sort(key=lambda d: d["spread"], reverse=True)
    top_disagreements = top_disagreements[:3]

    ok_stage1 = sum(1 for s in stage1 if s["status"] == "ok")
    # "half or more of the members had a failure" — integer math (no float /2).
    if failed_models and len(failed_models) * 2 >= len(stage1):
        next_action = "Several models failed — run model_healthcheck and retry."
    elif confidence == "low" or top_disagreements:
        next_action = (
            "Low agreement — consider synthesis=True or another round (rounds=2)."
        )
    else:
        next_action = "Clear winner — adopt the top-ranked answer."

    return {
        "winner_id": winner_id,
        "winner_model": winner_model,
        "winner_mean_score": winner_mean,
        "confidence": confidence,
        "survivors": ok_stage1,
        "failed_models": failed_models,
        "top_disagreements": top_disagreements,
        "synthesized": bool(stage3 and stage3.get("status") == "ok"),
        # Structured semantic taxonomy from the chairman (consensus / contradictions
        # / partial_coverage / unique_insights / blind_spots). None unless synthesis
        # ran and produced a parseable analysis block.
        "analysis": (
            stage3.get("analysis")
            if stage3 and stage3.get("status") == "ok"
            else None
        ),
        "recommended_next_action": next_action,
    }


async def _run_stage2_for_round(
    ok_stage1: list[dict],
    members_by_id: dict[str, dict],
    question: str,
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
    progress: ProgressFn,
) -> tuple[list[dict], list[str]]:
    """Run stage 2 (peer-rankings) over a given set of stage1 survivors.

    Returns (stage2_results, new_notes). The caller is expected to extend its
    own `notes` list with the returned `new_notes` — this avoids passing in a
    mutable list and the implicit cross-coroutine sharing that would break if
    the function were ever called concurrently.
    """
    progress("phase", {"phase": "stage2", "rankers": [s["id"] for s in ok_stage1]})

    async def _stage2_wrap(ranker_cfg: dict, others: list[dict]) -> dict:
        r = await _run_member_stage2(
            ranker_cfg, question, others, files_section, max_response_tokens, call_fn
        )
        progress("stage2_ranker", {
            "id": r["ranker_id"],
            "model": ranker_cfg["model"],
            "status": r["status"],
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
        })
        return r

    tasks = []
    for ranker_result in ok_stage1:
        ranker_cfg = members_by_id[ranker_result["id"]]
        others = [o for o in ok_stage1 if o["id"] != ranker_result["id"]]
        if not others:
            continue
        tasks.append(_stage2_wrap(ranker_cfg, others))
    stage2 = await asyncio.gather(*tasks) if tasks else []
    new_notes = [
        f"{s['ranker_id']}: stage2 error — {s['error']}; ranking ignored"
        for s in stage2 if s["status"] != "ok"
    ]
    return stage2, new_notes


async def run_council(
    question: str,
    files_section: str | None = None,
    max_response_tokens: int = 8192,
    members: list[dict] | None = None,
    call_fn: CallFn | None = None,
    synthesis: bool = False,
    rounds: int = 1,
    web_search: bool = False,
    on_progress: ProgressFn | None = None,
    context_in_stage2: bool = True,
) -> dict:
    """End-to-end orchestration of stages 1+2 across N rounds, optionally
    stage 3 synthesis at the end.

    `context_in_stage2=False`: drop the (potentially up-to-500KB) context files
    from stage 2 ranking and stage 3 synthesis prompts — stage 1 still gets the
    full context, but rankers compare answers off the question + answers alone.
    With large context_paths this avoids re-sending ~125k tokens × 7 rankers per
    round. Default True preserves the original behaviour.

    `rounds=1` (default): standard Karpathy stage1 → stage2 → optional stage3.
    `rounds=2+`: after each round, surviving members get another stage 1 with
        their previous answer + others' answers + critique digest, then a fresh
        stage 2. Final stage 3 (if requested) consumes the last round.

    Returns (when rounds=1, for backward compat the per-round fields ARE the
    only fields; when rounds>=2, the top-level fields reference the FINAL
    round, and `rounds_detail` carries each round's stage1/stage2/aggregate):
        {
            "stage1": [...],     # last-round stage1
            "stage2": [...],     # last-round stage2
            "aggregate": [...],  # last-round aggregate
            "rounds_detail": [{"stage1":..., "stage2":..., "aggregate":...}, ...],
            "stage3": <dict>|None,
            "notes": [...],
        }
    Raises RuntimeError("council fully failed") if every stage1 member errored
    in the FIRST round.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if rounds > MAX_ROUNDS:
        raise ValueError(f"rounds must be <= {MAX_ROUNDS}")

    members = members if members is not None else resolve_members(None)
    call_fn = call_fn or call_openai_compat
    progress = on_progress or _noop_progress
    members_by_id = {m["id"]: m for m in members}
    # Stage 2 (ranking) and stage 3 (synthesis) optionally skip the context
    # files; stage 1 always gets them. See context_in_stage2 in the docstring.
    stage2_files = files_section if context_in_stage2 else None

    notes: list[str] = []
    rounds_detail: list[dict] = []

    # One shared search cache per run — members issue overlapping queries and
    # otherwise pay Exa per duplicate. Round-1 stage 1 and (when synthesis runs)
    # the stage-3 chairman share it; stage 2 and rounds 2+ stay search-free.
    search_cache = RunSearchCache() if web_search else None

    # --- Round 1 ---------------------------------------------------------
    progress("phase", {"phase": "stage1", "members": [m["id"] for m in members]})

    async def _stage1_wrap(member: dict) -> dict:
        r = await _run_member_stage1(
            member, question, files_section, max_response_tokens, call_fn,
            web_search=web_search, on_progress=progress, search_cache=search_cache,
        )
        progress("stage1_member", {
            "id": r["id"], "model": r["model"], "status": r["status"],
            "error": r.get("error"), "latency_ms": r.get("latency_ms"),
            "tool_calls_count": len(r.get("tool_calls_log") or []),
        })
        return r

    stage1 = await asyncio.gather(*(_stage1_wrap(m) for m in members))
    ok_stage1 = [s for s in stage1 if s["status"] == "ok"]
    if not ok_stage1:
        progress("phase", {"phase": "error", "error": "council fully failed"})
        raise RuntimeError("council fully failed")

    for s in stage1:
        if s["status"] != "ok":
            notes.append(
                f"{s['id']} ({s['model']}): stage1 error — {s['error']}; excluded from both stages"
            )

    stage2, stage2_notes = await _run_stage2_for_round(
        ok_stage1, members_by_id, question, stage2_files,
        max_response_tokens, call_fn, progress,
    )
    notes.extend(stage2_notes)
    aggregate = _aggregate(stage2)
    rounds_detail.append({"stage1": stage1, "stage2": stage2, "aggregate": aggregate})

    # --- Round 2+ --------------------------------------------------------
    for round_idx in range(2, rounds + 1):
        progress("phase", {"phase": f"round{round_idx}_stage1"})
        prior_stage1 = stage1  # immutable snapshot of previous round
        prior_aggregate = aggregate
        prior_stage2 = stage2
        digest = _build_rankings_digest(prior_stage1, prior_aggregate, prior_stage2)

        ok_ids = {s["id"] for s in prior_stage1 if s["status"] == "ok"}
        prior_answers_by_id = {
            s["id"]: s["answer"] for s in prior_stage1 if s["status"] == "ok"
        }

        async def _stage1_round_n_wrap(member: dict) -> dict:
            if member["id"] not in ok_ids:
                # Skipped permanently — keep the round-1 error result so the
                # member doesn't reappear.
                prior = next(s for s in prior_stage1 if s["id"] == member["id"])
                return prior
            own_prev = prior_answers_by_id[member["id"]]
            others = [
                (prior_stage1[i]["model"], prior_stage1[i]["answer"])
                for i, s in enumerate(prior_stage1)
                if s["status"] == "ok" and s["id"] != member["id"]
            ]
            r = await _run_member_stage1_round_n(
                member, question, own_prev, others, digest,
                files_section, max_response_tokens, call_fn,
            )
            progress("stage1_member", {
                "id": r["id"], "model": r["model"], "status": r["status"],
                "error": r.get("error"), "latency_ms": r.get("latency_ms"),
            })
            return r

        stage1 = await asyncio.gather(*(_stage1_round_n_wrap(m) for m in members))
        ok_stage1 = [s for s in stage1 if s["status"] == "ok"]
        if not ok_stage1:
            notes.append(
                f"round {round_idx} stage1 had no survivors — keeping round {round_idx - 1} as final"
            )
            # Restore the prior round's snapshots so the returned payload is
            # self-consistent: stage1/stage2/aggregate (and ok_stage1, used by
            # stage 3) all reference round N-1, which had live answers. Without
            # this, stage1 would point at the all-error round while stage2/
            # aggregate still came from N-1 — and stage 3 would be skipped.
            stage1 = prior_stage1
            ok_stage1 = [s for s in prior_stage1 if s["status"] == "ok"]
            stage2 = prior_stage2
            aggregate = prior_aggregate
            break

        stage2, stage2_notes = await _run_stage2_for_round(
            ok_stage1, members_by_id, question, stage2_files,
            max_response_tokens, call_fn, progress,
        )
        notes.extend(stage2_notes)
        aggregate = _aggregate(stage2)
        rounds_detail.append({"stage1": stage1, "stage2": stage2, "aggregate": aggregate})

    # --- Stage 3 (optional) ---------------------------------------------
    stage3: dict | None = None
    if synthesis:
        chairman = _pick_chairman(
            aggregate, [s["id"] for s in ok_stage1], members_by_id
        )
        if chairman is None:
            notes.append("stage3 synthesis skipped — no stage1 survivors")
        else:
            progress("phase", {"phase": "stage3", "chairman": chairman["id"]})
            stage3 = await _run_stage3_synthesis(
                chairman=chairman,
                question=question,
                stage1=stage1,
                aggregate=aggregate,
                stage2=stage2,
                files_section=stage2_files,
                max_response_tokens=max_response_tokens,
                call_fn=call_fn,
                web_search=web_search,
                search_cache=search_cache,
            )
            progress("stage3", {
                "id": stage3["chairman_id"],
                "model": stage3["chairman_model"],
                "status": stage3["status"],
                "error": stage3.get("error"),
                "latency_ms": stage3.get("latency_ms"),
            })
            if stage3["status"] != "ok":
                notes.append(
                    f"stage3 synthesis error from {stage3['chairman_id']} "
                    f"({stage3['chairman_model']}): {stage3['error']}"
                )
            elif stage3.get("analysis") is None:
                notes.append(
                    "stage3 analysis JSON missing/unparseable — prose synthesis "
                    "intact, structured analysis (blind_spots etc.) unavailable."
                )

    progress("phase", {"phase": "done"})
    return {
        "stage1": stage1,
        "stage2": stage2,
        "aggregate": aggregate,
        "rounds_detail": rounds_detail,
        "stage3": stage3,
        "notes": notes,
        "usage": _compute_usage(rounds_detail, stage3, search_cache),
        "summary": _build_summary(stage1, stage2, aggregate, stage3),
    }
