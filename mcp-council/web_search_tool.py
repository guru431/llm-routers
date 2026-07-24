"""Shared tool-loop for `web_search`. Used by both council.py (per-member
stage 1 with web_search=True) and single_call.py (model_ask with
web_search=True). Pure refactor — behavior identical to the pre-extraction
implementation that lived inside council.py."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from dlp import scrub_outbound_query
from openai_client import call_openai_compat
from web_search import (
    WEB_SEARCH_TOOL_SPEC,
    WebSearchError,
    format_error_for_llm,
    format_results_for_llm,
    web_search_exa,
)

# Hard cap on tool-using turns. Thinking-style models (glm/qwen/deepseek-pro)
# routinely need 6-10 narrow searches on multi-question prompts; the older
# cap of 5 dropped them entirely. 12 leaves headroom without runaway risk
# (cost cap ≈ 12 × $0.005 per Exa search = $0.06 per member per stage).
# On turn MAX+1 we additionally pass tool_choice="none" — see run_with_tool_loop.
MAX_TOOL_ITERATIONS = 12

# Run-wide ceiling on ACTUAL (billed) Exa searches, shared via RunSearchCache
# across every member + the stage-3 chairman. Per-member MAX_TOOL_ITERATIONS
# bounds one member; without a run cap a 7-member council + chairman could reach
# ~100 distinct paid queries. 40 ≈ $0.20 worst-case — past it, further distinct
# queries return a budget_exhausted note to the model instead of hitting Exa.
MAX_RUN_SEARCHES = 40

CallFn = Callable[..., Awaitable[dict]]
ProgressFn = Callable[[str, dict[str, Any]], None]


def _noop_progress(event_type: str, payload: dict[str, Any]) -> None:  # noqa: ARG001
    return None


class RunSearchCache:
    """Per-council-run web_search cache keyed by normalized query.

    All council members run stage 1 concurrently and frequently issue the same
    obvious query, paying Exa per call. This caches by normalized query for the
    duration of ONE run (created fresh in run_council — never global, so results
    can't go stale across runs). Stores the in-flight asyncio.Task so concurrent
    identical queries collapse to a single Exa call rather than racing.
    """

    def __init__(self, search_fn=None, max_searches: int = MAX_RUN_SEARCHES) -> None:
        # None → resolve the module-level web_search_exa at call time so test
        # patches of `web_search_tool.web_search_exa` take effect.
        self._search_fn = search_fn
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._max_searches = max_searches
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _norm(query: str) -> str:
        return " ".join(query.strip().lower().split())

    async def search(self, query: str) -> dict:
        key = self._norm(query)
        async with self._lock:
            task = self._tasks.get(key)
            if task is None:
                # Run-wide budget: a NEW distinct query is a billed Exa call.
                # Cached repeats (task is not None) are always free and allowed.
                if self.misses >= self._max_searches:
                    raise WebSearchError(
                        f"run web_search budget exhausted "
                        f"({self._max_searches} searches) — answer from what you have"
                    )
                self.misses += 1
                fn = self._search_fn or web_search_exa
                task = asyncio.ensure_future(fn(query))
                self._tasks[key] = task
            else:
                self.hits += 1
        # Await outside the lock so a slow Exa call doesn't serialize other
        # distinct queries. WebSearchError propagates to every awaiter that is
        # ALREADY waiting on this task, but the failed task is then evicted so a
        # LATER member asking the same query gets a fresh attempt instead of the
        # cached exception forever (a transient 5xx/timeout otherwise poisoned
        # the query for the whole run). `misses` is deliberately NOT decremented:
        # the run-wide budget stays a hard ceiling, so a persistently failing
        # query can't be retried without bound.
        try:
            return await task
        except Exception:
            async with self._lock:
                if self._tasks.get(key) is task:
                    del self._tasks[key]
            raise


async def execute_tool_call(
    tc: dict,
    on_progress: ProgressFn,
    member_id: str,
    search_cache: "RunSearchCache | None" = None,
) -> tuple[str, dict]:
    """Execute one OpenAI-style tool_call (currently only web_search).
    Returns (tool_message_content, log_entry)."""
    # A model can emit a malformed tool_calls array — a bare scalar, a null, or
    # `function` set to a non-object. Guard every access so a bad shape becomes a
    # tool-error message the loop can relay, not an AttributeError that kills the
    # whole member mid-fan-out.
    if not isinstance(tc, dict):
        return ("# Tool call error\nMalformed tool_call (not an object).",
                {"name": "", "ok": False, "error": "tool_call is not an object"})
    fn = tc.get("function")
    if not isinstance(fn, dict):
        fn = {}
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "")
    log: dict[str, Any] = {"name": name, "raw_arguments": raw_args, "ok": False}
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except (TypeError, ValueError) as e:
        log["error"] = f"invalid JSON in tool arguments: {e}"
        return (f"# Tool call error\nInvalid JSON arguments to `{name}`: {e}", log)
    # Arguments must be a JSON object; a bare array/string/number would break the
    # `.get("query")` below with an AttributeError.
    if not isinstance(args, dict):
        log["error"] = "tool arguments are not a JSON object"
        return (f"# Tool call error\nArguments to `{name}` must be a JSON object.", log)

    if name == "web_search":
        query = (args.get("query") or "").strip()
        log["query"] = query
        if not query:
            log["error"] = "empty query"
            return (f"# Web search failed\nEmpty query passed to web_search.", log)
        # Outbound DLP: a model can echo a secret it saw (context, peer answer,
        # its own reasoning) into a query and ship it to a third-party search
        # API. Scrub BEFORE the query leaves the process — a blocked query never
        # hits Exa and the model gets a refusal it can retry from.
        _safe, block_reason = scrub_outbound_query(query)
        if block_reason is not None:
            log["error"] = block_reason
            log["dlp_blocked"] = True
            on_progress("tool_call", {
                "member_id": member_id, "name": "web_search",
                "query": "[redacted]", "status": "error", "error": block_reason,
            })
            return (
                f"# Web search blocked (data-loss prevention)\n{block_reason}. "
                "Rephrase without any secret/credential/local path and try again.",
                log,
            )
        try:
            result = await (
                search_cache.search(query) if search_cache is not None
                else web_search_exa(query)
            )
        except WebSearchError as e:
            log["error"] = str(e)
            on_progress("tool_call", {
                "member_id": member_id, "name": "web_search",
                "query": query, "status": "error", "error": str(e),
            })
            return (format_error_for_llm(str(e), query), log)
        log.update({
            "ok": True,
            "num_results": len(result["results"]),
            "cost_dollars": result.get("cost_dollars"),
            "latency_ms": result.get("latency_ms"),
            # Result URLs backing this query — consumed by dlp.build_claim_ledger
            # to map each executed search to the sources it surfaced.
            "sources": [r.get("url") for r in result["results"] if r.get("url")],
        })
        on_progress("tool_call", {
            "member_id": member_id, "name": "web_search",
            "query": query, "status": "ok",
            "num_results": len(result["results"]),
            "latency_ms": result.get("latency_ms"),
        })
        return (format_results_for_llm(result), log)

    log["error"] = f"unknown tool: {name}"
    return (f"# Tool call error\nUnknown tool `{name}`. Available: web_search.", log)


async def run_with_tool_loop(
    *,
    member: dict,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    call_fn: CallFn | None = None,
    tools: list[dict] | None = None,
    on_progress: ProgressFn | None = None,
    search_cache: "RunSearchCache | None" = None,
) -> tuple[dict, list[dict]]:
    """Drive the chat loop, executing tool_calls until the model emits content
    or the iteration cap is hit. Returns (final_result_dict, tool_call_log).

    `messages` is mutated as the loop adds assistant + tool messages.
    """
    call_fn = call_fn or call_openai_compat
    progress = on_progress or _noop_progress
    if tools is None:
        tools = [WEB_SEARCH_TOOL_SPEC]
    tool_log: list[dict] = []
    last_result: dict | None = None
    # Accumulate usage across ALL loop iterations — each call_fn turn spends
    # tokens/attempts, but only the final result dict carries its own (last)
    # turn's numbers. Without this, usage-accounting undercounts a multi-turn
    # web_search member to just its closing call. Surfaced under loop_* keys so
    # council._compute_usage sums them instead of the single last-turn figures.
    loop_calls = loop_tin = loop_tout = loop_attempts = 0
    total_turns = MAX_TOOL_ITERATIONS + 1
    for turn in range(total_turns):
        # Last turn: forbid new tool calls so the model writes a final answer
        # from whatever it has already gathered. Without this, models that
        # prefer many narrow queries lose all their work on hitting the cap.
        # If a provider ignores tool_choice="none", we fall through with no
        # content and the caller marks the member as error — same as before.
        force_no_tools = turn == total_turns - 1
        result = await call_fn(
            base_url=member["base_url"],
            api_key=api_key,
            model=member["model"],
            messages=messages,
            max_tokens=max_tokens,
            extra_payload=member.get("extra"),
            tools=tools,
            tool_choice="none" if force_no_tools else None,
        )
        last_result = result
        loop_calls += 1
        loop_tin += result.get("tokens_in") or 0
        loop_tout += result.get("tokens_out") or 0
        loop_attempts += result.get("attempts") or 1
        tool_calls = result.get("tool_calls")
        # A provider can return a malformed `tool_calls`: a bare object instead
        # of a list, a null element, a scalar, or `function` set to a non-object.
        # execute_tool_call() tolerates all of those, but the loop around it used
        # to call `tc.get(...)` unconditionally right after — turning a local
        # tool-call error into an AttributeError that killed the whole member,
        # exactly the opposite of the fail-soft contract. Normalize here so the
        # rest of the turn only ever sees well-formed calls.
        if tool_calls and not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if isinstance(tool_calls, list):
            malformed = [tc for tc in tool_calls
                         if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict)]
            for bad in malformed:
                tool_log.append({
                    "name": "", "ok": False,
                    "error": f"malformed tool_call from provider ({type(bad).__name__})",
                })
            tool_calls = [tc for tc in tool_calls
                          if isinstance(tc, dict) and isinstance(tc.get("function"), dict)]
        if not tool_calls or force_no_tools:
            # Stop here when there are no tool calls OR this is the forced-final
            # turn: any tool_calls a provider returns despite tool_choice="none"
            # are DROPPED (not executed), so a run can't exceed
            # MAX_TOOL_ITERATIONS actual searches by one. If `content` is present
            # it's the final answer; if not, the caller marks the member error.
            result.update({
                "loop_calls": loop_calls,
                "loop_tokens_in": loop_tin,
                "loop_tokens_out": loop_tout,
                "loop_attempts": loop_attempts,
            })
            return result, tool_log

        # Record the assistant turn that produced these tool_calls, then
        # answer each tool_call with a `tool` role message.
        assistant_msg: dict = {
            "role": "assistant",
            "content": result.get("content") or "",
            "tool_calls": tool_calls,
        }
        # DeepSeek thinking-mode REQUIRES that reasoning_content be echoed back
        # in the conversation — without this, the next call returns http 400.
        # Other providers ignore the field, so this is safe to always include.
        rc = result.get("reasoning_content")
        if rc:
            assistant_msg["reasoning_content"] = rc
        messages.append(assistant_msg)
        for tc in tool_calls:
            tool_msg, log_entry = await execute_tool_call(
                tc, progress, member["id"], search_cache=search_cache
            )
            tool_log.append(log_entry)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                # Reuse the name the executor already normalized rather than
                # re-digging into the (possibly odd) provider payload.
                "name": log_entry.get("name") or "",
                "content": tool_msg,
            })
    # Hit the cap. Return whatever the model said last, even if it's another
    # tool_calls request — caller will see no content and mark error.
    if last_result is not None:
        last_result.update({
            "loop_calls": loop_calls,
            "loop_tokens_in": loop_tin,
            "loop_tokens_out": loop_tout,
            "loop_attempts": loop_attempts,
        })
    return last_result, tool_log  # type: ignore[return-value]
