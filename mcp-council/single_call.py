"""Single-model call engine for `model_ask`.

Thin wrapper over openai_client.call_openai_compat. Supports web_search via the
shared tool-loop. Stateless: each call is independent.
"""

from __future__ import annotations

import os

from models import effective_max_tokens
from openai_client import call_openai_compat
from web_search import WEB_SEARCH_TOOL_SPEC
from web_search_tool import RunSearchCache, run_with_tool_loop


async def run_single(
    cfg: dict,
    *,
    prompt: str,
    max_tokens: int,
    web_search: bool = False,
    max_web_searches: int | None = None,
    tool_log_out: list | None = None,
) -> str:
    """One LLM call. Returns the model's text answer (or '' if empty).

    `max_web_searches` overrides the run-wide paid-search ceiling on the
    web_search path. A cache is ALWAYS created there: the run-wide cap and the
    duplicate-query dedup live in RunSearchCache, which only run_council and
    run_critique used to build. Going through the same tool loop with
    search_cache=None meant this path had no ceiling on paid Exa calls at all —
    up to MAX_TOOL_ITERATIONS turns, each of which may carry several tool_calls,
    all executed.

    `tool_log_out` — if given, the tool-loop log entries are appended to it so
    the caller can audit queries / cost / DLP blocks (this path writes no full
    dump).

    Raises:
        RuntimeError if the env var for this cfg's api key is not set, or (web_search
            path) if the tool-loop exhausts its iterations without a final answer.
        CouncilHTTPError on network / HTTP / parsing failure.
    """
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise RuntimeError(f"env var {cfg['env_key']} not set for {cfg['id']}")

    effective_max = effective_max_tokens(max_tokens, cfg)
    messages = [{"role": "user", "content": prompt}]

    if web_search:
        cache = (
            RunSearchCache(max_searches=max_web_searches)
            if max_web_searches is not None else RunSearchCache()
        )
        result, tool_log = await run_with_tool_loop(
            member=cfg,
            api_key=api_key,
            messages=messages,
            max_tokens=effective_max,
            tools=[WEB_SEARCH_TOOL_SPEC],
            search_cache=cache,
        )
        if tool_log_out is not None:
            tool_log_out.extend(tool_log)
        content = result.get("content")
        if not content:
            # Loop exhausted its iteration cap with no final answer (the model
            # kept calling tools). Same contract as the council path: surface a
            # hard error instead of silently returning "".
            raise RuntimeError(
                "no final content after tool iterations "
                f"(finish_reason={result.get('finish_reason')})"
            )
        return content

    result = await call_openai_compat(
        base_url=cfg["base_url"],
        api_key=api_key,
        model=cfg["model"],
        messages=messages,
        max_tokens=effective_max,
        extra_payload=cfg.get("extra"),
    )
    return result.get("content") or ""
