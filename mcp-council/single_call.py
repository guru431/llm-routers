"""Single-model call engine for `model_ask`.

Thin wrapper over openai_client.call_openai_compat. Supports web_search via the
shared tool-loop. Stateless: each call is independent.
"""

from __future__ import annotations

import os

from openai_client import call_openai_compat
from web_search import WEB_SEARCH_TOOL_SPEC
from web_search_tool import run_with_tool_loop


async def run_single(
    cfg: dict,
    *,
    prompt: str,
    max_tokens: int,
    web_search: bool = False,
) -> str:
    """One LLM call. Returns the model's text answer (or '' if empty).

    Raises:
        RuntimeError if the env var for this cfg's api key is not set, or (web_search
            path) if the tool-loop exhausts its iterations without a final answer.
        CouncilHTTPError on network / HTTP / parsing failure.
    """
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise RuntimeError(f"env var {cfg['env_key']} not set for {cfg['id']}")

    effective_max = max(max_tokens, cfg.get("min_max_tokens", 0))
    messages = [{"role": "user", "content": prompt}]

    if web_search:
        result, _tool_log = await run_with_tool_loop(
            member=cfg,
            api_key=api_key,
            messages=messages,
            max_tokens=effective_max,
            tools=[WEB_SEARCH_TOOL_SPEC],
        )
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
