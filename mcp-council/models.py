"""Model catalog for mcp-council.

Single source of truth for both council deliberation members and single-model
routine workers. Replaces the old config.py + duplicate configs in
mcp-deepseek/mcp-minimax.

Quirks (`extra`, `min_max_tokens`) per model are documented inline below —
they encode provider-specific requirements (e.g. GLM needs thinking disabled,
Kimi k2.7-code needs reasoning_effort "minimal") to avoid truncated/garbage output.

Env key names (`env_key`) are read from the process environment; see the
project README for how keys are provided to the MCP server.

Pricing (`price_in`/`price_out`, USD per 1M tokens) drives council
usage.estimated_cost_usd. Only models with a published per-token PAYG price get
real numbers — flat-rate subscription models (OCG $10/mo: glm/kimi/qwen/minimax;
ChatGPT-flat: codex/gpt-5.5; Helicone gemini 3.1-pro-preview has no listed price)
are None and contribute 0 to the cost estimate. DeepSeek PAYG list prices as of
2026-05.
"""

from __future__ import annotations

OCG = "https://opencode.ai/zen/go/v1"
DS = "https://api.deepseek.com/v1"
HEL = "https://ai-gateway.helicone.ai/v1"
MM = "https://api.minimaxi.chat/v1"


CATALOG: dict[str, dict] = {
    # --- Council members (default participants of council_ask) ---
    "glm": {
        "model": "glm-5.2",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "extra": {"thinking": {"type": "disabled"}},
        # OCG flat-rate subscription — no published per-token price.
        "price_in": None,
        "price_out": None,
    },
    "kimi": {
        "model": "kimi-k2.7-code",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        # k2.7-code не поддерживает reasoning_effort="none" (HTTP 400, в отличие
        # от k2.6) — допустимы minimal|low|medium. minimal — ближайшее к none.
        "extra": {"reasoning_effort": "minimal"},
        "min_max_tokens": 30000,
        "price_in": None,
        "price_out": None,
    },
    "deepseek-pro": {
        # via OCG-прокси с 2026-06-07 (DeepSeek direct PAYG исчерпан, вряд ли вернётся)
        "model": "deepseek-v4-pro",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        # DeepSeek PAYG list price (50% off promo): $0.435/1M in, $0.87/1M out.
        "price_in": 0.435,
        "price_out": 0.87,
    },
    "qwen": {
        "model": "qwen3.6-plus",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "price_in": None,
        "price_out": None,
    },
    "minimax": {
        "model": "minimax-m3",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "min_max_tokens": 30000,
        "price_in": None,
        "price_out": None,
    },
    "gemini": {
        "model": "gemini-3.1-pro-preview",
        "base_url": HEL,
        "env_key": "HELICONE_GATEWAY_KEY",
        "min_max_tokens": 30000,
        # No published price for 3.1-pro-preview via Helicone Gateway.
        "price_in": None,
        "price_out": None,
    },
    "codex": {
        # codex-agent-server (local OpenAI-compatible wrapper over `codex exec`,
        # ChatGPT subscription). `sandbox: read-only` forces pure text generation
        # — без него дефолт сервера тоже read-only, но члену совета агентный режим
        # не нужен ни при каких настройках сервера. Сервер должен быть запущен на
        # :8766; CODEX_AGENT_TOKEN передаётся через окружение MCP-сервера.
        "model": "gpt-5.5",
        "base_url": "http://127.0.0.1:8766/v1",
        "env_key": "CODEX_AGENT_TOKEN",
        "extra": {"sandbox": "read-only"},
        # `codex exec gpt-5.5` is a reasoning model spawned as a subprocess
        # (cold start) — a real POST, not a light /health GET. The default 12s
        # probe almost always ReadTimeouts on a healthy server, so give this
        # local agent-server member a longer healthcheck ceiling.
        "healthcheck_timeout": 75.0,
        # ChatGPT/Codex flat subscription — no per-token price.
        "price_in": None,
        "price_out": None,
    },

    # --- Routine workers (model_ask only) ---
    "deepseek-flash": {
        # via OCG-прокси с 2026-06-07 (DeepSeek direct PAYG исчерпан, вряд ли вернётся)
        "model": "deepseek-v4-flash",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        # DeepSeek PAYG list price: $0.14/1M in, $0.28/1M out.
        "price_in": 0.14,
        "price_out": 0.28,
    },
    "minimax-direct": {
        "model": "abab7-chat-preview",
        "base_url": MM,
        "env_key": "MINIMAX_API_KEY",
        "enabled": False,
        "price_in": None,
        "price_out": None,
    },
}


COUNCIL_DEFAULT: list[str] = [
    "glm",
    "kimi",
    "deepseek-pro",
    "qwen",
    "minimax",
    "gemini",
    "codex",
]


# Named council presets — convenience over hand-listing model ids. Definitions
# are heuristic (tune against bench/ results) and kept as EXPLICIT lists so a
# change is a one-line edit that never silently reshuffles a caller's council.
# No "local" preset: the catalog has no local-runtime members.
PRESETS: dict[str, list[str]] = {
    "best": list(COUNCIL_DEFAULT),                    # all strongest members
    "balanced": ["deepseek-pro", "glm", "gemini"],    # strong + mid mix, fewer calls
    "cheap": ["glm", "qwen"],                          # lowest-cost OCG pair
}


class UnknownModelError(RuntimeError):
    """Raised when a model_id is not present in CATALOG."""


class DisabledModelError(RuntimeError):
    """Raised when a model_id is present but disabled (enabled: False)."""


class UnknownPresetError(RuntimeError):
    """Raised when a preset name is not in PRESETS."""


def resolve_preset(name: str) -> list[str]:
    """Return the model-id list for a named preset (copy). Raises UnknownPresetError."""
    if name not in PRESETS:
        raise UnknownPresetError(
            f"unknown preset: '{name}'. Available: {sorted(PRESETS)}"
        )
    return list(PRESETS[name])


def resolve_member(id: str) -> dict:
    """Return cfg dict with `id` injected. Raises UnknownModelError / DisabledModelError."""
    if id not in CATALOG:
        raise UnknownModelError(
            f"unknown model_id: '{id}'. Available: {sorted(CATALOG.keys())}"
        )
    cfg = CATALOG[id]
    if cfg.get("enabled") is False:
        raise DisabledModelError(f"model '{id}' is disabled in catalog")
    return {"id": id, **cfg}


def resolve_members(ids: list[str] | None) -> list[dict]:
    """Resolve a list of model_ids into cfgs. None → COUNCIL_DEFAULT.

    Preserves input order, dropping duplicate ids (a model can appear only once —
    duplicates would collide on council pseudonyms and skew aggregation).
    Raises on first invalid id.
    """
    if ids is None:
        ids = COUNCIL_DEFAULT
    seen: set[str] = set()
    unique: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return [resolve_member(i) for i in unique]
