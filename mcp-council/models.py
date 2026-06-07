"""Model catalog for mcp-council.

Single source of truth for both council deliberation members and single-model
routine workers. Replaces the old config.py + duplicate configs in
mcp-deepseek/mcp-minimax.

Quirks (`extra`, `min_max_tokens`) per model are documented inline below —
they encode provider-specific requirements (e.g. GLM needs thinking disabled,
Kimi needs reasoning_effort "none") to avoid truncated/garbage output.

Env key names (`env_key`) are read from the process environment; see the
project README for how keys are provided to the MCP server.
"""

from __future__ import annotations

OCG = "https://opencode.ai/zen/go/v1"
DS = "https://api.deepseek.com/v1"
HEL = "https://ai-gateway.helicone.ai/v1"
MM = "https://api.minimaxi.chat/v1"


CATALOG: dict[str, dict] = {
    # --- Council members (default participants of council_ask) ---
    "glm": {
        "model": "glm-5.1",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "extra": {"thinking": {"type": "disabled"}},
    },
    "kimi": {
        "model": "kimi-k2.6",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "extra": {"reasoning_effort": "none"},
        "min_max_tokens": 30000,
    },
    "deepseek-pro": {
        # via OCG-прокси с 2026-06-07 (DeepSeek direct PAYG исчерпан, вряд ли вернётся)
        "model": "deepseek-v4-pro",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
    },
    "qwen": {
        "model": "qwen3.6-plus",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
    },
    "minimax": {
        "model": "minimax-m2.7",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "min_max_tokens": 30000,
    },
    "gemini": {
        "model": "gemini-3.1-pro-preview",
        "base_url": HEL,
        "env_key": "HELICONE_GATEWAY_KEY",
        "min_max_tokens": 30000,
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
    },

    # --- Routine workers (model_ask only) ---
    "deepseek-flash": {
        # via OCG-прокси с 2026-06-07 (DeepSeek direct PAYG исчерпан, вряд ли вернётся)
        "model": "deepseek-v4-flash",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
    },
    "minimax-direct": {
        "model": "abab7-chat-preview",
        "base_url": MM,
        "env_key": "MINIMAX_API_KEY",
        "enabled": False,
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


class UnknownModelError(RuntimeError):
    """Raised when a model_id is not present in CATALOG."""


class DisabledModelError(RuntimeError):
    """Raised when a model_id is present but disabled (enabled: False)."""


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

    Preserves input order. Raises on first invalid id.
    """
    if ids is None:
        ids = COUNCIL_DEFAULT
    return [resolve_member(i) for i in ids]
