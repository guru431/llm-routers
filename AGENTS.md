# AGENTS.md — llm_routers

Зонтик для LLM-routing инструментов. Актуальный состав:

- `mcp-council/` — единый MCP-сервер с двумя группами tool'ов:
  - **Council** (single-shot): `council_ask` (3-stage Karpathy с подмножеством моделей ≥2) и `model_ask` (один прямой вызов модели из `models.CATALOG`). Async-pattern для council: `council_ask_async` + `council_status/result/cancel/list_jobs`.
  - **Dialogue** (продолжительные диалоги с anti-convergence): `model_debate` / `model_panel` / `model_socratic` + `dialogue_continue/status/result/cancel/list_sessions`. Все starter-tool'ы async-only (5-50 мин). Hard cap rounds=20, активных сессий=20.
- `claude-code-router/` — HTTP-прокси `ccr` (npm `@musistudio/claude-code-router` ставится глобально; в репо — только `config.example.json` + `custom_router.js`).
- `claude-agent-server/` — обёртка `claude -p` в OpenAI-compatible API на :8765.
- `bench/` — раннер LLM-бенчмарков, пишет markdown-отчёт в корень репо.

## Точки входа

- Детальные правила (реестр моделей, sandbox-лимиты, ключи, тесты): [CLAUDE.md](CLAUDE.md).

## Project-specific gotchas

- **Удалённые пакеты:** `mcp-deepseek` и `mcp-minimax` удалены. Tool refs `mcp__deepseek-helper__*` и `mcp__minimax-helper__*` больше не существуют — использовать `mcp__council__model_ask` с нужным `model_id` из `models.CATALOG`.
- **`minimax-direct` отключён** в `CATALOG` (billing off) — не передавать как `model_id` в `model_ask`, выпадет ошибка.
- **`claude-agent-server`: tool calling не работает** — `claude -p` воспринимает custom tools как prompt injection. Только chat completions.
- **`mcp-council/dialogue/`: три не-очевидных грабли:**
  - `task.cancel()` на не-стартовавшей корутине не входит в её `try/except` — нужен `await asyncio.sleep(0)` перед cancel.
  - `tests/dialogue/` НЕ должна иметь `__init__.py` — иначе `tests.dialogue` затеняет production `dialogue/`. Basename тестов должны быть уникальны (`test_dialogue_state.py`, не `test_state.py`).
  - Failure threshold в `run_dialogue` считает distinct participants, не error-entries (один failing участник = 2 entries за раунд).

## Env keys

`DEEPSEEK_KEY`, `OPENCODE_GO_KEY`, `HELICONE_GATEWAY_KEY`, `EXA_API_KEY` — передаются через окружение (для MCP-сервера — `~/.claude.json` → `mcpServers.council.env`). В репозиторий не попадают.
