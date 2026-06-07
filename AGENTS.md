# AGENTS.md — llm_routers

## ⚠️ Публичный репозиторий — не допускать утечек

Этот репозиторий **публичный** (GitHub). При любом редактировании НЕ коммить:
- секреты, ключи, токены, пароли (только через env-переменные / `*.example.env` с пустыми значениями);
- реальные приватные IP (`192.168.x`, `10.x`) и внутренние хосты/SSH-порты;
- внутренние домены и персональные данные (email, телефон, ФИО, адреса);
- имена внутренних проектов, серверов и людей.

Защита — pre-commit hook `.githooks/pre-commit`: generic-сканер форматов ключей + локальный `.sanitize-patterns` (gitignored denylist конкретных значений). После клона активировать: `git config core.hooksPath .githooks`. Сам `.sanitize-patterns` НИКОГДА не коммить.

Зонтик для LLM-routing инструментов. Актуальный состав:

- `mcp-council/` — единый MCP-сервер с двумя группами tool'ов:
  - **Council** (single-shot): `council_ask` (3-stage Karpathy с подмножеством моделей ≥2) и `model_ask` (один прямой вызов модели из `models.CATALOG`). Async-pattern для council: `council_ask_async` + `council_status/result/cancel/list_jobs`.
  - **Dialogue** (продолжительные диалоги с anti-convergence): `model_debate` / `model_panel` / `model_socratic` + `dialogue_continue/status/result/cancel/list_sessions`. Все starter-tool'ы async-only (5-50 мин). Hard cap rounds=20, активных сессий=20.
- `claude-code-router/` — HTTP-прокси `ccr` (npm `@musistudio/claude-code-router` ставится глобально; в репо — только `config.example.json` + `custom_router.js`).
- `claude-agent-server/` — обёртка `claude -p` в OpenAI-compatible API на :8765.
- `codex-agent-server/` — обёртка `codex exec` в OpenAI-compatible API на :8766. Два режима: read-only (дефолт, чистый чат) и workspace-write (агент правит файлы); выбирается именем модели (`gpt-5.5` vs `gpt-5.5-agent`) или полем `sandbox` в body, `tools` форсят read-only.
- `bench/` — раннер LLM-бенчмарков, пишет markdown-отчёт в корень репо.

## Точки входа

- Детальные правила (реестр моделей, sandbox-лимиты, ключи, тесты): [CLAUDE.md](CLAUDE.md).

## Project-specific gotchas

- **Удалённые пакеты:** `mcp-deepseek` и `mcp-minimax` удалены. Tool refs `mcp__deepseek-helper__*` и `mcp__minimax-helper__*` больше не существуют — использовать `mcp__council__model_ask` с нужным `model_id` из `models.CATALOG`.
- **`minimax-direct` отключён** в `CATALOG` (billing off) — не передавать как `model_id` в `model_ask`, выпадет ошибка.
- **`claude-agent-server`: tool calling не работает** — `claude -p` воспринимает custom tools как prompt injection. Только chat completions.
- **`codex-agent-server` на Windows:** `codex` резолвится через `shutil.which()` → `codex.CMD`. CreateProcess не дописывает PATHEXT, поэтому `subprocess.run(["codex", ...])` падает с FileNotFoundError — нельзя звать по короткому имени. Агентный `workspace-write` реально пишет файлы: `workdir` обязан быть внутри `CODEX_AGENT_WORKDIR_ROOT` (иначе 400). Глобальные MCP Codex гасятся `-c mcp_servers={}` на каждом вызове.
- **`mcp-council/dialogue/`: три не-очевидных грабли:**
  - `task.cancel()` на не-стартовавшей корутине не входит в её `try/except` — нужен `await asyncio.sleep(0)` перед cancel.
  - `tests/dialogue/` НЕ должна иметь `__init__.py` — иначе `tests.dialogue` затеняет production `dialogue/`. Basename тестов должны быть уникальны (`test_dialogue_state.py`, не `test_state.py`).
  - Failure threshold в `run_dialogue` считает distinct participants, не error-entries (один failing участник = 2 entries за раунд).

## Env keys

`DEEPSEEK_KEY`, `OPENCODE_GO_KEY`, `HELICONE_GATEWAY_KEY`, `CODEX_AGENT_TOKEN`, `EXA_API_KEY` — передаются через окружение (для MCP-сервера — `~/.claude.json` → `mcpServers.council.env`). В репозиторий не попадают. `CODEX_AGENT_TOKEN` — bearer к локальному `codex-agent-server` :8766 (член совета `codex`); без него `council_ask` с участником `codex` падает с `CouncilHTTPError`. `MINIMAX_API_KEY` нужен только для `minimax-direct`, который сейчас disabled (billing off).
