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
  - **Council** (single-shot): `council_ask` (3-stage Karpathy с подмножеством моделей ≥2), `model_ask` (один прямой вызов модели из `models.CATALOG`), `model_healthcheck` (пинг моделей CATALOG: ключ/статус/latency + circuit-breakers snapshot), `council_estimate` (dry-run оценка стоимости до запуска), `council_capabilities` (машиночитаемый снимок моделей/пресетов/лимитов + ЖИВОЙ список tool'ов — авторитетнее любого списка в доках) и `council_purge_logs` (ручная зачистка артефактов сверх авто-purge на старте). Async-pattern для council: `council_ask_async` + `council_status/result/cancel/list_jobs`.
  - **Critique** (независимая адверсариальная критика): `council_critique` — N критиков с РАЗНЫМИ линзами (`lenses.py::LENSES`) ищут дефекты вслепую → питоновый кросс-линзовый дедуп → каждую находку атакуют верификаторы, которые её не поднимали, с задачей ОПРОВЕРГНУТЬ. Отличается от `council_ask` мандатом: там N моделей отвечают на один вопрос и ранжируют друг друга, здесь у каждого критика свой мандат и вместо ранжирования идёт попытка опровержения. `human_review_required` всегда True. Async — `council_critique_async` + те же `council_status/result/cancel/list_jobs`.
  - **Dialogue** (продолжительные диалоги с anti-convergence): `model_debate` / `model_panel` / `model_socratic` + `dialogue_continue/fork/status/result/cancel/list_sessions`. Все starter-tool'ы async-only (5-50 мин). Hard cap rounds=20, активных сессий=20. `dialogue_continue` возобновляет сессию в фазе `done` **или** `interrupted`.
- `claude-code-router/` — HTTP-прокси `ccr` (npm `@musistudio/claude-code-router` ставится глобально; в репо — только `config.example.json` + `custom_router.js`).
- `claude-agent-server/` — обёртка `claude -p` в OpenAI-compatible API на :8765.
- `codex-agent-server/` — обёртка `codex exec` в OpenAI-compatible API на :8766. Два режима: read-only (дефолт, чистый чат) и workspace-write (агент правит файлы); выбирается именем модели (`gpt-5.6-sol` vs `gpt-5.6-sol-agent`) или полем `sandbox` в body, `tools` форсят read-only.
- `bench/` — раннер LLM-бенчмарков, пишет markdown-отчёт в корень репо.
- `tools/model_freshness.py` — сторож свежести каталога моделей: листинги OCG/Helicone (двусторонний диф — и новые версии, и исчезнувшие id) плюс CLI-пробы `codex`/`claude` на следующие версии (у подписочных CLI листинга нет, а «модель существует» ≠ «аккаунту доступна»). Ничего не переключает — пишет P3 в `FINDINGS.md`. Weekly-таск `ModelFreshnessCheck`, вс 05:00.

## Точки входа

- Детальные правила (реестр моделей, sandbox-лимиты, ключи, тесты): [CLAUDE.md](CLAUDE.md).

## Project-specific gotchas

- **Удалённые пакеты:** `mcp-deepseek` и `mcp-minimax` удалены. Tool refs `mcp__deepseek-helper__*` и `mcp__minimax-helper__*` больше не существуют — использовать `mcp__council__model_ask` с нужным `model_id` из `models.CATALOG`.
- **`minimax-direct` отключён** в `CATALOG` (billing off) — не передавать как `model_id` в `model_ask`, выпадет ошибка.
- **Смена модели в каталоге — не только про номер версии, но и про квоту.** На подписке OCG у `kimi-k3` 2× usage и 220 запросов/5ч (у `kimi-k2.7-code` — 1150), у `grok-4.5` — 120. Для рутинных/цикличных вызовов брать дешёвые слоты (`deepseek-flash`), дорогие оставлять совету. У Codex CLI имя модели зависит от типа аккаунта: на подписке ChatGPT работает `gpt-5.6-sol`, голые `gpt-5.6`/`gpt-5.6-codex` → 400.
- **`claude-agent-server`: tool calling — три состояния** (не «просто не работает»): native Anthropic tool use — нет; эмулируемый через prompt-injection — есть, best-effort (надёжность ≈7/12 на `test_server.py`); для критичных workflow — только text/chat completions. Детали и измеренная надёжность — `claude-agent-server/README.md`.
- **`codex-agent-server` на Windows:** `codex` резолвится через `shutil.which()` → `codex.CMD`. CreateProcess не дописывает PATHEXT, поэтому `subprocess.run(["codex", ...])` падает с FileNotFoundError — нельзя звать по короткому имени. Агентный `workspace-write` реально пишет файлы: `workdir` обязан быть внутри `CODEX_AGENT_WORKDIR_ROOT` (иначе 400). Глобальные MCP Codex гасятся `-c mcp_servers={}` на каждом вызове.
- **MCP SDK: поддерживаются оба мажора, это намеренно.** `mcp` 2.0 переименовал `mcp.server.fastmcp.FastMCP` → `mcp.server.mcpserver.MCPServer`; `server.py` пробует оба имени, спека — `mcp[cli]>=1.0,<3`, CI гоняет матрицу по 1.x и 2.x. Не «упрощать» до одного импорта: тот же интерпретатор обслуживает другие локальные проекты, которые всё ещё на `mcp.server.fastmcp`. Потолок по мажору не снимать — именно его отсутствие уронило CI на выходе 2.0.0.
- **`mcp-council/dialogue/`: три не-очевидных грабли:**
  - `task.cancel()` на не-стартовавшей корутине не входит в её `try/except` — нужен `await asyncio.sleep(0)` перед cancel.
  - `tests/dialogue/` НЕ должна иметь `__init__.py` — иначе `tests.dialogue` затеняет production `dialogue/`. Basename тестов должны быть уникальны (`test_dialogue_state.py`, не `test_state.py`).
  - Failure threshold в `run_dialogue` считает distinct participants, не error-entries (один failing участник = 2 entries за раунд).

## Env keys

`OPENCODE_GO_KEY` (glm, kimi, deepseek-pro, qwen, minimax, deepseek-flash — все через OCG), `HELICONE_GATEWAY_KEY`, `CODEX_AGENT_TOKEN`, `EXA_API_KEY` — передаются через окружение (для MCP-сервера — `~/.claude.json` → `mcpServers.council.env`). В репозиторий не попадают. `CODEX_AGENT_TOKEN` — bearer к локальному `codex-agent-server` :8766 (член совета `codex`); без него `council_ask` с участником `codex` падает с `CouncilHTTPError`. `MINIMAX_API_KEY` нужен только для `minimax-direct`, который сейчас disabled (billing off).
