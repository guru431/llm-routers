# CLAUDE.md — llm_routers

## ⚠️ Публичный репозиторий — не допускать утечек

Этот репозиторий **публичный** (GitHub). При любом редактировании НЕ коммить:
- секреты, ключи, токены, пароли (только через env-переменные / `*.example.env` с пустыми значениями);
- реальные приватные IP (`192.168.x`, `10.x`) и внутренние хосты/SSH-порты;
- внутренние домены и персональные данные (email, телефон, ФИО, адреса);
- имена внутренних проектов, серверов и людей.

Защита — pre-commit hook [`.githooks/pre-commit`](.githooks/pre-commit): generic-сканер форматов ключей + локальный `.sanitize-patterns` (gitignored denylist конкретных значений). После клона активировать: `git config core.hooksPath .githooks`. Сам `.sanitize-patterns` НИКОГДА не коммить.

Зонтичный проект для MCP-серверов routing'а LLM-запросов из Claude Code.

## Что внутри

- `mcp-council/` — единый MCP-сервер с двумя группами tool'ов:
  - **Council (single-shot deliberation):**
    - **`council_ask(question, models=None, models_preset=None, ...)`** — Karpathy 3-stage council из 7 моделей (или подмножества через `models=[...]`, **минимум 2**; либо `models_preset="full|diverse-3|fast-2-single-provider"` вместо ручного списка — взаимоисключимо с `models`; имена описательные, легаси `best|balanced|cheap` работают как алиасы). Для архитектурных решений, спорных вопросов, важного code review, debug сложных багов. **НЕ для рутины** (2-8 мин, дорого). Результат содержит machine-readable `summary` (winner/`confidence`(=алиас `agreement_confidence` — это МЕРА СОГЛАСИЯ ранкеров, не корректности)/failed_models[+typed `failure_reason`]/top_disagreements/recommended_next_action/`analysis` + corroboration-сигналы `independent_votes`/`winner_ranked_by`/`provider_domains`/`single_provider`/`quorum_ok`/`incomplete_rankings` + фактическая глубина `rounds_requested`/`rounds_attempted`/`rounds_completed`/`stop_reason`) и `usage` (llm_calls/tokens/web_search/retries/cache_hits/`web_search_cost_usd`/`reference_payg_cost_usd`+`reference_payg_priced_calls`). **Кворум:** `independent_votes` — это число ранкеров, поставивших winner СВОЙ высший балл (≥6/10), а не просто упомянувших его в списке: при полных рейтингах каждый кандидат присутствует у каждого ранкера, так что «присутствие» не измеряло ничего. Ранкер с плоским рейтингом (всем одинаково) и ранкер с неполным списком в поддержку не засчитываются; `winner_ranked_by` показывает более слабый счёт «упомянут вообще». `confidence` не станет `high` и `recommended_next_action` не станет «adopt», пока winner не набрал ≥2 таких голосов И ≥2 provider-доменов (5 OCG-моделей = 1 домен, не 5 голосов) — защищает 2-модельный `fast-2-single-provider` preset и OCG-outage от ложного вердикта. Непустой `incomplete_rankings` (ранкер пропустил кого-то из peers — сначала один repair-повтор, затем принимается с пометкой) тоже срезает `high` до `medium` и включает human review: средние сравниваются на разном числе оценок. `rounds_completed`/`stop_reason` (`completed|budget|round_collapse`) отделяют фактически проведённые раунды от запрошенных — recommendation больше не предлагает «раунд N+1», которого не было. `reference_payg_cost_usd` — это НЕ списанные деньги (все члены на flat-rate подписке), а справочная PAYG-оценка; реальный incremental расход ≈ `web_search_cost_usd` (Exa). При `synthesis=True` chairman дополнительно отдаёт структурный `analysis` (consensus/contradictions/partial_coverage/unique_insights/blind_spots) и получает web_search для фактчека спорных claim'ов.
    - **`model_ask(model_id, prompt, context_paths=[], example_paths=[], ...)`** — один прямой вызов конкретной модели из `models.CATALOG`. Для тяжёлой суммаризации логов, шаблонной генерации, переводов. Заменяет старые `deepseek_read/draft` и `minimax_read/draft` (пакеты `mcp-deepseek` и `mcp-minimax` удалены 2026-05-21).
    - **`model_healthcheck(models=None)`** — пинг каждой модели CATALOG (или подмножества) тривиальным промптом: ключ/HTTP-статус/latency/empty-response. Возвращает per-model `status` (ok|disabled|no_key|auth|insufficient_balance|rate_limited|timeout|empty_response|network|circuit_open|error) + `circuit_breakers` snapshot + `context_roots_configured`. Использовать ДО council при подозрении на проблему провайдера.
    - **`council_critique(subject, lenses=None, lenses_preset=None, models=None, verifiers_per_finding=2, ...)`** — независимая адверсариальная критика. Три стадии: (1) N критиков с РАЗНЫМИ линзами (`lenses.py::LENSES` — correctness, security, concurrency, failure-modes, performance, data-integrity, api-contract, simplicity, testing, observability) ищут дефекты вслепую, у каждой линзы явный `out_of_scope` — это и не даёт критикам сойтись в одно мнение; (2) чисто питоновый кросс-линзовый дедуп (LLM-дедупер сам требовал бы верификации); (3) каждую находку атакуют `verifiers_per_finding` моделей, которые её НЕ поднимали, каждая под своим углом (`does-not-reproduce` / `already-handled` / `misreads-the-code` / `not-reachable` / `wrong-severity`) и с инструкцией «при сомнении → refuted». Находка отбрасывается при опровержении половиной и больше — но остаётся видимой в секции Refuted, потому что ошибочное опровержение реального бага и есть главный failure mode режима. Линзы раскладываются по provider-доменам с чередованием; `summary.panel_quorum_ok=False` (панель <2 доменов) помечает прогон как один коррелированный источник, а не как ревью. `human_review_required` всегда True: верификация фильтрует шум моделей, а не доказывает корректность. Пресеты линз — `lenses_preset`: `code-review` (дефолт), `security-audit`, `design-review`, `reliability`, `fast-3`. Async — `council_critique_async` + те же `council_status/result/cancel`.
    - **Чем отличается от `council_ask`:** council_ask — N моделей отвечают на один вопрос с ОДНИМ мандатом и ранжируют друг друга («какой ответ лучше»); council_critique — у каждого критика свой мандат, и вместо ранжирования идёт попытка опровержения («что здесь сломано и что из этого выживет»). Разные инструменты, не замена друг другу.
    - Async-pattern для council: `council_ask_async` + `council_status/result/cancel/list_jobs`. Job-state персистится на диск (`logs/jobs/`, override `COUNCIL_JOBS_DIR`); при рестарте сервера незавершённые задачи помечаются `interrupted` (partial-прогресс через `council_status`).
  - **Dialogue (продолжительные диалоги между моделями с anti-convergence):**
    - **`model_debate(question, participants=None, moderator=None, rounds=5, ...)`** — 2+ моделей с противоположными позициями (модератор автогенерирует), N раундов critique/response. Default participants `["glm","kimi","codex"]`.
    - **`model_panel(question, participants=None, roles=None, diversity_monitor=True, devils_advocate_rotation=True, rounds=5, ...)`** — 4+ моделей в свободной дискуссии (min 4 distinct, жёсткого верхнего потолка нет; default = DEFAULT_PANEL_PARTICIPANTS = 7, вкл. codex), devil's advocate ротация + diversity monitor (re-prompt согласившимся). `monitor_model` обязан отличаться от participants (иначе участник оценивает собственное согласие) — как и `moderator` в `model_debate`; fail-fast на пересечении во всех трёх режимах.
    - **`model_socratic(topic, questioner=None, respondent=None, moderator=None, rounds=5, ...)`** — questioner задаёт углубляющие вопросы, respondent отвечает, optional moderator пишет note + summary. Default: deepseek-pro / glm.
    - **`dialogue_continue(session_id, directive, rounds=3)`** — продолжить сессию в терминальной фазе **`done` или `interrupted`** (последняя = прогон, умерший при рестарте сервера; история и параметры персистятся, поэтому он возобновляем так же) ещё N раундов с user-directive ("углубитесь в X"). N считается от `current_round`, не от `total_rounds`.
    - **`dialogue_status/result/cancel/list_sessions`** — наблюдение/выгрузка/отмена. Все 3 starter tool'а async-only (длительность 5-50 мин). Hard cap rounds=20, активных сессий=20.
- `claude-code-router/` — HTTP-прокси `ccr` (npm `@musistudio/claude-code-router`, ставится глобально). В репо живут `config.example.json` (template без ключей) + `custom_router.js` (симлинк в `~/.claude-code-router/`). Реальный `config.json` с ключами — только в `~/.claude-code-router/`, в git не попадает. См. [claude-code-router/README.md](claude-code-router/README.md) для деталей привязки и потоков запросов.
- `claude-agent-server/` — обёртка `claude -p` CLI в OpenAI-compatible HTTP API (port 8765). Превращает подписку Claude Max/Pro в локальный endpoint для n8n, чат-ботов и любых OpenAI-клиентов. Опционально — boot-task в Windows Task Scheduler (`install_task.ps1`). **Tool calling — три состояния:** (1) native Anthropic tool use — **нет**; (2) эмулируемый через prompt-injection — **есть, но best-effort** (описания функций в system-prompt, парсинг `<tool_call>`, измеренная надёжность ≈7/12 на `test_server.py`); (3) для критичных workflow — только text/chat completions или другой backend. Источник истины по надёжности — [claude-agent-server/README.md](claude-agent-server/README.md).
- `codex-agent-server/` — обёртка `codex exec` CLI в OpenAI-compatible HTTP API (port 8766). Превращает подписку ChatGPT/Codex в локальный endpoint. Один API, два режима: **read-only** (чистый чат — дефолт) и **workspace-write** (агент правит файлы). Режим выбирается именем модели (`gpt-5.5` vs `gpt-5.5-agent`) или полем `sandbox` в body; `tools` всегда форсят read-only. Для агентного — containment-проверка `workdir` внутри `CODEX_AGENT_WORKDIR_ROOT`. Дизайн-спека — локально в `docs/superpowers/specs/2026-05-31-codex-agent-server-design.md` (каталог `docs/` в `.gitignore` — в публичный клон не попадает). См. [codex-agent-server/README.md](codex-agent-server/README.md).
- `bench/` — раннер LLM-бенчмарков (`run.py`, `judge.py`, `report.py`, `models.json`, `prompts/`, `results/`). Пишет markdown-отчёт в корень репо. **Lifecycle прогона:** `manifest.json` несёт `status` (`started|completed|failed`) + `expected_cells`/`completed_cells`; `results/runs/latest.txt` = последний ЗАПУЩЕННЫЙ прогон (может быть частичным), `latest-complete.txt` = последний ДОШЕДШИЙ до конца. `judge.py`/`report.py` по умолчанию берут complete-указатель и громко предупреждают, если считают неполную матрицу. Прерванный прогон дособирается `python run.py --resume <run-id>` (не новый UUID). Каждый повтор `--repeats` судится отдельно, поэтому bootstrap-CI качества — реально repeat-based, а не разброс по задачам. Ground truth для детерминированного гейта живёт в `prompts/tasks.json::expected` (значения, а не только форма ответа).

## Реестр моделей (`mcp-council/models.py`)

Единый `CATALOG` — source of truth для обоих tool'ов:

| id | model | назначение |
|---|---|---|
| `glm` | glm-5.2 | council member (OCG) |
| `kimi` | kimi-k2.7-code | council member (OCG) |
| `deepseek-pro` | deepseek-v4-pro | council member (OCG) |
| `qwen` | qwen3.6-plus | council member (OCG) |
| `minimax` | minimax-m3 | council member (OCG) |
| `gemini` | gemini-3.1-pro-preview | council member (Helicone Gateway) |
| `codex` | gpt-5.5 | council member (codex-agent-server :8766, read-only) |
| `deepseek-flash` | deepseek-v4-flash | routine worker (model_ask only) |
| `minimax-direct` | abab7-chat-preview | disabled (billing off) |

`COUNCIL_DEFAULT` = первые 7 (без flash и direct). При `models=None` в `council_ask` совещание идёт ровно по этому списку. **`codex` требует запущенного `codex-agent-server` на :8766** — если он недоступен, member падает с `CouncilHTTPError` и совет продолжает остальными.

## Принципы

- **В основном stateless** — каждый одиночный MCP-вызов (`council_ask`, `model_ask`, healthcheck) независим. Реальные исключения с process/disk-состоянием: (1) process-global circuit breaker (`circuit_breaker.py`) — после `FAILURE_THRESHOLD` подряд infra-ошибок (5xx/timeout/network) хост degraded на `COOLDOWN_SECONDS`, вызовы short-circuit'ятся (402/401/400 breaker не открывают); (2) async job store (`state.py`, `council_ask_async`) — job-состояние в памяти + снапшоты на диск (`logs/jobs/`); (3) dialogue sessions (`dialogue/state.py`) — то же (`logs/dialogues/`). При рестарте сервера незавершённые job/session помечаются `interrupted` (partial-результат доступен). Ещё process-global: HTTP connection pool + in-flight semaphore (`openai_client.py`).
- **Sandbox** — `sandbox.py` блокирует `.env`, ключи, secrets, settings.json. Лимит 50 файлов / 500 KB суммарно. Контекстные файлы **fail-closed**: без `COUNCIL_CONTEXT_ROOTS` `context_paths` запрещены (deny-list не является границей доверия для нейтрально-именованных приватных файлов). Задать `COUNCIL_CONTEXT_ROOTS`=<workspace dir(s)> (os.pathsep-разделитель) чтобы включить файловый контекст с проверкой «внутри корня»; `COUNCIL_CONTEXT_FAIL_OPEN=1` — вернуть старый deny-list-only режим. Эффективная поза видна в `model_healthcheck` (`context_roots_configured`, `context_fail_open`) и в startup-логе (stderr). Граница проверяется **в момент чтения**, а не только при валидации пути: `read_files_with_limit` открывает файл один раз и заново применяет все правила к тому, что реально открыто (fstat — обычный ли это файл, deny-list и roots по пути, private-key/binary sniff по прочитанным байтам), поэтому подмена пути на symlink/junction между валидацией и чтением не протаскивает чужой файл. Наружу уходит путь **относительно matched root** (или basename), а не абсолютный — локальная раскладка каталогов и имя пользователя не отправляются провайдерам.
- **Retention/redaction логов** — `logs/calls/*.json` (полные дампы), корневой `logs/council_*.log`, `logs/jobs|dialogues|events` подпадают под TTL (`COUNCIL_LOG_RETENTION_HOURS`, дефолт 168ч) и квоту (`COUNCIL_LOG_DIR_QUOTA_BYTES`). Purge выполняется **на старте сервера** и по требованию через `council_purge_logs`. Дампы и JSONL пишутся уже отредактированными (`retention.redact`; `COUNCIL_LOG_REDACT=0` отключает — только для отладки). Снапшоты `jobs/`/`dialogues/` НЕ редактируются намеренно: это рабочее состояние, которое должно round-trip'иться дословно (восстановленный job отдаёт результат клиенту), их ограничивает только TTL. Битый снапшот при загрузке уезжает в `<dir>/corrupt/` и не роняет старт.
- **Single source of truth** — `models.py::CATALOG` хранит всех моделей, дубликатов `sandbox.py`/`logger.py` больше нет. Пресеты совета — `models.py::PRESETS`, линзы критики — `lenses.py::LENSES` / `LENS_PRESETS`.
- **Корреляция ≠ независимость** — и совет, и критика считают distinct provider-домены (`models.py::provider_domain`), а не число моделей: 5 OCG-членов делят один шлюз и один ключ, падают вместе и соглашаются по коррелированным причинам. Отсюда `quorum_ok` в council и `panel_quorum_ok` / `verification_quorum_ok` в critique.

## Ключи

Все ключи читаются `mcp-council` (передаются через `~/.claude.json` → `mcpServers.council.env`):
- `OPENCODE_GO_KEY` — для glm, kimi, deepseek-pro, qwen, minimax, deepseek-flash (через OCG)
- `HELICONE_GATEWAY_KEY` — для gemini
- `CODEX_AGENT_TOKEN` — для codex (bearer к локальному codex-agent-server :8766)
- `MINIMAX_API_KEY` — для minimax-direct (currently disabled в catalog, billing off)
- `EXA_API_KEY` — для web_search в любом tool

Значения передаются через окружение MCP-сервера и в репозиторий не попадают.

## Registration в Claude Code

MCP-сервер регистрируется в `~/.claude.json` под top-level `mcpServers` как `council`. Tool-пути: `mcp__council__council_ask`, `mcp__council__council_critique`, `mcp__council__model_ask`, `mcp__council__model_healthcheck`, `mcp__council__council_ask_async`, `mcp__council__council_critique_async`, `mcp__council__council_status/result/cancel/list_jobs`.

## Tests

Канонический прогон всех pytest-сьютов — из корня:

```bash
python run_tests.py              # --quick (дефолт): pytest без live-серверов
python run_tests.py --full       # + compileall по всем пакетам
python run_tests.py --integration  # live codex-agent-server/integration_suite.py (нужен сервер :8766 + токен)
```

Сьюты: `mcp-council`, `claude-agent-server`, `codex-agent-server`, `bench`. Подпроекты — независимые пакеты с одноимёнными модулями (`server.py` ×3, `cache.py`), поэтому один процесс pytest их не соберёт (коллизия `sys.modules`). `run_tests.py` запускает каждый сьют отдельным интерпретатором. Отдельный подпроект:

```bash
cd mcp-council
pip install -e ".[dev]"
pytest -v
```

`codex-agent-server/integration_suite.py` — live-сьют (CLI, не pytest), бьёт по запущенному серверу :8766, в `run_tests.py` не входит.

## Связанные документы

- Бенчи моделей: раннер [`bench/`](bench/)
- Per-model quirks (thinking/reasoning_effort) задаются в `mcp-council/models.py::CATALOG`.

## Coexistence

См. [AGENTS.md](AGENTS.md) для общих правил (для Codex CLI и других AI-агентов).
