# mcp-council

MCP-сервер: Karpathy 3-stage council из 7 LLM с опциональной auto-synthesis, multi-round debate, confidence-weighted aggregation и per-model web search через Exa.ai.

## Workflow

1. **Stage 1: Independent** — 7 моделей независимо отвечают на вопрос (parallel).
2. **Stage 2: Anonymized peer-ranking** — каждая модель оценивает чужие ответы под псевдонимами Member A/B/C… (parallel, self-ranking исключён, per-ranker random shuffle псевдонимов для устранения positional bias). Каждый ranker самооценивает уверенность (1-10), которая используется как вес в aggregate.
3. **Stage 3: Synthesis** — опционально (`synthesis=true`); по умолчанию синтез делает основной Claude-агент в сессии. Если включён — chairman (модель с наивысшим weighted rank, fallback DeepSeek) сам пишет финальный ответ внутри MCP.

## Council members (7)

| id | model | provider | env_key |
|---|---|---|---|
| glm | glm-5.2 | OpenCode Go | `OPENCODE_GO_KEY` |
| kimi | kimi-k2.7-code | OpenCode Go | `OPENCODE_GO_KEY` |
| deepseek-pro | deepseek-v4-pro | OpenCode Go | `OPENCODE_GO_KEY` |
| qwen | qwen3.6-plus | OpenCode Go | `OPENCODE_GO_KEY` |
| minimax | minimax-m3 | OpenCode Go | `OPENCODE_GO_KEY` |
| gemini | gemini-3.1-pro-preview | Helicone AI Gateway | `HELICONE_GATEWAY_KEY` |
| codex | gpt-5.5 | codex-agent-server :8766 (read-only) | `CODEX_AGENT_TOKEN` |

`deepseek-pro` теперь идёт через OCG-прокси (DeepSeek direct PAYG исчерпан с 2026-06-07). **Provider-домены** (независимые точки отказа): 5 моделей на OCG (`glm`/`kimi`/`deepseek-pro`/`qwen`/`minimax`) делят один ключ и падают вместе → при OCG outage живыми голосами остаются `gemini` (Helicone) и `codex` (локальный codex-agent-server), т.е. **два независимых домена, а не только Gemini**. Именно поэтому council считает distinct provider-домены, а не число имён: `summary.provider_domains`/`single_provider`/`quorum_ok` гейтят «adopt»-вердикт (см. `_build_summary`). См. также `_pick_chairman` — DeepSeek используется как fallback chairman.

## Tools

### `council_ask` — синхронный (блокирующий, 2-8 мин)

```python
council_ask(
    question: str,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 8192,
    synthesis: bool = False,    # True → MCP сам делает stage 3 синтез
    rounds: int = 1,             # 2-3 → multi-round debate
    web_search: bool = False,    # True → каждая модель в stage 1 имеет web_search tool через Exa.ai
) -> str
```

Возвращает один markdown-документ с stage1+stage2+stage3 (опц.) + aggregate.

### `council_ask_async` — неблокирующий, через job_id

```python
council_ask_async(question, context_paths=None, max_response_tokens=8192,
                  synthesis=False, rounds=1, web_search=False) -> {"job_id": "job-…", ...}
```

Возвращает за ~50 мс. Затем:

- `council_status(job_id)` — текущая фаза, per-member progress, elapsed.
- `council_result(job_id)` — финальный markdown, когда `phase=="done"`.
- `council_cancel(job_id)` — отмена.
- `council_list_jobs(limit=20)` — список последних job'ов.

Используй когда вызывающий агент (Claude в сессии) хочет продолжать отвечать пользователю пока council работает.

Оба `council_ask`/`council_ask_async` принимают `models=[...]` (подмножество CATALOG, ≥2) или `models_preset` (`"full"` / `"diverse-3"` / `"fast-2-single-provider"` — описательные имена, НЕ рейтинг качества; легаси `best`/`balanced`/`cheap` — алиасы), взаимоисключимо.

### `model_ask` — один прямой вызов конкретной модели

```python
model_ask(model_id: str, prompt: str, context_paths=None, example_paths=None,
          max_response_tokens=4096, web_search=False) -> str
```

Один вызов модели из `models.CATALOG` (без council deliberation). Для тяжёлой суммаризации, шаблонной генерации, переводов, QA по файлам. `deepseek-flash` доступен только здесь. Заменил старые `deepseek_read/draft` и `minimax_read/draft`.

### `model_healthcheck` — пинг моделей

```python
model_healthcheck(models: list[str] | None = None) -> dict
```

Пингует каждую модель CATALOG (или подмножество) тривиальным промптом; возвращает per-model `status` (ok|disabled|no_key|auth|insufficient_balance|rate_limited|timeout|empty_response|network|circuit_open|error), `circuit_breakers` snapshot, `context_roots_configured`/`context_fail_open`, и агрегаты `ok`/`disabled`/`failed` (disabled-члены считаются отдельно, а не как failed). Использовать ДО council при подозрении на проблему провайдера.

### Dialogue tools — продолжительные диалоги моделей (async-only)

Отдельная группа для многораундовых обсуждений с anti-convergence (детали — `dialogue/`):

- `model_debate(question, participants=["glm","kimi","codex"], moderator=None, rounds=5, ...)` — 2+ модели с противоположными позициями (модератор автогенерирует), N раундов critique/response.
- `model_panel(question, participants=<7 default>, roles=None, diversity_monitor=True, devils_advocate_rotation=True, rounds=5, ...)` — 4+ моделей в свободной дискуссии, devil's advocate ротация + diversity monitor.
- `model_socratic(topic, questioner="deepseek-pro", respondent="glm", moderator=None, rounds=5, ...)` — questioner углубляет вопросами, respondent отвечает, optional moderator note+summary.
- `dialogue_continue(session_id, directive, rounds=3)` — продолжить done/interrupted-сессию ещё N раундов (считаются от `current_round`).
- `dialogue_status` / `dialogue_result` / `dialogue_cancel` / `dialogue_list_sessions` — наблюдение/выгрузка. `dialogue_result` отдаёт `warnings` (например провал финального summary) на успешном `done`.

Все 3 стартовых tool'а async (5-50 мин): возвращают `session_id`, прогресс через `dialogue_status`. `moderator`/`monitor_model` обязан отличаться от participants (fail-fast). Снапшоты — `logs/dialogues/<id>.json` (override `COUNCIL_DIALOGUES_DIR`), при рестарте незавершённые → `interrupted` (resumable через `dialogue_continue`).

### Real-time event stream (Monitor-friendly)

Каждый `council_ask_async` создаёт `logs/events/<job_id>.jsonl` (один JSON-event на строку, line-buffered). `council_ask_async` возвращает путь в поле `event_log`. Внешний наблюдатель — например Claude в основной сессии с tool `Monitor` — может `tail -F <event_log>` и реагировать на события в реальном времени без polling'а `council_status`.

Event types:
- `phase` — `{"phase": queued|stage1|stage2|stage3|done|error|cancelled, ...}`
- `stage1_member` — `{"id": "...", "model": "...", "status": "ok"|"error", "latency_ms": int, "tool_calls_count": int}`
- `stage2_ranker` — то же для ranker'ов
- `stage3` — то же для chairman synthesis
- `tool_call` — `{"member_id": ..., "name": "web_search", "query": str, "status": "ok"|"error", "num_results": int|None}` — испускается каждый раз когда модель сделала web_search во время своего stage 1
- `result_ready` — `{"status": "ok"|"error"|"cancelled", "dump_path": str|None}` — финальный event, файл затем закрывается

### Web search per-model (`web_search=True`)

Когда включено: каждая stage-1 модель получает OpenAI-style tool `web_search(query)` через Exa.ai. Модели **независимо** формулируют свои queries (probe показал что 6 моделей выдают 6 разных формулировок — от простых до boolean syntax типа `"A" OR "B"`), исполнитель в MCP дёргает Exa, отдаёт title/url/summary/highlights, модель может вызвать ещё раз или сразу написать финальный ответ. Cap `MAX_TOOL_ITERATIONS=12` tool-turns на модель (на последнем turn форсится `tool_choice="none"`, и любые всё-таки возвращённые tool_calls **отбрасываются**, а не исполняются — реальный потолок ровно 12 поисков на члена). Сверх этого — **run-wide budget** `MAX_RUN_SEARCHES=40` оплаченных (distinct) Exa-запросов на весь council (общий кэш `RunSearchCache`): при исчерпании новые distinct-запросы возвращают модели `budget_exhausted`, кэш-повторы бесплатны.

Stage 2 (peer-ranking) — **без** tools; при `synthesis=True` chairman (Stage 3) тоже получает `web_search` для фактчека спорных claim'ов (делит общий кэш со stage 1).

Trade-off: каждая модель тратит +30-90s на 1-3 search iterations. Стоимость Exa ~$0.005-0.01/query, run-budget 40 запросов ⇒ worst-case ≈ $0.20 за council. Точная стоимость (billed once per distinct query) — в `usage.web_search_cost_usd`.

## Когда применять

Архитектурное решение, спорный технический вопрос, важный code review, разбор сложного бага. **НЕ** используй для рутины (быстрых вопросов, шаблонной генерации) — дорого и медленно.

Дополнительно:
- `rounds=2` примерно удваивает время и токены — оправдано когда answers сильно расходятся в round 1, и хочется чтобы модели увидели критику и улучшили ответы.
- `synthesis=true` экономит контекст в основной сессии за счёт ещё одного API-вызова к chairman'у; теряется преимущество "chairman знает весь разговор".

## Install

```bash
cd llm_routers/mcp-council
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -v
```

## Run server (stdio)

```bash
OPENCODE_GO_KEY=<...> HELICONE_GATEWAY_KEY=<...> CODEX_AGENT_TOKEN=<...> EXA_API_KEY=<...> python server.py
```

`EXA_API_KEY` обязательный только если кто-то вызывает с `web_search=True`. Все 4 ключа задаются через переменные окружения (в Claude Code — через `~/.claude.json` → `mcpServers.council.env`).

## Design

Async-job исполнение council живёт в `state.py`; промпты стадий — в `prompts.py` (`STAGE3_SYSTEM` / `STAGE1_ROUND_N_SYSTEM`); выбор chairman — `_pick_chairman` в `council.py`.

## Sandbox

`sandbox.py` — deny-list заблокированных путей (`.env`, ключи, secrets, settings.json), лимит 50 файлов / 500 KB.

Контекстные файлы (`context_paths` / `example_paths`) — **fail-closed**: deny-list ловит секретные имена/содержимое, но не является границей доверия для приватного файла с нейтральным именем. Поэтому:

- без `COUNCIL_CONTEXT_ROOTS` — `context_paths` **запрещены** (по умолчанию);
- `COUNCIL_CONTEXT_ROOTS=<dir(s)>` (os.pathsep-разделитель) — файловый контекст разрешён, но каждый файл обязан резолвиться **внутри** одного из корней;
- `COUNCIL_CONTEXT_FAIL_OPEN=1` — вернуть старый deny-list-only режим (любой не-blacklisted файл проходит).

Эффективная поза видна в `model_healthcheck` (`context_roots_configured`, `context_fail_open`) и в startup-логе (stderr).

## Logging

- JSONL события per-call: `logs/council_YYYY-MM-DD.log` (метаданные).
- Полный дамп per-call: `logs/calls/<timestamp>-<hash>.json` (question + stage1 ответы + stage2 рейтинги + stage3 synthesis + errors + latency). Используется для анализа качества council.

## HTTP behaviour

- `DEFAULT_TIMEOUT`: connect=5s / read=600s / write=30s / pool=5s (thinking-модели через OCG могут долго держать соединение без emitting bytes; короткий connect/pool не даёт мёртвому хосту съесть весь бюджет).
- Retry on HTTP 408/429/500/502/503/504/529 + timeout: 2 попытки, backoff (15s, 45s) **+ случайный jitter 0-5s** (де-синхронизирует fan-out) и учёт заголовка `Retry-After` (capped 120s).
- **429/529 (throttling) НЕ открывают circuit breaker** — хост жив, просто просит сбавить темп; breaker копит только infra-outage (5xx/timeout/network). 402/401/400 тоже не открывают.
- HTTP 402 (insufficient balance) — без retry, сразу error.
- In-flight semaphore (`MAX_CONNECTIONS=64`) согласован с connection pool: burst из нескольких async-job'ов очередится на семафоре, а не падает в `PoolTimeout`.
- Pустой `str(exception)` в httpx ошибках заменяется на `type(e).__name__` (`ReadTimeout`, `ConnectTimeout`, …).
