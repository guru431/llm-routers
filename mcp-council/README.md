# mcp-council

MCP-сервер: Karpathy 3-stage council из 7 LLM с опциональной auto-synthesis, multi-round debate, confidence-weighted aggregation и per-model web search через Exa.ai. Плюс `council_critique` — независимая адверсариальная критика: N критиков с разными линзами → кросс-линзовый дедуп → попытка опровержения каждой находки.

## Workflow

1. **Stage 1: Independent** — 7 моделей независимо отвечают на вопрос (parallel).
2. **Stage 2: Anonymized peer-ranking** — каждая модель оценивает чужие ответы под псевдонимами Member A/B/C… (parallel, self-ranking исключён, per-ranker random shuffle псевдонимов для устранения positional bias). Каждый ranker самооценивает уверенность (1-10), которая используется как вес в aggregate.
3. **Stage 3: Synthesis** — опционально (`synthesis=true`); по умолчанию синтез делает основной Claude-агент в сессии. Если включён — chairman (модель с наивысшим weighted rank, fallback DeepSeek) сам пишет финальный ответ внутри MCP.

## Council members (7)

| id | model | provider | env_key |
|---|---|---|---|
| glm | glm-5.2 | OpenCode Go | `OPENCODE_GO_KEY` |
| kimi | kimi-k3 | OpenCode Go | `OPENCODE_GO_KEY` |
| deepseek-pro | deepseek-v4-pro | OpenCode Go | `OPENCODE_GO_KEY` |
| qwen | qwen3.7-plus | OpenCode Go | `OPENCODE_GO_KEY` |
| minimax | minimax-m3 | OpenCode Go | `OPENCODE_GO_KEY` |
| gemini | gemini-3.1-pro-preview | Helicone AI Gateway | `HELICONE_GATEWAY_KEY` |
| codex | gpt-5.6-sol | codex-agent-server :8766 (read-only) | `CODEX_AGENT_TOKEN` |

`deepseek-pro` теперь идёт через OCG-прокси (DeepSeek direct PAYG исчерпан с 2026-06-07). **Provider-домены** (независимые точки отказа): 5 моделей на OCG (`glm`/`kimi`/`deepseek-pro`/`qwen`/`minimax`) делят один ключ и падают вместе → при OCG outage живыми голосами остаются `gemini` (Helicone) и `codex` (локальный codex-agent-server), т.е. **два независимых домена, а не только Gemini**. Именно поэтому council считает distinct provider-домены, а не число имён: `summary.provider_domains`/`single_provider`/`quorum_ok` гейтят «adopt»-вердикт (см. `_build_summary`). См. также `_pick_chairman` — `deepseek-pro` используется как fallback chairman (это предпочтение по доступности внутри дефолтного состава, а НЕ независимый провайдер: он ходит через тот же OCG-шлюз).

**Что считается голосом.** `summary.independent_votes` — число ранкеров, поставивших winner СВОЙ высший балл (и не ниже 6/10). Просто «winner присутствует в списке ранкера» голосом не считается: при полных рейтингах там присутствуют все, включая последнее место. Не засчитываются также плоские рейтинги (всем одинаково — предпочтение не выражено) и ранкеры с неполным списком. Более слабый счёт «упомянут вообще» доступен как `winner_ranked_by`, а пропуски — как `incomplete_rankings` (непустой список срезает `confidence` с `high` до `medium` и включает `human_review_required`: средние тогда сравниваются на разном числе оценок).

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

### `council_critique` — независимая адверсариальная критика (3-10 мин)

```python
council_critique(
    subject: str,
    context_paths: list[str] | None = None,
    lenses: list[str] | None = None,          # ≥2 из lenses.LENSES
    lenses_preset: str | None = None,         # code-review | security-audit | design-review | reliability | fast-3
    models: list[str] | None = None,          # ≥2; None → все 7
    models_preset: str | None = None,
    verifiers_per_finding: int = 2,           # 0..5; 0 = пропустить верификацию
    max_verified_findings: int = 24,
    max_response_tokens: int = 8192,
    web_search: bool = False,
    deadline_seconds / max_cost_usd / max_web_searches = None,
) -> str   # markdown-отчёт
```

Три стадии:

1. **Lensed critics (independent)** — каждому критику выдаётся СВОЙ мандат из `lenses.py::LENSES` с явным `out_of_scope`. Именно `out_of_scope` не даёт N критикам сойтись в один общий список «вот пара багов, и добавьте тестов»: без него разные модели с одним промптом дают одно мнение за N latency. Критики не видят друг друга. Линзы раскладываются по моделям с **чередованием provider-доменов**, так что уже 2 линзы садятся на 2 независимых домена.
2. **Кросс-линзовый дедуп (pure Python, без LLM)** — находки схлопываются по token-overlap + нормализованному location. Дедупер намеренно консервативный: слить два разных бага хуже, чем показать почти-дубль. LLM-дедупер тут не годится — он сам требовал бы верификации.
3. **Adversarial verification** — каждую находку атакуют `verifiers_per_finding` моделей, которые её **не поднимали**, каждая под своим углом: `does-not-reproduce` / `already-handled` / `misreads-the-code` / `not-reachable` / `wrong-severity`. Инструкция явно асимметричная — «при сомнении ставь `refuted: true`», потому что ложная находка стоит человеку сессии отладки. Опровержение половиной и больше → находка выпадает из основного отчёта, но **остаётся видимой в секции Refuted**: ошибочно опровергнутый реальный баг — главный failure mode этого режима.

Доступные линзы: `correctness`, `security`, `concurrency`, `failure-modes`, `performance`, `data-integrity`, `api-contract`, `simplicity`, `testing`, `observability`.

Что в `summary`: `findings_kept`/`findings_refuted`, `by_severity`, `by_status`, `cross_lens_corroborated` (находки, до которых дошли ≥2 линзы — самый сильный сигнал режима), `lenses_with_findings` vs `lenses_with_surviving_findings`, `panel_quorum_ok` (панель охватила ≥2 provider-домена; иначе отчёт помечается как один коррелированный источник, а не ревью), `verification_quorum_ok` на каждой находке, и `human_review_required` — **всегда True**. Верификация фильтрует шум моделей; корректность она не доказывает.

`council_critique_async` — то же в фоне: возвращает `job_id`, дальше `council_status`/`council_result`/`council_cancel` (тот же job-store, что у `council_ask_async`; критики видны как stage1, верификаторы как stage2).

**Чем отличается от `council_ask`:** там N моделей отвечают на один вопрос с ОДНИМ мандатом и ранжируют друг друга — «какой ответ лучше». Здесь у каждого критика свой мандат, и вместо ранжирования идёт попытка опровержения — «что здесь сломано и что из этого выживет». Это разные инструменты, а не замена друг другу.

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

Каждый `council_ask_async` / `council_critique_async` создаёт `logs/events/<job_id>.jsonl` (один JSON-event на строку, line-buffered). `council_ask_async` возвращает путь в поле `event_log`. Внешний наблюдатель — например Claude в основной сессии с tool `Monitor` — может `tail -F <event_log>` и реагировать на события в реальном времени без polling'а `council_status`.

Event types:
- `phase` — `{"phase": queued|stage1|stage2|stage3|done|error|cancelled, ...}`
- `stage1_member` — `{"id": "...", "model": "...", "status": "ok"|"error", "latency_ms": int, "tool_calls_count": int}`
- `stage2_ranker` — то же для ranker'ов
- `stage3` — то же для chairman synthesis
- `tool_call` — `{"member_id": ..., "name": "web_search", "query": str, "status": "ok"|"error", "num_results": int|None}` — испускается каждый раз когда модель сделала web_search во время своего stage 1
- `result_ready` — `{"status": "ok"|"error"|"cancelled", "dump_path": str|None}` — финальный event, файл затем закрывается

Для `council_critique_async` те же типы переиспользованы: `phase` даёт `critique|dedup|verify|done`, критики приходят как `stage1_member` (id = `<lens>@<model>`), верификаторы как `stage2_ranker` (id = `verify<N>:<angle>@<model>`, плюс поле `refuted`).

### Web search per-model (`web_search=True`)

Когда включено: каждая stage-1 модель получает OpenAI-style tool `web_search(query)` через Exa.ai. Модели **независимо** формулируют свои queries (probe показал что 6 моделей выдают 6 разных формулировок — от простых до boolean syntax типа `"A" OR "B"`), исполнитель в MCP дёргает Exa, отдаёт title/url/summary/highlights, модель может вызвать ещё раз или сразу написать финальный ответ. Cap `MAX_TOOL_ITERATIONS=12` — это **tool-TURN'ов** на модель, а не поисков: модель вправе вернуть несколько `tool_calls` в одном turn'е, и цикл исполняет их все, так что поисков на члена может быть больше 12 (на последнем turn форсится `tool_choice="none"`, и любые всё-таки возвращённые tool_calls **отбрасываются**, а не исполняются). Жёсткий потолок — **run-wide budget** `MAX_RUN_SEARCHES=40` оплаченных (distinct) Exa-запросов на весь прогон (общий кэш `RunSearchCache`): при исчерпании новые distinct-запросы возвращают модели `budget_exhausted`, кэш-повторы бесплатны.

`usage.web_search_calls` — число обращений к инструменту (включая ошибки и DLP-блокировки), `usage.web_search_ok` — сколько из них реально вернули результат; оплачиваются только distinct-запросы, см. `web_search_cost_usd`.

Stage 2 (peer-ranking) — **без** tools; при `synthesis=True` chairman (Stage 3) тоже получает `web_search` для фактчека спорных claim'ов (делит общий кэш со stage 1).

Trade-off: каждая модель тратит +30-90s на 1-3 search iterations. Стоимость Exa ~$0.005-0.01/query, run-budget 40 запросов ⇒ worst-case ≈ $0.20 за council. Точная стоимость (billed once per distinct query) — в `usage.web_search_cost_usd`.

## Когда применять

`council_ask` — архитектурное решение, спорный технический вопрос, разбор сложного бага: когда нужен **лучший ответ** на открытый вопрос.

`council_critique` — ревью значимого диффа/модуля, security-аудит, проверка дизайна перед реализацией, «что мы упустили»: когда нужно **найти дефекты и отсеять выдуманные**.

**НЕ** используй ни то, ни другое для рутины (быстрых вопросов, шаблонной генерации) — дорого и медленно.

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

**Проверки применяются к тому, что реально открыто.** `resolve_and_validate` — ранний дешёвый гейт; настоящая граница — `read_files_with_limit`: файл открывается ОДИН раз, и по этому открытию заново проверяются тип (`fstat` — обычный ли файл), deny-list и allowed roots по пути, private-key/binary sniff по прочитанным байтам и суммарный размер. Поэтому подмена пути на symlink/junction между валидацией и чтением не даёт прочитать чужой файл.

**Наружу уходит относительный путь.** В промпт (и, соответственно, всем внешним провайдерам) заголовок файла пишется как путь относительно matched root — или как basename, если корней нет. Абсолютный путь раскрывал имя пользователя, внутреннее имя проекта и раскладку каталогов, ничего не давая модели; он остаётся только в локальных audit-метаданных дампа.

## Logging

- JSONL события per-call: `logs/council_YYYY-MM-DD.log` (метаданные).
- Полный дамп per-call: `logs/calls/<timestamp>-<hash>.json` (question + stage1 ответы + stage2 рейтинги + stage3 synthesis + errors + latency). Используется для анализа качества council.

### Retention и redaction

| Что | Где | Под ретеншеном |
|---|---|---|
| полные дампы вызовов | `logs/calls/*.json` | да |
| per-day JSONL журнал | `logs/council_*.log` (корень `logs/`) | да |
| снапшоты async-job | `logs/jobs/*.json` | да |
| дампы диалогов | `logs/dialogues/*.json` | да |
| event-журналы | `logs/events/*.jsonl` | да |

- **TTL** — `COUNCIL_LOG_RETENTION_HOURS` (дефолт `168` = 7 дней; `0` отключает).
- **Квота на каталог** — `COUNCIL_LOG_DIR_QUOTA_BYTES` (дефолт 256 MB), удаление oldest-first после TTL-прохода.
- **Когда чистится** — автоматически при **старте сервера** и по требованию через `council_purge_logs`. Настроенный TTL — реальный срок хранения, а не обещание, действующее только когда кто-то вручную дёрнул tool.
- **Redaction** — дампы и JSONL записываются уже с замаскированными credential-подобными токенами (`retention.redact` поверх сериализованного JSON). `COUNCIL_LOG_REDACT=0` пишет как есть — только для отладки.
- Снапшоты `jobs/` и `dialogues/` **намеренно не редактируются**: это рабочее состояние, которое обязано round-trip'иться дословно (восстановленный job отдаёт свой результат клиенту), их ограничивает TTL/квота. Структурно битый снапшот при загрузке переезжает в `<dir>/corrupt/` и не мешает старту сервера.

## HTTP behaviour

- `DEFAULT_TIMEOUT`: connect=5s / read=600s / write=30s / pool=5s (thinking-модели через OCG могут долго держать соединение без emitting bytes; короткий connect/pool не даёт мёртвому хосту съесть весь бюджет).
- Retry on HTTP 408/429/500/502/503/504/529 + timeout: 2 попытки, backoff (15s, 45s) **+ случайный jitter 0-5s** (де-синхронизирует fan-out) и учёт заголовка `Retry-After` (capped 120s).
- **429/529 (throttling) НЕ открывают circuit breaker** — хост жив, просто просит сбавить темп; breaker копит только infra-outage (5xx/timeout/network). 402/401/400 тоже не открывают.
- HTTP 402 (insufficient balance) — без retry, сразу error.
- In-flight semaphore (`MAX_CONNECTIONS=64`) согласован с connection pool: burst из нескольких async-job'ов очередится на семафоре, а не падает в `PoolTimeout`.
- Pустой `str(exception)` в httpx ошибках заменяется на `type(e).__name__` (`ReadTimeout`, `ConnectTimeout`, …).
