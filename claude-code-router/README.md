# claude-code-router

Кастомизация npm-пакета [`@musistudio/claude-code-router`](https://github.com/musistudio/claude-code-router) (`ccr`) — HTTP-прокси, который принимает Anthropic API-запросы от Claude Code и роутит их в произвольных LLM-провайдеров (OpenAI-совместимый upstream).

В отличие от соседних пакетов `mcp-*` это **не MCP-сервер**, а отдельный сервис, слушающий по умолчанию `127.0.0.1:3456` (loopback; для LAN-доступа явно выставить `HOST: 0.0.0.0` в конфиге).

## Состав

| Файл | Назначение |
|---|---|
| [`config.example.json`](config.example.json) | Template для ccr-конфига с плейсхолдерами `<…_HERE>`. Реальный `config.json` с ключами хранится **вне репо** в `~/.claude-code-router/config.json` (см. ниже). |
| [`custom_router.js`](custom_router.js) | Кастомная JS-логика: per-project выбор модели через `ANTHROPIC_MODEL` env (известный upstream-id → `opencode,<model>`; `claude-*` имя → fallback на `Router.default`) |

## Установка

Сам пакет ставится глобально через npm — он не лежит в этом репозитории:

```bash
npm i -g @musistudio/claude-code-router
# проверить установку (не зависит от версии CLI-флагов ccr):
npm ls -g @musistudio/claude-code-router
```

## Привязка файлов к ccr

ccr хардкодит чтение конфига из `$HOME/.claude-code-router/` (нет env-переопределения).

**`config.json` живёт ТОЛЬКО в `~/.claude-code-router/`** (реальный файл, не симлинк) — он содержит API-ключи и в репозиторий не попадает (`.gitignore`). В репо лежит [`config.example.json`](config.example.json) как template — копировать его и подставлять реальные ключи.

**`custom_router.js`** не содержит секретов, поэтому его удобно подключить симлинком из репо для версионирования:

```
~/.claude-code-router/config.json        (реальный файл, ключи внутри, вне git)
~/.claude-code-router/custom_router.js  → <repo>/claude-code-router/custom_router.js
```

Первичная установка (PowerShell):

```powershell
# 1. Положить реальный config из template:
Copy-Item .\config.example.json "$env:USERPROFILE\.claude-code-router\config.json"
# затем отредактировать config.json — подставить:
#   APIKEY  ← ваш входной токен для защиты прокси
#   api_key ← ключ upstream-провайдера

# 2. Симлинк custom_router.js (не содержит секретов):
New-Item -ItemType SymbolicLink `
  -Path "$env:USERPROFILE\.claude-code-router\custom_router.js" `
  -Target "$PWD\custom_router.js"
```

Остальные runtime-файлы (`logs/`, `.claude-code-router.pid`, `plugins/`, `presets/`) остаются в `~/.claude-code-router/` и в репо не попадают.

Проверка:

```powershell
Get-Item "$env:USERPROFILE\.claude-code-router\config.json" | Select-Object Name, LinkType, Length
Get-Item "$env:USERPROFILE\.claude-code-router\custom_router.js" | Select-Object Name, LinkType, Target
```

## Запуск

```bash
ccr start    # стартует HTTP-прокси на 127.0.0.1:3456 (HOST из config.json)
ccr stop
ccr status
```

Health-check:

```bash
curl -H "Authorization: Bearer $CCR_API_KEY" http://127.0.0.1:3456/
# → {"message":"LLMs API","version":"..."}
```

## Как клиенты используют ccr

Клиент (Claude Code или любой Anthropic SDK) ставит:
- `ANTHROPIC_BASE_URL=http://127.0.0.1:3456` (локально) или `http://<host>:3456` (из LAN)
- `ANTHROPIC_AUTH_TOKEN=$CCR_API_KEY`
- опционально `ANTHROPIC_MODEL=<upstream-model>` (любая модель upstream-провайдера)

ccr:
1. Принимает Anthropic-формат
2. `custom_router.js` смотрит `req.body.model`:
   - если это известный upstream-id → роут в `opencode,<model>`
   - если `claude-*` → fallback на `Router.default`
3. Конвертирует в OpenAI-формат и отправляет в upstream
4. Конвертирует ответ обратно в Anthropic-формат

## Каталог моделей и трассировка

`custom_router.js` решает роут по `req.body.model`. Источник допустимых upstream-id — **`config.Providers[opencode].models`** из живого `~/.claude-code-router/config.json`; хардкод `OPENCODE_MODELS_FALLBACK` в `custom_router.js` — только safety-net.

**Синхронизация каталога (три копии, нет build-step):**
1. `mcp-council/models.py::CATALOG` — канон OCG-моделей для всего репо;
2. `config.example.json` → `Providers[opencode].models` — ручное зеркало (template);
3. `OPENCODE_MODELS_FALLBACK` в `custom_router.js` — ручное зеркало (fallback).

При добавлении/удалении модели править все три. Реальный роут берёт список из вашего `config.json` (копия template), поэтому обычно достаточно обновить `config.json` + `ccr restart` — fallback используется только если список в конфиге **пуст/отсутствует** (сломанный конфиг), и тогда роутер печатает `WARNING: config.Providers[opencode].models is empty or missing` (fail-closed видимость дрейфа, а не молчаливый хардкод).

**Трассировка решений.** CCR-контракт кастомного роутера даёт только `(req, config)` и ждёт назад строку `"provider,model"` (или `null`) — объекта ответа, куда можно повесить trace-заголовок, нет. Поэтому на **каждое** решение роутер печатает одну структурную строку в stdout:

```json
{"ccr_route":{"provider":"opencode","model":"glm-5.2","reason":"opencode_model_match"}}
```

`reason` — почему выбран роут:

| reason | когда |
|---|---|
| `opencode_model_match` | `model` найден в каталоге opencode → `opencode,<model>` |
| `router_default_fallback` | `claude-*` имя → fallback на `Router.default` (единственный fallback: провайдер один) |
| `no_model_builtin_routing` | `model` отсутствует → встроенный scenario-routing CCR (`null`) |
| `unknown_model_rejected` | неизвестный id → `throw` (не молча в Router.default) |

Grep по логам CCR (`grep ccr_route`) восстанавливает путь каждого запроса. **Multi-provider health/cost/latency fallback отсутствует by design** — апстрим один (`opencode`); единственный fallback — `Router.default` для `claude-*`/missing-model.

## Ключи

`config.json` хранится **только в `~/.claude-code-router/`** (вне git, не симлинк) и содержит:

- **APIKEY** — входной токен для защиты прокси (вы задаёте сами)
- **api_key upstream-провайдера** — ключ к выбранному OpenAI-совместимому API

При ротации: обновить значение в `~/.claude-code-router/config.json` → `ccr restart`.

Файл `config.json` в репо не попадает (`.gitignore` → `claude-code-router/config.json`); версионируется только template [`config.example.json`](config.example.json).

## Coexistence

См. [../AGENTS.md](../AGENTS.md) для общих правил, [../CLAUDE.md](../CLAUDE.md) для Claude-specific.
