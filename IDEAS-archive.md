# Ideas archive — llm_routers

Audit trail закрытых идей (аналог FINDINGS-archive). Записи не удаляются. Новые — сверху.
Все 23 идеи из батча 2026-07-13 реализованы 2026-07-17 по явному решению владельца
(«реализовать все 21/23 в полный рост») — осознанный override прежних wontfix-решений
из FINDINGS-archive для enterprise-надстроек.

---

## 2026-07-13 · Изолированные capability profiles [P1]
**Idea:** Профили `chat`/`research`/`agent` с разными полномочиями (fs/tools/session/token/workspace).
**Status:** done
**Resolved:** 2026-07-17 — введены именованные профили в обоих agent-серверах: codex `resolve_profile_and_sandbox` (chat/research→read-only, agent→workspace-write+отдельный token+workdir containment, agent+tools→400); claude отклоняет `agent` (400), chat/research документированы как chat+web-emulation. `usage.profile` + `/health` `default_profile`/`profiles`. Тесты в test_codex_server.py/test_handler.py.

## 2026-07-13 · Adaptive provider-aware council [P1]
**Idea:** Выбор участников по health/latency/failure domain; start-small → escalate при disagreement.
**Status:** done
**Resolved:** 2026-07-17 — `mcp-council/adaptive.py` (filter_healthy / pick_starting_subset (≥2 домена) / should_escalate) + `council.run_adaptive_council` (pre-flight healthcheck → diverse старт-субсет → эскалация состава при low quorum/agreement). Флаг `adaptive=True` в `council_ask`. Тесты в test_council_ideas.py.

## 2026-07-13 · Evidence-aware verdict вместо auto-adopt [P1]
**Idea:** Разделить agreement / evidence / test / source-quality / human-review; запрет auto-adopt для high-risk.
**Status:** done
**Resolved:** 2026-07-17 — `council._build_summary` расширен: `verdict.{agreement,evidence,evidence_sources,executable_test,source_quality,risk_class,human_review_required}`, `risk_class` (эвристика high-risk), `human_review_required`. High-risk тема никогда не даёт plain-"adopt". Тесты в test_council_ideas.py.

## 2026-07-13 · Единый budget/deadline manager [P1]
**Idea:** Общий scheduler лимитов wall-time/calls/tokens/searches/$ + dry-run estimate.
**Status:** done
**Resolved:** 2026-07-17 — `mcp-council/budget.py` (`RunBudget` + `estimate_run`); wall-time deadline проверяется на границе раундов (graceful, ответы сохраняются), потолки web-search/cost/calls. Параметры `deadline_seconds`/`max_cost_usd`/`max_web_searches` в `council_ask`; tool `council_estimate` (dry-run). Тесты в test_council_ideas.py.

## 2026-07-13 · Claim-to-source ledger и DLP [P1]
**Idea:** Хранить claims↔sources+provenance; фильтровать outbound search-запросы на secrets.
**Status:** done
**Resolved:** 2026-07-17 — `mcp-council/dlp.py`: `scrub_outbound_query` (блок секрет/credential/локальный путь ДО Exa, wired в web_search_tool) + `build_claim_ledger` (query→sources из tool_calls_log; web_search_tool теперь пишет `sources` URL). `claim_ledger` в результате council. Тесты в test_council_ideas.py.

## 2026-07-13 · Калиброванная peer-ranking модель [P2]
**Idea:** pairwise/Borda/BT aggregation + schema-enforced JSON + repair-retry + reliability weights.
**Status:** done
**Resolved:** 2026-07-17 — `council._aggregate_borda` (scale-free rank-based cross-check рядом с mean, ties=avg); stage-2 repair-retry с `response_format={"type":"json_object"}` (`_normalize_stage2` + `build_stage2_repair_user`); `summary.ranking_methods_agree` (mean vs Borda) блокирует adopt при расхождении. Тесты в test_council_ideas.py.

## 2026-07-13 · Append-only event journal [P2]
**Idea:** Журнал, из которого строятся snapshots; versioned dumps; terminal reserved; dialogue тоже.
**Status:** done
**Resolved:** 2026-07-17 — `schema_version` в council-snapshot (state.py) и dialogue-dump (engine.py, versioned recovery); dialogue получил append-only event journal (engine.emit_event + state.event_writer, wired в server: per-round/diversity/terminal `result_ready` события в logs/events/<session>.jsonl). Council journal уже был (event_log.py). Тесты в test_dialogue_ideas.py.

## 2026-07-13 · Dialogue resume, extend и branching [P2]
**Idea:** Разделить resume/extend/fork; ветвление transcript + сравнение веток.
**Status:** done
**Resolved:** 2026-07-17 — добавлен `dialogue_fork(session_id, directive, rounds)`: deep-copy транскрипта в новую session, продолжение на КОПИИ, оригинал нетронут (branch + сравнение через dialogue_result обеих). Общая runner-фабрика `_build_resume_runner` для continue/fork. Тесты в test_dialogue_ideas.py.

## 2026-07-13 · Token-aware rolling memory [P2]
**Idea:** Per-model token budget, verbatim recent + rolling summary старого; replace failed participant.
**Status:** done
**Resolved:** 2026-07-17 — `dialogue/prompts.py`: `estimate_tokens` + `HISTORY_TOKEN_BUDGET`; `format_history_section(max_history_tokens=...)` дропает старые раунды за пределами verbatim-окна с rolling-summary маркером (recent всегда сохраняются). Высокий дефолт — короткие диалоги не затронуты. Тесты в test_dialogue_ideas.py.

## 2026-07-13 · Semantic diversity monitor 2.0 [P2]
**Idea:** pairwise stance/uncertainty/новые claims до-после reprompt; monitor-failure отдельно от score.
**Status:** done
**Resolved:** 2026-07-17 — `panel.run_diversity_check_v2` (structured: status ok|failed, score, agreers, uncertainty) отличает monitor-сбой от genuine score=0; monitor-failure → warning; post-reprompt re-measure (delta). Новое поле `DialogueState.diversity_monitor_status` (персистится). Back-compat `run_diversity_check`. Тесты в test_dialogue_ideas.py.

## 2026-07-13 · Настоящий streaming и cancellation [P2]
**Idea:** Claude stream-json / Codex JSON → OpenAI SSE deltas; real usage/stop_reason; cancel при disconnect.
**Status:** done
**Resolved:** 2026-07-17 — оба agent-сервера: `run_*_stream` читают stdout построчно (`claude --output-format stream-json --verbose`, `codex exec --json`), `_send_stream_live` отдаёт реальные SSE-deltas, real usage/stop_reason (`estimate:false`) при наличии; client-disconnect убивает дерево процессов (write-fail → gen.close → _kill_process_tree); `StreamUnsupported` → fallback на буфер. Тесты в test_codex_server.py/test_handler.py.

## 2026-07-13 · Native structured output [P2]
**Idea:** response_format/json_schema → prompt+validate+repair; полный JSON Schema для tool emulation.
**Status:** done
**Resolved:** 2026-07-17 — оба сервера: `response_format` (json_object/json_schema) → инъекция схемы + валидация + один repair-retry + `usage.structured_output`; `build_tools_system_prompt` эмитит полный JSON Schema (не только flat); required-arg валидация tool-call + repair-retry. Byte-identical helpers. Тесты добавлены.

## 2026-07-13 · Readiness, metrics и bounded queue [P2]
**Idea:** /ready + /metrics probes; overload → bounded queue + Retry-After.
**Status:** done
**Resolved:** 2026-07-17 — оба сервера: `GET /ready` (auth/bin/root probes, 503 с деталями), `GET /metrics` (thread-safe счётчики + latency ring buffer median/p90), bounded queue (`_acquire_slot` с QUEUE_WAIT/MAX_QUEUE) + `429 + Retry-After`. (Прежде /ready был помечён wontfix — реализован по решению владельца.) Тесты добавлены.

## 2026-07-13 · Безопасный cache с provenance [P2]
**Idea:** Opt-in response cache, per-entry limit, fingerprint, singleflight, ETag/metadata, privacy mode.
**Status:** done
**Resolved:** 2026-07-17 — `mcp-council/response_cache.py`: opt-in LRU+TTL кэш council-ответов, fingerprint(question+members+catalog+params+context), per-entry byte cap, singleflight (коллапс конкурентных miss), provenance (cached_at/age/fingerprint footer), privacy (in-memory, не на диск). Флаг `cache=True` в `council_ask`. Тесты в test_council_ideas.py.

## 2026-07-13 · Immutable benchmark manifests [P2]
**Idea:** UUID/UTC/git SHA/prompt+catalog hashes/params; immutable run directory.
**Status:** done
**Resolved:** 2026-07-17 — `bench/run.py`: manifest.json (run_id=uuid4, started_at UTC-Z, git_sha, sha256 models/tasks, cli_args, params) в immutable `results/runs/<run_id>/`; `results/runs/latest.txt`; каждая запись с run_id/repeat_idx/cold. `bench/_store.py` для навигации. report.py читает latest run (fallback на flat).

## 2026-07-13 · Статистический benchmark protocol [P2]
**Idea:** warm-up/repeats/interleaving/persistent conns/cold-warm/bootstrap CI/regression thresholds.
**Status:** done
**Resolved:** 2026-07-17 — `bench/run.py`: `--repeats`/`--warmup`/`--seed`, randomized interleaving (seeded shuffle), один persistent httpx.Client, cold vs warm (repeat_idx). `bench/report.py`: `bootstrap_ci` (stdlib, 1000 resamples, гейт repeats≥2/≥5 samples), `--baseline` + REGRESSION-пороги (latency +25% / quality −0.5).

## 2026-07-13 · Deterministic evals + blinded multi-judge [P2]
**Idea:** Детерминированный gate (schema/tests/shell) до LLM; 2-3 blinded judges + adjudication + hash-bound.
**Status:** done
**Resolved:** 2026-07-17 — `bench/judge.py`: `deterministic_score` (T4 schema/T6 classify/T2-T3 bullets/T7 shell/T8 compile+`--exec-code`) как первый gate (`judge_method`/`deterministic_pass`); `JUDGE_MODELS` (`--judges`) с blinded independent + `adjudicate` (медиана, `judge_disagreement` при spread≥3), per-judge hash-bound `judge_scores`. Single-judge byte-identical прежнему.

## 2026-07-13 · Benchmark diff/dashboard [P3]
**Idea:** Сравнение двух manifests (quality/latency/failure/cost + confidence); Markdown/CSV/HTML; авто-regressions.
**Status:** done
**Resolved:** 2026-07-17 — `bench/report_diff.py`: `python report_diff.py <baseline> <current>` — дельты quality/latency/failure/tok_out + bootstrap CI + regression-флаги (worst-first); экспорт Markdown (default)/`--csv`/`--html` (self-contained, без CDN).

## 2026-07-13 · Генерируемый API/capabilities reference [P2]
**Idea:** Генерировать docs + /v1/models metadata из реальных signatures/constants.
**Status:** done
**Resolved:** 2026-07-17 — `mcp-council/capabilities.py`: `build_capabilities()` (models/roles/prices/presets/limits/tools/verdict-axes из CATALOG/PRESETS/констант) + tool `council_capabilities`; `model_ask_models_line()` из CATALOG + unit-тест, что docstring `model_ask` содержит все enabled id (защита от дрейфа). agent-серверы: docs из signatures. Тесты в test_council_ideas.py.

## 2026-07-13 · CI для safety и docs contracts [P2]
**Idea:** CI jobs: quick tests, secret scan, markdown link check, hook fixtures, API conformance.
**Status:** done
**Resolved:** 2026-07-17 — `.github/workflows/ci.yml`: jobs tests (run_tests.py --quick), compileall (все пакеты), secret-scan (gitleaks), markdown-links (`scripts/check_markdown_links.py`), hook-fixtures (`scripts/test_pre_commit_hook.sh` — проверяет блок секрета/sensitive-имени + pass clean). Оба скрипта проверены локально.

## 2026-07-13 · CCR strict catalog и traceable fallback [P2]
**Idea:** Allowlist из общего каталога; unknown→ошибка; health/cost/latency fallback + trace header.
**Status:** done
**Resolved:** 2026-07-17 — `claude-code-router/custom_router.js`: structured `{"ccr_route":{provider,model,reason}}` trace на каждом решении (reasons: opencode_model_match/router_default_fallback/no_model_builtin_routing/unknown_model_rejected); пустой config.models → loud WARNING перед fallback. README: three-copy sync каталога, reason-таблица, single-provider-by-design. (Multi-provider fallback не выдумывался — один провайдер по дизайну.)

## 2026-07-13 · Privacy/retention controls [P2]
**Idea:** TTL/size quotas, redaction, encryption-at-rest, purge API для logs/transcripts/events/cache/bench.
**Status:** done
**Resolved:** 2026-07-17 — `mcp-council/retention.py`: TTL-purge (COUNCIL_LOG_RETENTION_HOURS, дефолт 168h) + size-quota (oldest-first) для logs/{jobs,dialogues,events,dumps} + `redact()` (маскирует credential-форматы). Tool `council_purge_logs`. Тесты в test_retention.py. (Encryption-at-rest не делался — логи на локальном диске одного пользователя; TTL+redaction закрывают основной риск.)

## 2026-07-13 · Private lifecycle для findings и ADR [P3]
**Idea:** Private tracked backlog со status manifest и stale >90d reminder.
**Status:** done
**Resolved:** 2026-07-17 — `scripts/backlog_manifest.py` (сканирует FINDINGS/IDEAS + archives, пишет gitignored `backlog/manifest.json`, флагит open-записи >90d, `--fail-on-stale` exit 2 для cron); `backlog/README.md` (tracked, документирует lifecycle без внутренних данных); `.gitignore` для manifest.json. Проверено локально (65 записей, 0 stale).
