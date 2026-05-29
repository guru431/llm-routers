# Findings — llm_routers
Побочные находки. Ревизия: MonthlyStratReview 1-го числа. Stale >90 дней → alert.

## 2026-05-29 · dialogue_continue panel теряет исходные флаги [P3]
**Контекст:** Code review Opus 4.8 (CODE_REVIEW_opus48_2026-05-29.md), mcp-council/server.py:1160-1180
**Что:** При continue для panel хардкодит threshold=7 и всегда гоняет devils_advocate+diversity_monitor; DialogueState не персистит diversity_threshold/devils_advocate_rotation/diversity_monitor. Сессия с custom-настройками молча меняет поведение при continue.
**Предложение:** Хранить эти 3 параметра в DialogueState и читать в continue-раннере.
**Статус:** open

## 2026-05-29 · CCR template default 0.0.0.0 [P3]
**Контекст:** Code review Opus 4.8 (CODE_REVIEW_opus48_2026-05-29.md), claude-code-router/config.example.json:4
**Что:** Дефолт HOST=0.0.0.0 (LAN-exposed) + APIKEY-плейсхолдер; копия без заданного APIKEY = unauthenticated upstream-keyed proxy в LAN.
**Предложение:** Дефолт 127.0.0.1, opt-in на 0.0.0.0 (как loopback-default в claude-agent-server).
**Статус:** open

## 2026-05-29 · token-estimate занижает для ru/CJK [P3]
**Контекст:** Code review Opus 4.8 (CODE_REVIEW_opus48_2026-05-29.md), claude-agent-server/server.py:352-354
**Что:** Usage считается как len(text)//4, для русского/CJK занижение в 2-3×. Помечено estimate:true.
**Предложение:** Оставить как есть или уточнить в поле; действие не требуется если потребитель не считает по ним стоимость.
**Статус:** open
