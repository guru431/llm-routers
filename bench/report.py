"""Generate markdown report from benchmark + judge results.

Reads:
    bench/models.json
    bench/prompts/tasks.json
    bench/results/<model_id>.jsonl
    bench/results/_judge.jsonl

Writes:
    LLM_MODELS_BENCH_2026-05-15.md  (repo root)
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_JSON = ROOT / "models.json"
TASKS_JSON = ROOT / "prompts" / "tasks.json"
RESULTS = ROOT / "results"
JUDGE_FILE = RESULTS / "_judge.jsonl"
OUT = ROOT.parent / "LLM_MODELS_BENCH_2026-05-15.md"

# Markers wrapping the hand-written TL;DR. On re-run, if OUT already exists we
# splice the existing block between these markers back in, so manual edits survive.
TLDR_BEGIN = "<!-- manual-tldr -->"
TLDR_END = "<!-- /manual-tldr -->"


def _extract_manual_tldr(path: Path) -> list[str] | None:
    """Return the lines between the TL;DR markers (inclusive) from an existing
    report, or None if the file/markers are absent."""
    if not path.exists():
        return None
    old = path.read_text(encoding="utf-8").splitlines()
    try:
        i = old.index(TLDR_BEGIN)
        j = old.index(TLDR_END)
    except ValueError:
        return None
    if j < i:
        return None
    return old[i:j + 1]


def fmt_s(v):
    if v is None:
        return "—"
    return f"{v:.2f}"


def heuristic(task_id: str, text: str) -> str:
    if not text:
        return "—"
    if task_id == "T4_json_extract":
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
            d = json.loads(cleaned)
            keys = {"person", "date", "time", "action"}
            missing = keys - set(d.keys())
            return "✓" if not missing else f"✗ missing {','.join(sorted(missing))}"
        except Exception as e:
            return f"✗ {type(e).__name__}"
    if task_id == "T6_classify":
        clean = text.strip().lower().rstrip(".!").strip().split()
        if not clean:
            return "✗ empty"
        first = clean[0]
        if first in {"question", "complaint", "request", "praise", "spam", "other"}:
            return "✓" + (f" ({len(clean)} words)" if len(clean) > 1 else "")
        return f"✗ '{first}'"
    if task_id == "T8_python_function":
        try:
            cleaned = re.sub(r"^```(?:python|py)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
            compile(cleaned, "<test>", "exec")
            return "✓ compiles" + (" (has parse_duration)" if "def parse_duration" in cleaned else "")
        except SyntaxError as e:
            return f"✗ {e.msg[:30]}"
    if task_id == "T7_bash_oneliner":
        lines = [l for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(lines) == 1:
            return "✓ single line"
        return f"✗ {len(lines)} lines"
    if task_id in ("T2_yt_summary_en", "T3_yt_summary_ru"):
        bullets = re.findall(r"^\s*[-•*]|\d+[\.)]", text, flags=re.MULTILINE)
        n = len(bullets)
        return f"{n} bullets" + (" ✓" if n == 5 else " ✗")
    return ""


def main():
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
    tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))["tasks"]
    task_ids = [t["id"] for t in tasks]

    # Load all results: results[(model_id, task_id)] = record
    results: dict[tuple[str, str], dict] = {}
    for jl in sorted(RESULTS.glob("*.jsonl")):
        if jl.name.startswith("_"):
            continue
        mid = jl.stem
        for line in jl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                results[(mid, r["task_id"])] = r
            except Exception:
                pass

    # Load judge scores
    judges: dict[tuple[str, str], dict] = {}
    if JUDGE_FILE.exists():
        for line in JUDGE_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                judges[(d["model_id"], d["task_id"])] = d
            except Exception:
                pass

    model_by_id = {m["id"]: m for m in models}

    # === Aggregates per model ===
    rows = []
    for m in models:
        mid = m["id"]
        if m.get("skip_reason"):
            rows.append({
                "id": mid, "provider": m["provider"], "model": m["model"],
                "skip": m["skip_reason"], "ok": 0, "n": 0,
            })
            continue
        ttfts, ttfts_r, totals, scores = [], [], [], []
        ok = 0
        empty_text = 0
        errors = []
        for tid in task_ids:
            r = results.get((mid, tid))
            if not r:
                continue
            if r.get("error"):
                errors.append(r["error"][:40])
                continue
            if not (r.get("text") or "").strip():
                empty_text += 1
                continue
            ok += 1
            if r["ttft_s"] is not None:
                ttfts.append(r["ttft_s"])
            if r.get("ttft_reasoning_s") is not None:
                ttfts_r.append(r["ttft_reasoning_s"])
            if r["total_s"] is not None:
                totals.append(r["total_s"])
            j = judges.get((mid, tid))
            if j and j.get("score") is not None:
                scores.append(j["score"])
        rows.append({
            "id": mid,
            "provider": m["provider"],
            "model": m["model"],
            "n": len(task_ids),
            "ok": ok,
            "empty_text": empty_text,
            "ttft_p50": statistics.median(ttfts) if ttfts else None,
            # p90 needs ≥5 samples to be meaningful; below that we return None
            # (renders as '—') instead of falling back to max(), which would
            # be misleadingly labeled p90.
            "ttft_p90": statistics.quantiles(ttfts, n=10)[8] if len(ttfts) >= 5 else None,
            "ttft_r_p50": statistics.median(ttfts_r) if ttfts_r else None,
            "total_p50": statistics.median(totals) if totals else None,
            "total_p90": statistics.quantiles(totals, n=10)[8] if len(totals) >= 5 else None,
            "quality_avg": statistics.mean(scores) if scores else None,
            "quality_n": len(scores),
            # Coverage-penalized quality: a model that answered 2/8 tasks at Q5
            # should NOT outrank a stable 8/8 model at Q4.6. Used for ranking;
            # quality_avg is still shown verbatim (with a '*' when partial).
            "quality_eff": (
                statistics.mean(scores) * len(scores) / len(task_ids)
                if scores and task_ids else None
            ),
            # claude-opus-4-8 is also the judge — its own family's scores are
            # self-assessed and prone to self-preference bias. Flagged with '†'.
            "self_judged": m["provider"] == "claude_agent",
            "errors": errors,
        })

    # === Build markdown ===
    lines = []
    lines.append("# LLM Models Benchmark — 2026-05-15")
    lines.append("")
    lines.append(f"**Запущен с:** локальная Windows-машина")
    lines.append(f"**Моделей:** {len(rows)} ({sum(1 for r in rows if not r.get('skip'))} активных)")
    lines.append(f"**Задач:** {len(tasks)} (RU-edit, YT-summary EN/RU, JSON-extract, RU→EN translate, classify, bash one-liner, Python function)")
    lines.append(f"**Judge:** claude-opus-4-8 (через agent server, температура 0)")
    lines.append("")
    lines.append(
        "> ⚠️ **Self-judge bias:** judge — claude-opus-4-8, и модели семейства "
        "`claude_agent` судят сами себя (помечены `†`). Их Q завышены из-за "
        "self-preference; сравнивать их с другими провайдерами с осторожностью. "
        "`*` у Q = оценка по неполному покрытию задач (quality_n < задач) — "
        "ранжирование использует покрытие-взвешенный Q, не сырой средний."
    )
    lines.append("")

    # === TL;DR ===
    active_with_q = [r for r in rows if not r.get("skip") and r.get("quality_avg") is not None]
    by_quality = sorted(active_with_q, key=lambda r: -r["quality_avg"])
    by_ttft = sorted(active_with_q, key=lambda r: r["ttft_p50"] if r["ttft_p50"] is not None else 9999)

    # === Manual TL;DR ===
    # Hand-written block, preserved across re-runs via the marker pair below
    # (see read-old-file logic at the end of main()). The default text is a
    # SNAPSHOT from 2026-05-15 — numbers may lag the auto-generated tables below.
    lines.append(TLDR_BEGIN)
    lines.append("## TL;DR")
    lines.append("")
    lines.append("_Снимок 2026-05-15 — таблицы ниже могут быть свежее этого ручного блока._")
    lines.append("")
    lines.append("**Победители по use-case:**")
    lines.append("")
    lines.append("- **Голос/чат (минимум TTFT при разумном качестве):** `groq-llama-3.3-70b` (TTFT 1.0s, Total 1.3s, Q4.12), `or-qwen3-235b` (1.5s/3.1s/Q4.62), `or-qwen3-vl-30b` (1.7s/2.6s/Q4.50)")
    lines.append("- **Максимальное качество (Q=5.0):** `claude-opus-4-7`, `ollama-gpt-oss-20b`, `ocg-glm-5/5.1`, `ocg-kimi-k2.5/k2.6`, `ocg-qwen3.5-plus`/`3.6-plus` — но все медленные (TTFT 7-40s)")
    lines.append("- **Локальный desktop-проект (текущая прод-цель):** `or-qwen3-235b` лучший общий выбор (TTFT 1.5s/Total 3.1s/Q4.62). Внутри OpenCode Go подписки — `ocg-mimo-v2.5-pro` (3.6s/4.1s/Q4.75) или текущий `ocg-mimo-v2.5` (3.2s/4.0s/Q4.25)")
    lines.append("- **Ночные cron-скрипты (качество > скорость):** `claude-opus-4-7` через agent server (бесплатно по Max-подписке, Q5.0) или продолжать `ocg-minimax-m2.7` (Q4.5)")
    lines.append("")
    lines.append("**Главные сюрпризы:**")
    lines.append("")
    lines.append("- **Groq Llama-3.3-70B** — самый быстрый ответ в бенче (TTFT 1.0s, Total 1.3s) и стабильно Q4+. Free-tier 60 RPM ограничение")
    lines.append("- **OpenCode Go mimo-серия (v2-pro/v2.5-pro/v2.5)** — лучший trade-off в OpenCode Go подписке: TTFT 3.2-3.6s, Q4.25-4.75. Превосходит текущий выбор для локального desktop-проекта")
    lines.append("- **Reasoning-модели (kimi-k2.6, deepseek-v4-flash, glm-5/5.1, ollama qwen3.5:9b)** тратят 30-90% бюджета токенов на thinking → высокая latency, иногда пустой `content` если max_tokens исчерпан")
    lines.append("- **MiniMax direct через OpenAI-compat endpoint не возвращает SSE-стрим** — все 4 модели имеют TTFT=Total. Реальный TTFT неизвестен (нужен их native endpoint)")
    lines.append("- **OpenRouter блокирует** все Google + Anthropic модели с 403 \"violation of provider ToS\" — нужно включить privacy/data opt-in на их dashboard")
    lines.append("- **Hy3-preview и Kimi-k2.5/k2.6 нестабильны** на длинных input/output — 1-3/8 пустых ответов даже при HTTP 200")
    lines.append(TLDR_END)
    lines.append("")
    lines.append("## Методика")
    lines.append("")
    lines.append("- **TTFT** — время от send до первого `delta.content` chunk'а в SSE-стриме (для серверов без стрима — TTFT=Total)")
    lines.append("- **TTFT-R** — время до первого `delta.reasoning_content` или `message.thinking` (только у reasoning/thinking-моделей)")
    lines.append("- **Total** — wall-clock полного ответа")
    lines.append("- **Quality (0-5)** — LLM-as-judge по рубрикам категории (rubric на каждую категорию см. `bench/judge.py::RUBRIC`)")
    lines.append("- **OK** — задач с непустым финальным `text` (reasoning-only ответы считаются empty)")
    lines.append("- Параметры запросов: `temperature=0.2`, `max_tokens=2048` (Ollama: `num_predict=4096`)")
    lines.append("- Запуск последовательный (не параллельный — чтобы не искажать TTFT rate-limit'ами)")
    lines.append("- Источники: `bench/run.py` (раннер), `bench/judge.py` (judge), `bench/results/*.jsonl` (сырые данные)")
    lines.append("")
    lines.append("## Сводная таблица (медианы по 8 задачам)")
    lines.append("")
    lines.append("Отсортировано по quality desc, при равенстве — по TTFT asc.")
    lines.append("")
    lines.append("Колонки: **TTFT** = первый токен ответа; **TTFT-R** = первый reasoning-токен (только у thinking-моделей); **Total** = полное время до конца ответа; **Q** = LLM-as-judge 0-5; **OK** = задач с непустым ответом.")
    lines.append("")
    lines.append("| Модель | Provider | TTFT p50 | TTFT p90 | TTFT-R p50 | Total p50 | Total p90 | Q | OK |")
    lines.append("|--------|----------|----------|----------|------------|-----------|-----------|---|-----|")

    active = [r for r in rows if not r.get("skip")]
    # Rank by coverage-penalized quality so a model with 2/8 answers can't top
    # the table on two lucky high scores; tie-break by TTFT.
    active.sort(key=lambda r: (
        -(r["quality_eff"] if r["quality_eff"] is not None else -1),
        r["ttft_p50"] if r["ttft_p50"] is not None else 9999,
    ))
    for r in active:
        q = f"{r['quality_avg']:.2f}" if r["quality_avg"] is not None else "—"
        if r["quality_avg"] is not None and r["quality_n"] < r["n"]:
            q += "*"  # partial coverage
        if r.get("self_judged"):
            q += "†"  # self-judged family
        ok_str = f"{r['ok']}/{r['n']}"
        if r.get("empty_text"):
            ok_str += f" (+{r['empty_text']} empty)"
        lines.append(
            f"| `{r['id']}` | {r['provider']} | "
            f"{fmt_s(r['ttft_p50'])}s | {fmt_s(r['ttft_p90'])}s | "
            f"{fmt_s(r.get('ttft_r_p50'))}s | "
            f"{fmt_s(r['total_p50'])}s | {fmt_s(r['total_p90'])}s | "
            f"{q} | {ok_str} |"
        )
    lines.append("")

    skipped = [r for r in rows if r.get("skip")]
    if skipped:
        lines.append("## Не тестировались (auth / quota / balance)")
        lines.append("")
        for r in skipped:
            lines.append(f"- `{r['id']}` — {r['skip']}")
        lines.append("")

    # === Per-task tables (TTFT/total per task per model) ===
    lines.append("## Латенси по задачам")
    lines.append("")
    for t in tasks:
        tid = t["id"]
        lines.append(f"### {tid} ({t['category']})")
        lines.append("")
        lines.append("| Модель | TTFT | TTFT-R | Total | Q | Эвристика |")
        lines.append("|--------|------|--------|-------|---|-----------|")
        task_rows = []
        for m in models:
            mid = m["id"]
            r = results.get((mid, tid))
            if not r:
                continue
            j = judges.get((mid, tid))
            ttft = r["ttft_s"]
            ttft_r = r.get("ttft_reasoning_s")
            total = r["total_s"]
            err = r.get("error")
            text = r.get("text") or ""
            h = heuristic(tid, text) if not err else ""
            score = j.get("score") if j else None
            if err:
                display = "✗ " + err[:30]
            elif not text.strip():
                display = "✗ empty (reasoning_only)"
            else:
                display = None
            task_rows.append({
                "mid": mid, "ttft": ttft, "ttft_r": ttft_r, "total": total,
                "score": score, "heur": h, "display_err": display,
            })
        # sort by quality desc then ttft asc
        task_rows.sort(key=lambda x: (
            -(x["score"] if x["score"] is not None else -1),
            x["ttft"] if x["ttft"] is not None else 9999,
        ))
        for x in task_rows:
            if x["display_err"]:
                lines.append(f"| `{x['mid']}` | — | — | — | — | {x['display_err']} |")
            else:
                q = str(x["score"]) if x["score"] is not None else "—"
                lines.append(
                    f"| `{x['mid']}` | {fmt_s(x['ttft'])}s | {fmt_s(x['ttft_r'])}s | "
                    f"{fmt_s(x['total'])}s | {q} | {x['heur']} |"
                )
        lines.append("")

    # === Top recommendations ===
    lines.append("## Рекомендации")
    lines.append("")
    if active:
        by_ttft = sorted([r for r in active if r["ttft_p50"] is not None], key=lambda r: r["ttft_p50"])[:5]
        by_quality = sorted(
            [r for r in active if r["quality_eff"] is not None],
            key=lambda r: -r["quality_eff"],
        )[:5]
        balanced = sorted(
            [r for r in active if r["quality_avg"] is not None and r["ttft_p50"] is not None],
            key=lambda r: (r["ttft_p50"] / max(0.5, r["quality_avg"])),
        )[:5]
        lines.append("**Топ-5 по TTFT (реактивность для голос/чат):**")
        for r in by_ttft:
            q = f"{r['quality_avg']:.2f}" if r["quality_avg"] is not None else "—"
            lines.append(f"- `{r['id']}` — TTFT p50 {r['ttft_p50']:.2f}s, quality {q}")
        lines.append("")
        lines.append("**Топ-5 по качеству (покрытие-взвешенному):**")
        for r in by_quality:
            mark = ("*" if r["quality_n"] < r["n"] else "") + ("†" if r.get("self_judged") else "")
            lines.append(
                f"- `{r['id']}` — quality {r['quality_avg']:.2f}{mark} "
                f"(eff {r['quality_eff']:.2f}), TTFT p50 {fmt_s(r['ttft_p50'])}s"
            )
        lines.append("")
        lines.append("**Топ-5 по balance (TTFT/quality):**")
        for r in balanced:
            lines.append(f"- `{r['id']}` — TTFT/Q = {r['ttft_p50']/r['quality_avg']:.2f}, TTFT {r['ttft_p50']:.2f}s, Q {r['quality_avg']:.2f}")
    lines.append("")

    # === Errors block ===
    err_rows = [r for r in active if r["errors"]]
    if err_rows:
        lines.append("## Ошибки на отдельных моделях")
        lines.append("")
        for r in err_rows:
            lines.append(f"- `{r['id']}` ({len(r['errors'])} fail): {'; '.join(r['errors'][:3])}")
        lines.append("")

    # Preserve a hand-edited TL;DR from a previous run: replace the freshly
    # generated default block with the existing one between the markers.
    preserved = _extract_manual_tldr(OUT)
    if preserved is not None:
        try:
            i = lines.index(TLDR_BEGIN)
            j = lines.index(TLDR_END)
            lines[i:j + 1] = preserved
        except ValueError:
            pass

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {OUT} ({len(lines)} lines, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
