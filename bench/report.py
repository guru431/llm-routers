"""Generate markdown report from benchmark + judge results.

Reads:
    bench/models.json
    bench/prompts/tasks.json
    bench/results/<model_id>.jsonl
    bench/results/_judge.jsonl

Writes:
    the markdown report to `--out`, else $BENCH_REPORT_OUT, else
    LLM_MODELS_BENCH_2026-05-15.md in the repo root (gitignored — the living
    document with hand-written addenda is kept outside this public repo).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import _store

ROOT = Path(__file__).resolve().parent
MODELS_JSON = ROOT / "models.json"
TASKS_JSON = ROOT / "prompts" / "tasks.json"
RESULTS = ROOT / "results"
JUDGE_FILE = RESULTS / "_judge.jsonl"
# Where the report is written. The LIVING document (with hand-written addenda)
# lives outside this repo, so the destination is configurable: `--out` wins, then
# $BENCH_REPORT_OUT, then a copy inside the repo. The in-repo default is
# gitignored on purpose — regenerating a report must not push a second, diverging
# copy into a PUBLIC repository.
OUT = Path(os.environ.get("BENCH_REPORT_OUT") or (ROOT.parent / "LLM_MODELS_BENCH_2026-05-15.md"))

# Judge panel assumed when the run's manifest carries none (legacy runs judged
# before judge.py started recording `judge_panel`).
DEFAULT_JUDGE_PANEL = ["claude-opus-4-8"]

# Regression thresholds vs a baseline run (ideas 16 & 18).
LATENCY_REGRESSION_PCT = 25      # median latency grew by more than this % → flag
QUALITY_REGRESSION_DELTA = 0.5   # quality dropped by more than this (0-5) → flag
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0

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


def fmt_s_unit(v):
    """Like fmt_s but appends the 's' unit only for real values, so a missing
    metric (e.g. p90 with <5 samples) renders as '—' rather than '—s'."""
    if v is None:
        return "—"
    return f"{v:.2f}s"


def bootstrap_ci(samples: list[float], repeats: int,
                 lo: float = 2.5, hi: float = 97.5,
                 seed: int = BOOTSTRAP_SEED,
                 resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the MEDIAN of `samples` (pure stdlib).

    Returns (lo, hi) or None when there isn't enough signal. Per idea 16 a CI is
    only produced when the protocol was repeated (repeats>=2) or there are enough
    samples (>=5); with fewer than 2 points there's nothing to resample.
    """
    n = len(samples)
    if n < 2:
        return None
    if not (repeats >= 2 or n >= 5):
        return None
    rng = random.Random(seed)
    meds: list[float] = []
    for _ in range(resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(resample))
    meds.sort()

    def pct(p: float) -> float:
        idx = int(round(p / 100 * (len(meds) - 1)))
        return meds[max(0, min(len(meds) - 1, idx))]

    return (pct(lo), pct(hi))


def bootstrap_ci_task_mean(groups: list[list[float]], repeats: int,
                           lo: float = 2.5, hi: float = 97.5,
                           seed: int = BOOTSTRAP_SEED,
                           resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float] | None:
    """Hierarchical percentile bootstrap CI for the MEAN OF TASK MEANS.

    `groups` is one inner list of judge scores per TASK. Each resample redraws
    tasks with replacement and, inside every drawn task, redraws its repeats —
    mirroring exactly how `quality_avg` is computed (average within a task, then
    across tasks).

    This replaced a flat `bootstrap_ci(all_scores)`, which reported the CI of the
    MEDIAN of one pooled sample: a different estimator than the mean shown next
    to it, and one dominated by between-TASK variance — i.e. precisely the
    "spread across tasks" the per-repeat judging was meant to stop reporting.
    """
    n = len(groups)
    if n < 2:
        return None
    total = sum(len(g) for g in groups)
    if not (repeats >= 2 or total >= 5):
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        acc = 0.0
        for _ in range(n):
            g = groups[rng.randrange(n)]
            k = len(g)
            acc += sum(g[rng.randrange(k)] for _ in range(k)) / k
        means.append(acc / n)
    means.sort()

    def pct(p: float) -> float:
        idx = int(round(p / 100 * (len(means) - 1)))
        return means[max(0, min(len(means) - 1, idx))]

    return (pct(lo), pct(hi))


def task_score_groups(judges: dict, model_id: str, task_ids: list[str]) -> list[list[float]]:
    """Per-task judge scores for one model: one inner list per task with scores.

    The single place both the current run and the `--baseline` run go through, so
    the two quality numbers compared by the regression gate are the SAME
    statistic. The baseline used to be a flat mean over every score (weighting a
    task by how many repeats it happened to complete) while the current run
    averaged per task first — so the ⚠️REGRESSION flag could fire purely because
    the repeat count changed."""
    groups = []
    for tid in task_ids:
        j = judges.get((model_id, tid))
        if j and j.get("scores"):
            groups.append(list(j["scores"]))
    return groups


def quality_from_groups(groups: list[list[float]]) -> float | None:
    """mean(task means) — the Q shown in the table."""
    if not groups:
        return None
    return statistics.mean([statistics.mean(g) for g in groups])


def self_judged(model: dict, panel: list[str]) -> bool:
    """True when a benched model is (or is family of) one of the ACTUAL judges.

    Derived from the panel recorded in the manifest instead of a hardcoded
    `provider == "claude_agent"`: with `--judges` the judge can be any model, and
    the old test both mis-flagged claude models judged by someone else and missed
    a real self-judge from another family."""
    names = {str(model.get("id") or "").lower(), str(model.get("model") or "").lower()}
    names.discard("")
    for jm in panel:
        j = str(jm).lower().strip()
        if not j:
            continue
        if j in names:
            return True
        family = j.split("-")[0]
        if family and any(family in n for n in names):
            return True
    return False


def fmt_ci(median_val, ci, unit: str = "") -> str:
    """`1.50s [1.20–1.80]` when a CI is available, else plain `fmt_s_unit`."""
    if median_val is None:
        return "—"
    base = f"{median_val:.2f}{unit}"
    if ci is None:
        return base
    return f"{base} [{ci[0]:.2f}–{ci[1]:.2f}]"


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
        bullets = re.findall(r"^\s*(?:[-•*]|\d+[.)])\s", text, flags=re.MULTILINE)
        n = len(bullets)
        return f"{n} bullets" + (" ✓" if n == 5 else " ✗")
    return ""


def load_run(run_arg: str | None = None):
    """Load one immutable run (or the flat legacy layout when run_arg/latest is
    absent). Returns (records_by_model, last_by_cell, judges, manifest, run_dir).

    records_by_model keeps EVERY record (all repeats) for bootstrap CIs;
    last_by_cell keeps the last record per (model,task) for per-task tables.
    `judges` maps (model,task) → {"last": record, "scores": [score per repeat]}:
    judge.py now scores EVERY repeat, so quality has as many samples per cell as
    latency does. Collapsing them to one record here would recreate exactly the
    mismatch the per-repeat judging was added to fix.
    """
    run_dir = _store.resolve_run_dir(run_arg)
    records_by_model: dict[str, list[dict]] = defaultdict(list)
    last_by_cell: dict[tuple[str, str], dict] = {}
    for jl in _store.result_files(run_dir):
        mid = jl.stem
        for line in jl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            tid = r.get("task_id")
            if tid is None:
                continue
            records_by_model[mid].append(r)
            last_by_cell[(mid, tid)] = r
    judges: dict[tuple[str, str], dict] = {}
    jf = _store.judge_file(run_dir)
    if jf.exists():
        by_repeat: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for line in jf.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            key = (d.get("model_id"), d.get("task_id"))
            if None in key:
                continue
            # Last write wins PER REPEAT (a rescore of that repeat), so repeats
            # accumulate instead of overwriting each other.
            by_repeat[key][d.get("repeat_idx") or 0] = d
        for key, per_rep in by_repeat.items():
            recs = [per_rep[k] for k in sorted(per_rep)]
            scores = [r["score"] for r in recs if r.get("score") is not None]
            judges[key] = {"last": recs[-1], "scores": scores, "n_repeats": len(recs)}
    manifest = _store.load_manifest(run_dir)
    return records_by_model, last_by_cell, judges, manifest, run_dir


def main():
    ap = argparse.ArgumentParser(description="Generate the markdown benchmark report.")
    ap.add_argument("--run", help="run id / run dir / manifest.json to report (default: latest run)")
    ap.add_argument("--baseline", help="run id / run dir / manifest.json to flag regressions against")
    ap.add_argument("--out", help="output markdown path (default: $BENCH_REPORT_OUT, else the "
                                  "gitignored repo-root LLM_MODELS_BENCH_2026-05-15.md)")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else OUT

    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
    tasks_data = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    tasks = tasks_data["tasks"]
    task_ids = [t["id"] for t in tasks]
    # Methodology section reports the per-request budget; derive it from
    # _meta.max_output_tokens (the runner's source of truth), not a hardcode.
    # Ollama uses num_predict = MAX_TOKENS * 2 in run.py.
    max_output_tokens = tasks_data.get("_meta", {}).get("max_output_tokens", 2048)

    # Load the run being reported (latest, or --run) and the optional baseline.
    records_by_model, results, judges, manifest, run_dir = load_run(args.run)
    repeats = int(manifest.get("repeats", 1)) if manifest else 1

    # Judges that actually scored this run (recorded by judge.py in the manifest).
    judge_panel = (manifest or {}).get("judge_panel") or DEFAULT_JUDGE_PANEL

    baseline_q: dict[str, float] = {}
    baseline_total: dict[str, float] = {}
    baseline_dir = None
    if args.baseline:
        b_recs, _b_last, b_judges, _b_manifest, baseline_dir = load_run(args.baseline)
        for m in models:
            mid = m["id"]
            b_tot = [r["total_s"] for r in b_recs.get(mid, [])
                     if not r.get("error") and (r.get("text") or "").strip() and r.get("total_s") is not None]
            if b_tot:
                baseline_total[mid] = statistics.median(b_tot)
            # Same estimator as the current run (mean of task means) — see
            # task_score_groups.
            b_q = quality_from_groups(task_score_groups(b_judges, mid, task_ids))
            if b_q is not None:
                baseline_q[mid] = b_q

    # Actual run-date span of the loaded raw records (ts = "YYYY-MM-DDT..."). The
    # report title/filename is a fixed 2026-05-15 snapshot, so surface when the
    # tables actually aggregate cells collected on different dates.
    run_dates = sorted({r["ts"][:10] for r in results.values()
                        if isinstance(r.get("ts"), str) and len(r["ts"]) >= 10})

    model_by_id = {m["id"]: m for m in models}

    # === Aggregates per model ===
    regressions: list[str] = []
    rows = []
    for m in models:
        mid = m["id"]
        if m.get("skip_reason"):
            rows.append({
                "id": mid, "provider": m["provider"], "model": m["model"],
                "skip": m["skip_reason"], "ok": 0, "n": 0,
            })
            continue
        # Latency samples span ALL repeats (records_by_model), so the bootstrap CI
        # reflects cold/warm spread; ok/empty/errors stay per-task (last record).
        ttfts, ttfts_r, totals = [], [], []
        for r in records_by_model.get(mid, []):
            if r.get("error") or not (r.get("text") or "").strip():
                continue
            if r.get("ttft_s") is not None:
                ttfts.append(r["ttft_s"])
            if r.get("ttft_reasoning_s") is not None:
                ttfts_r.append(r["ttft_reasoning_s"])
            if r.get("total_s") is not None:
                totals.append(r["total_s"])
        # Quality: EVERY judged repeat is a sample, grouped BY TASK. quality_avg
        # averages within a task first, then across tasks, so a task with more
        # completed repeats does not weigh more; the CI bootstraps that same
        # two-level structure (see bootstrap_ci_task_mean).
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
        groups = task_score_groups(judges, mid, task_ids)
        scores = [s for g in groups for s in g]
        task_means = [statistics.mean(g) for g in groups]
        total_p50 = statistics.median(totals) if totals else None
        quality_avg = quality_from_groups(groups)
        # Regression flags vs baseline (idea 16/18).
        regressed = False
        if mid in baseline_total and total_p50 is not None and baseline_total[mid] > 0:
            grew = (total_p50 - baseline_total[mid]) / baseline_total[mid] * 100
            if grew > LATENCY_REGRESSION_PCT:
                regressed = True
                regressions.append(
                    f"`{mid}` — Total p50 {baseline_total[mid]:.2f}s → {total_p50:.2f}s (+{grew:.0f}%)")
        if mid in baseline_q and quality_avg is not None:
            drop = baseline_q[mid] - quality_avg
            if drop > QUALITY_REGRESSION_DELTA:
                regressed = True
                regressions.append(
                    f"`{mid}` — Q {baseline_q[mid]:.2f} → {quality_avg:.2f} (−{drop:.2f})")
        rows.append({
            "id": mid,
            "provider": m["provider"],
            "model": m["model"],
            "n": len(task_ids),
            "ok": ok,
            "empty_text": empty_text,
            "ttft_p50": statistics.median(ttfts) if ttfts else None,
            "ttft_ci": bootstrap_ci(ttfts, repeats),
            # p90 needs ≥5 samples to be meaningful; below that we return None
            # (renders as '—') instead of falling back to max(), which would
            # be misleadingly labeled p90.
            # method="inclusive" keeps p90 on the same percentile definition as the
            # p50 medians above (statistics.median == inclusive 50th pct) — the
            # default "exclusive" diverges on the small bench samples.
            "ttft_p90": statistics.quantiles(ttfts, n=10, method="inclusive")[8] if len(ttfts) >= 5 else None,
            "ttft_r_p50": statistics.median(ttfts_r) if ttfts_r else None,
            "total_p50": total_p50,
            "total_ci": bootstrap_ci(totals, repeats),
            "total_p90": statistics.quantiles(totals, n=10, method="inclusive")[8] if len(totals) >= 5 else None,
            "quality_avg": quality_avg,
            "quality_ci": bootstrap_ci_task_mean(groups, repeats),
            # Judged samples (all repeats) and how many TASKS they cover — the
            # two are different denominators and are reported separately.
            "quality_n": len(scores),
            "quality_tasks": len(task_means),
            # Coverage-penalized quality: a model that answered 2/8 tasks at Q5
            # should NOT outrank a stable 8/8 model at Q4.6. Coverage is measured
            # in TASKS (not samples) — with repeats>1 the sample count exceeds the
            # task count and would inflate the penalty term above 1.
            "quality_eff": (
                quality_avg * len(task_means) / len(task_ids)
                if task_means and task_ids else None
            ),
            # A model from the judging panel's own family scores itself — prone
            # to self-preference bias. Flagged with '†'. Derived from the panel
            # actually used for this run, not a hardcoded provider.
            "self_judged": self_judged(m, judge_panel),
            "errors": errors,
            "regressed": regressed,
        })

    # === Build markdown ===
    lines = []
    lines.append("# LLM Models Benchmark — 2026-05-15")
    lines.append("")
    lines.append(f"**Запущен с:** локальная Windows-машина")
    lines.append(f"**Моделей:** {len(rows)} ({sum(1 for r in rows if not r.get('skip'))} активных)")
    lines.append(f"**Задач:** {len(tasks)} (RU-edit, YT-summary EN/RU, JSON-extract, RU→EN translate, classify, bash one-liner, Python function)")
    judge_str = ", ".join(f"`{j}`" for j in judge_panel)
    panel_note = "" if (manifest or {}).get("judge_panel") else " (панель не записана в manifest — предполагается дефолтная)"
    lines.append(f"**Judge:** {judge_str} (через agent server, температура 0){panel_note}")
    if manifest:
        lines.append(
            f"**Run:** `{manifest.get('run_id', '?')}` · started {manifest.get('started_at', '?')} · "
            f"git `{(manifest.get('git_sha') or '—')[:12]}` · repeats={manifest.get('repeats', 1)} · "
            f"seed={manifest.get('seed', 0)} · status={manifest.get('status', 'unknown')}"
        )
    elif run_dir is None:
        lines.append("**Run:** legacy flat `results/*.jsonl` (no manifest)")
    # A run dir is created before the first cell, so an interrupted run is
    # indistinguishable from a finished one unless the lifecycle is checked. Say
    # so IN THE REPORT — rankings built from a partial matrix used to read as
    # final.
    run_state = _store.run_status(run_dir)
    if run_dir is not None and not run_state["complete"]:
        lines.append("")
        lines.append(
            f"> ⚠️ **Неполный прогон:** run `{run_dir.name}` в статусе "
            f"`{run_state['status']}` ({run_state['completed_cells']}/"
            f"{run_state['expected_cells']} ячеек). Таблицы и ранжирование ниже "
            f"построены на НЕПОЛНОЙ матрице. Дособрать: "
            f"`python run.py --resume {run_dir.name}`."
        )
    if run_dates:
        if len(run_dates) == 1:
            lines.append(f"**Даты прогонов (из raw-данных):** {run_dates[0]}")
        else:
            lines.append(
                f"> ⚠️ **Смешанные прогоны:** raw-данные собраны в {len(run_dates)} разных дат "
                f"({run_dates[0]} … {run_dates[-1]}). Таблицы ниже агрегируют разные прогоны "
                f"под фиксированным заголовком 2026-05-15, а не единый снимок."
            )
    lines.append("")
    lines.append(
        f"> ⚠️ **Self-judge bias (гипотеза, не измерено):** judge — {judge_str}, и модели "
        "того же семейства судят сами себя (помечены `†`). Их Q могут быть завышены "
        "из-за self-preference — это правдоподобная гипотеза, но без контрольного judge она "
        "не подтверждена; сравнивать их с другими провайдерами с осторожностью. "
        "`*` у Q = оценка по неполному покрытию задач (quality_tasks < задач) — "
        "ранжирование использует покрытие-взвешенный Q, не сырой средний."
    )
    lines.append("")

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
    lines.append(
        "- **CI (латенси)** — `median [lo–hi]` = 95%-перцентильный bootstrap CI МЕДИАНЫ "
        f"({BOOTSTRAP_RESAMPLES} ресемплов, seed {BOOTSTRAP_SEED}); показывается только при repeats≥2 или ≥5 семплах"
    )
    lines.append(
        "- **CI (Q)** — интервал для ТОЙ ЖЕ величины, что в колонке: среднее по задачам. "
        "Иерархический bootstrap (ресемпл задач, внутри задачи — ресемпл повторов), "
        "поэтому это интервал среднего, а не медианы общего пула оценок"
    )
    lines.append(
        "- **Sampling (latency vs quality):** обе метрики берут ВСЕ повторы. "
        "Латенси — один семпл на каждую (модель, задача, повтор); качество — одна "
        "судейская оценка на тот же кортеж (judge оценивает каждый повтор отдельно). "
        "`quality_n` = число оценённых семплов, `quality_tasks` = сколько ЗАДАЧ они "
        "покрывают; `Q` усредняется сначала внутри задачи, потом по задачам — чтобы "
        "задача с бо́льшим числом удачных повторов не весила больше остальных."
    )
    lines.append(f"- Параметры запросов: `temperature=0.2`, `max_tokens={max_output_tokens}` (Ollama: `num_predict={max_output_tokens * 2}`)")
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
        if r["quality_avg"] is not None:
            q = fmt_ci(r["quality_avg"], r.get("quality_ci"))
        else:
            q = "—"
        if r["quality_avg"] is not None and r.get("quality_tasks", 0) < r["n"]:
            q += "*"  # partial TASK coverage (repeats inflate quality_n)
        if r.get("self_judged"):
            q += "†"  # self-judged family
        ok_str = f"{r['ok']}/{r['n']}"
        if r.get("empty_text"):
            ok_str += f" (+{r['empty_text']} empty)"
        id_cell = f"`{r['id']}`" + (" ⚠️REGRESSION" if r.get("regressed") else "")
        lines.append(
            f"| {id_cell} | {r['provider']} | "
            f"{fmt_ci(r['ttft_p50'], r.get('ttft_ci'), 's')} | {fmt_s_unit(r['ttft_p90'])} | "
            f"{fmt_s_unit(r.get('ttft_r_p50'))} | "
            f"{fmt_ci(r['total_p50'], r.get('total_ci'), 's')} | {fmt_s_unit(r['total_p90'])} | "
            f"{q} | {ok_str} |"
        )
    lines.append("")

    # Regression summary vs baseline (top of report body, per idea 18).
    if baseline_dir is not None:
        lines.append("## ⚠️ Регрессии vs baseline")
        lines.append("")
        lines.append(
            f"Baseline: `{baseline_dir.name}`. Пороги: latency +{LATENCY_REGRESSION_PCT}%, "
            f"quality −{QUALITY_REGRESSION_DELTA}."
        )
        lines.append("")
        if regressions:
            for reg in regressions:
                lines.append(f"- {reg}")
        else:
            lines.append("_Регрессий не обнаружено._")
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
            ttft = r.get("ttft_s")
            ttft_r = r.get("ttft_reasoning_s")
            total = r.get("total_s")
            err = r.get("error")
            text = r.get("text") or ""
            h = heuristic(tid, text) if not err else ""
            # Per-task table shows the LAST repeat's score (one row per cell);
            # the aggregate tables above use every repeat.
            score = j["last"].get("score") if j else None
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
        # error/empty rows last (so they don't hide among merely-unscored rows),
        # then by quality desc, then ttft asc
        task_rows.sort(key=lambda x: (
            1 if x["display_err"] else 0,
            -(x["score"] if x["score"] is not None else -1),
            x["ttft"] if x["ttft"] is not None else 9999,
        ))
        for x in task_rows:
            if x["display_err"]:
                lines.append(f"| `{x['mid']}` | — | — | — | — | {x['display_err']} |")
            else:
                q = str(x["score"]) if x["score"] is not None else "—"
                lines.append(
                    f"| `{x['mid']}` | {fmt_s_unit(x['ttft'])} | {fmt_s_unit(x['ttft_r'])} | "
                    f"{fmt_s_unit(x['total'])} | {q} | {x['heur']} |"
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
            mark = ("*" if r.get("quality_tasks", 0) < r["n"] else "") + ("†" if r.get("self_judged") else "")
            lines.append(
                f"- `{r['id']}` — quality {r['quality_avg']:.2f}{mark} "
                f"(eff {r['quality_eff']:.2f}), TTFT p50 {fmt_s_unit(r['ttft_p50'])}"
            )
        lines.append("")
        lines.append("**Топ-5 по balance (TTFT/quality):**")
        for r in balanced:
            # Same max(0.5, …) guard as the sort key above — a model whose judge
            # avg is exactly 0.0 (all answers garbage/refused) passed the
            # `quality_avg is not None` filter and would ZeroDivisionError here.
            ratio = r["ttft_p50"] / max(0.5, r["quality_avg"])
            lines.append(f"- `{r['id']}` — TTFT/Q = {ratio:.2f}, TTFT {r['ttft_p50']:.2f}s, Q {r['quality_avg']:.2f}")
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
    preserved = _extract_manual_tldr(out_path)
    if preserved is not None:
        try:
            i = lines.index(TLDR_BEGIN)
            j = lines.index(TLDR_END)
            lines[i:j + 1] = preserved
        except ValueError:
            pass

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out_path} ({len(lines)} lines, {out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
