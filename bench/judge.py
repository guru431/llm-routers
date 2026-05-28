"""LLM-as-judge scoring for benchmark results.

For each (task, model_response) pair, asks claude-opus-4-8 via agent server
to score the response 0-5 with brief reasoning. Writes scores to
bench/results/_judge.jsonl (key: model_id+task_id).

Usage:
    python judge.py                    # score all results, skip already-scored
    python judge.py --rescore          # rescore everything
    python judge.py --task T4_json_extract  # only this task
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
TASKS_JSON = ROOT / "prompts" / "tasks.json"
RESULTS = ROOT / "results"
JUDGE_FILE = RESULTS / "_judge.jsonl"

JUDGE_ENDPOINT = os.environ.get("JUDGE_ENDPOINT", "http://localhost:8765/v1/chat/completions")
JUDGE_MODEL = "claude-opus-4-8"

RUBRIC = {
    "edit": "Оцени правку русской устной речи. 5=идеально (пунктуация, без слов-паразитов, смысл не изменён), 3=ok с минорами, 1=серьёзные искажения смысла, 0=мусор/пусто/отказ.",
    "summarization": "Оцени суммари. 5=РОВНО 5 буллетов, покрывают основные идеи, лимит слов соблюдён, 3=ок но не 5 буллетов или есть преамбула, 1=плохо отражает суть, 0=мусор/пусто.",
    "structured": "Оцени JSON. 5=валидный JSON с правильными значениями всех 4 полей (person, date, time, action), 3=валидный JSON но 1-2 поля упущены/неверны, 1=невалидный JSON, 0=не JSON или мусор.",
    "translate": "Оцени перевод RU→EN. 5=точно, естественно, термины верны, 3=смысл передан с минорами, 1=ошибки/неточности, 0=не английский или мусор.",
    "classify": "Оцени классификацию. Эталон — 'complaint'. 5=точное одно слово complaint в нижнем регистре, 3=правильная категория но с лишним текстом, 1=неверная категория, 0=мусор/не из списка.",
    "code": "Оцени код. 5=решает задачу, синтаксис валиден, идиоматично, 3=работает с минорами, 1=есть ошибка или не решает задачу, 0=мусор/комментарий вместо кода."
}


_SCORE_PATTERNS = (
    re.compile(r"SCORE\s*[:=]\s*([0-5])\b", re.IGNORECASE),
    re.compile(r"\b([0-5])\s*/\s*5\b"),
    re.compile(r"\b([0-5])\s+out\s+of\s+5\b", re.IGNORECASE),
)


def _parse_score(text: str) -> int | None:
    """Extract 0-5 score from judge response. Recognised formats:
    `SCORE: N`, `N/5`, `N out of 5`.

    No bare-digit fallback: a loose `\\b[0-5]\\b` match latches onto numbered
    list items (`1. First reason ...`) and gives wildly wrong scores. If the
    judge violated the requested format we return None — the caller logs the
    raw text and the run can be inspected.
    """
    for pat in _SCORE_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def call_judge(prompt: str) -> dict:
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 250,
    }
    try:
        r = httpx.post(JUDGE_ENDPOINT, json=body, timeout=120.0)
        if r.status_code != 200:
            return {"score": None, "reason": f"judge HTTP {r.status_code}: {r.text[:200]}"}
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        score = _parse_score(text)
        return {"score": score, "reason": text}
    except Exception as e:
        return {"score": None, "reason": f"{type(e).__name__}: {e}"}


def build_prompt(task: dict, response_text: str) -> str:
    rubric = RUBRIC.get(task["category"], "Оцени релевантность и качество ответа.")
    truncated = response_text[:2000]
    return f"""Ты строгий judge для бенчмарка LLM. Задача и эталон ниже.

ЗАДАЧА (категория {task['category']}):
SYSTEM: {task['system']}
USER: {task['user']}

КРИТЕРИЙ: {rubric}

ОТВЕТ МОДЕЛИ (может быть обрезан):
\"\"\"
{truncated}
\"\"\"

Верни строго одну строку формата: `SCORE: N | REASON: краткое обоснование (≤20 слов)`. N — целое 0-5."""


def load_judged() -> set[tuple[str, str]]:
    if not JUDGE_FILE.exists():
        return set()
    seen = set()
    for line in JUDGE_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            seen.add((d["model_id"], d["task_id"]))
        except Exception:
            pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--task", help="filter by task id")
    args = ap.parse_args()

    tasks_by_id = {t["id"]: t for t in json.loads(TASKS_JSON.read_text(encoding="utf-8"))["tasks"]}
    judged = set() if args.rescore else load_judged()

    pairs: list[tuple[str, str, str]] = []  # (model_id, task_id, text)
    for jl in sorted(RESULTS.glob("*.jsonl")):
        if jl.name.startswith("_"):
            continue
        model_id = jl.stem
        for line in jl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or not r.get("text"):
                continue
            tid = r["task_id"]
            if args.task and tid != args.task:
                continue
            if (model_id, tid) in judged:
                continue
            pairs.append((model_id, tid, r["text"]))

    sys.stderr.write(f"Pairs to judge: {len(pairs)}\n")
    mode = "w" if args.rescore else "a"
    with JUDGE_FILE.open(mode, encoding="utf-8") as out:
        for i, (mid, tid, text) in enumerate(pairs, 1):
            task = tasks_by_id.get(tid)
            if not task:
                continue
            sys.stderr.write(f"[{i}/{len(pairs)}] {mid} / {tid}... ")
            sys.stderr.flush()
            res = call_judge(build_prompt(task, text))
            sys.stderr.write(f"score={res['score']}\n")
            rec = {
                "model_id": mid, "task_id": tid,
                "score": res["score"], "reason": res["reason"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
