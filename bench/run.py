"""LLM benchmark runner with TTFT measurement.

Usage:
    python run.py                 # full bench: all models x all tasks
    python run.py --smoke         # 1 task x 1 model per provider (sanity check)
    python run.py --task T4_json_extract --model ocg-minimax-m2.7   # single cell
    python run.py --providers opencode_go,gemini   # subset by provider

Reads:
    bench/models.json
    bench/prompts/tasks.json
    secrets/vault.env

Writes:
    bench/results/<model_id>.jsonl  (one line per task)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

import _store

ROOT = Path(__file__).resolve().parent
# Default: <repo>/secrets/vault.env (assumes bench/ sits at repo root).
# Override with VAULT_PATH env var if the layout differs or bench/ is a symlink.
VAULT = Path(os.environ.get("VAULT_PATH") or (ROOT.parent / "secrets" / "vault.env"))
MODELS_JSON = ROOT / "models.json"
TASKS_JSON = ROOT / "prompts" / "tasks.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TIMEOUT_CONNECT = 15.0
TIMEOUT_READ = 240.0
MAX_TOKENS = 2048
TEMPERATURE = 0.2


def load_vault() -> dict[str, str]:
    env: dict[str, str] = {}
    if not VAULT.exists():
        sys.stderr.write(f"vault not found: {VAULT}\n")
        return env
    # utf-8-sig: a vault.env written by a Windows editor / PowerShell can carry a
    # BOM, which plain utf-8 keeps as an invisible prefix on the FIRST key — so
    # env["﻿OPENCODE_GO_KEY"] silently never matches the lookup.
    for line in VAULT.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ============================================================================
# Per-API streaming implementations. All return (ttft_s, total_s, text, tok_out, err)
# ============================================================================

def call_openai(client: httpx.Client, endpoint: str, model: str, system: str, user: str, api_key: str | None) -> dict:
    """OpenAI-compatible streaming via SSE. Falls back to non-stream JSON if server ignores stream:true."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    ttft_reasoning: float | None = None
    buf: list[str] = []
    reasoning_buf: list[str] = []
    tok_out: int | None = None
    http_status: int = 0
    try:
        with client.stream(
            "POST", url, headers=headers, json=body,
            timeout=httpx.Timeout(TIMEOUT_READ, connect=TIMEOUT_CONNECT),
        ) as r:
            http_status = r.status_code
            if r.status_code != 200:
                err_body = r.read().decode("utf-8", errors="replace")[:500]
                return {
                    "ttft_s": None, "ttft_reasoning_s": None,
                    "total_s": time.perf_counter() - t0,
                    "text": "", "reasoning_text": "",
                    "tok_out": None, "http_status": r.status_code,
                    "streaming": False, "error": f"HTTP {r.status_code}: {err_body}",
                }
            content_type = r.headers.get("content-type", "")
            if "event-stream" not in content_type:
                full = r.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(full)
                    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                    text = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning_content", "") or ""
                    u = data.get("usage") or {}
                    tok_out = u.get("completion_tokens") or u.get("output_tokens")
                    total = time.perf_counter() - t0
                    if tok_out is None and text:
                        tok_out = max(1, len(text) // 4)
                    return {
                        "ttft_s": total, "ttft_reasoning_s": total,
                        "total_s": total, "text": text, "reasoning_text": reasoning,
                        "tok_out": tok_out, "http_status": http_status,
                        "streaming": False, "error": None,
                    }
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    return {
                        "ttft_s": None, "ttft_reasoning_s": None,
                        "total_s": time.perf_counter() - t0,
                        "text": full[:500], "reasoning_text": "",
                        "tok_out": None, "http_status": http_status,
                        "streaming": False, "error": f"non-stream parse: {e}",
                    }
            saw_done = False
            finish_reason: str | None = None
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "usage" in chunk and chunk["usage"]:
                        u = chunk["usage"]
                        tok_out = u.get("completion_tokens") or u.get("output_tokens") or tok_out
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    delta = choices[0].get("delta") or {}
                    rcontent = delta.get("reasoning_content")
                    if rcontent:
                        if ttft_reasoning is None:
                            ttft_reasoning = time.perf_counter() - t0
                        reasoning_buf.append(rcontent)
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        buf.append(content)
        total = time.perf_counter() - t0
        text = "".join(buf)
        reasoning = "".join(reasoning_buf)
        if tok_out is None and text:
            tok_out = max(1, len(text) // 4)
        # A stream that never reached a terminal marker ([DONE] or a
        # finish_reason) was truncated/dropped/malformed — regardless of how much
        # partial text arrived. The old guard only fired when the text was EMPTY,
        # so "half an answer, then the connection dropped" was recorded as a clean
        # success and entered latency/quality rankings as a completed response.
        # The partial text is still stored for diagnosis; `error` keeps it out of
        # the comparative metrics.
        err = None
        if not saw_done and finish_reason is None:
            err = (
                f"stream ended without [DONE]/finish_reason "
                f"({len(text)} chars of partial text kept for diagnosis)"
            )
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_reasoning,
            "total_s": total, "text": text, "reasoning_text": reasoning,
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": err,
        }
    except httpx.TimeoutException as e:
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_reasoning,
            "total_s": time.perf_counter() - t0,
            "text": "".join(buf), "reasoning_text": "".join(reasoning_buf),
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": f"timeout: {type(e).__name__}",
        }
    except Exception as e:
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_reasoning,
            "total_s": time.perf_counter() - t0,
            "text": "".join(buf), "reasoning_text": "".join(reasoning_buf),
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": f"{type(e).__name__}: {e}",
        }


def call_gemini(client: httpx.Client, endpoint: str, model: str, system: str, user: str, api_key: str) -> dict:
    """Google Gemini streamGenerateContent with SSE."""
    # Pass API key via header to keep it out of URL/access logs.
    url = f"{endpoint.rstrip('/')}/models/{model}:streamGenerateContent?alt=sse"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS,
        },
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    buf: list[str] = []
    tok_out: int | None = None
    http_status: int = 0
    finish_reason: str | None = None
    try:
        with client.stream(
            "POST", url, headers=headers, json=body,
            timeout=httpx.Timeout(TIMEOUT_READ, connect=TIMEOUT_CONNECT),
        ) as r:
            http_status = r.status_code
            if r.status_code != 200:
                err_body = r.read().decode("utf-8", errors="replace")[:500]
                return {
                    "ttft_s": None, "ttft_reasoning_s": None,
                    "total_s": time.perf_counter() - t0,
                    "text": "", "reasoning_text": "",
                    "tok_out": None, "http_status": r.status_code,
                    "streaming": False, "error": f"HTTP {r.status_code}: {err_body}",
                }
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                cands = chunk.get("candidates") or []
                if not cands:
                    continue
                # Gemini's terminal marker: `finishReason` on the candidate.
                # Without checking it, ANY EOF looked like a completed answer.
                if cands[0].get("finishReason"):
                    finish_reason = cands[0]["finishReason"]
                parts = (cands[0].get("content") or {}).get("parts") or []
                for p in parts:
                    txt = p.get("text")
                    if txt:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        buf.append(txt)
                um = chunk.get("usageMetadata")
                if um:
                    tok_out = um.get("candidatesTokenCount") or tok_out
        total = time.perf_counter() - t0
        text = "".join(buf)
        if tok_out is None and text:
            tok_out = max(1, len(text) // 4)
        # Require the protocol's OWN terminal success, like the OpenAI path.
        # `STOP` is a normal completion; MAX_TOKENS/SAFETY/RECITATION are real
        # truncations and must not be compared against complete answers.
        err = None
        if finish_reason is None:
            err = (
                f"stream ended without finishReason "
                f"({len(text)} chars of partial text kept for diagnosis)"
            )
        elif finish_reason not in ("STOP", "FINISH_REASON_STOP"):
            err = f"truncated by provider (finishReason={finish_reason})"
        return {
            "ttft_s": ttft, "ttft_reasoning_s": None,
            "total_s": total, "text": text, "reasoning_text": "",
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": err,
        }
    except httpx.TimeoutException as e:
        return {
            "ttft_s": ttft, "ttft_reasoning_s": None,
            "total_s": time.perf_counter() - t0,
            "text": "".join(buf), "reasoning_text": "",
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": f"timeout: {type(e).__name__}",
        }
    except Exception as e:
        return {
            "ttft_s": ttft, "ttft_reasoning_s": None,
            "total_s": time.perf_counter() - t0,
            "text": "".join(buf), "reasoning_text": "",
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": f"{type(e).__name__}: {e}",
        }


def call_ollama(client: httpx.Client, endpoint: str, model: str, system: str, user: str) -> dict:
    """Ollama /api/chat streaming (NDJSON)."""
    url = f"{endpoint.rstrip('/')}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS * 2},
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    ttft_thinking: float | None = None
    buf: list[str] = []
    thinking_buf: list[str] = []
    tok_out: int | None = None
    http_status: int = 0
    saw_done = False
    try:
        with client.stream(
            "POST", url, headers={"Content-Type": "application/json"}, json=body,
            timeout=httpx.Timeout(TIMEOUT_READ, connect=TIMEOUT_CONNECT),
        ) as r:
            http_status = r.status_code
            if r.status_code != 200:
                err_body = r.read().decode("utf-8", errors="replace")[:500]
                return {
                    "ttft_s": None, "ttft_reasoning_s": None,
                    "total_s": time.perf_counter() - t0,
                    "text": "", "reasoning_text": "",
                    "tok_out": None, "http_status": r.status_code,
                    "streaming": False, "error": f"HTTP {r.status_code}: {err_body}",
                }
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message") or {}
                thinking = msg.get("thinking")
                if thinking:
                    if ttft_thinking is None:
                        ttft_thinking = time.perf_counter() - t0
                    thinking_buf.append(thinking)
                content = msg.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    buf.append(content)
                if chunk.get("done"):
                    saw_done = True
                    tok_out = chunk.get("eval_count")
                    break
        total = time.perf_counter() - t0
        text = "".join(buf)
        thinking_text = "".join(thinking_buf)
        if tok_out is None and text:
            tok_out = max(1, len(text) // 4)
        # Ollama's terminal marker is `done: true`. Exiting the loop on EOF
        # without it means the NDJSON stream was cut — not a finished answer.
        err = None
        if not saw_done:
            err = (
                f"stream ended without done=true "
                f"({len(text)} chars of partial text kept for diagnosis)"
            )
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_thinking,
            "total_s": total, "text": text, "reasoning_text": thinking_text,
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": err,
        }
    except httpx.TimeoutException as e:
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_thinking,
            "total_s": time.perf_counter() - t0,
            "text": "".join(buf), "reasoning_text": "".join(thinking_buf),
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": f"timeout: {type(e).__name__}",
        }
    except Exception as e:
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_thinking,
            "total_s": time.perf_counter() - t0,
            "text": "".join(buf), "reasoning_text": "".join(thinking_buf),
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": f"{type(e).__name__}: {e}",
        }


def run_one(client: httpx.Client, model_cfg: dict, task: dict, env: dict[str, str]) -> dict:
    api = model_cfg.get("api", "openai")
    auth_env = model_cfg.get("auth_env")
    api_key = env.get(auth_env) if auth_env else None
    if api == "openai":
        return call_openai(client, model_cfg["endpoint"], model_cfg["model"], task["system"], task["user"], api_key)
    if api == "gemini":
        if not api_key:
            return {"ttft_s": None, "ttft_reasoning_s": None, "total_s": 0,
                    "text": "", "reasoning_text": "", "tok_out": None,
                    "http_status": 0, "streaming": False, "error": "no api key"}
        return call_gemini(client, model_cfg["endpoint"], model_cfg["model"], task["system"], task["user"], api_key)
    if api == "ollama":
        return call_ollama(client, model_cfg["endpoint"], model_cfg["model"], task["system"], task["user"])
    return {"ttft_s": None, "ttft_reasoning_s": None, "total_s": 0,
            "text": "", "reasoning_text": "", "tok_out": None,
            "http_status": 0, "streaming": False, "error": f"unknown api {api}"}


# ============================================================================
# Immutable-run bookkeeping (idea 15)
# ============================================================================

def _git_sha() -> str | None:
    """Current HEAD SHA of the repo, or None if git is unavailable / not a repo."""
    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        return None
    return None


def _sha16(data: bytes) -> str:
    """First 16 hex chars of sha256 — enough to fingerprint config file content."""
    return hashlib.sha256(data).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 task x 1 model per provider")
    ap.add_argument("--task", help="run only this task id")
    ap.add_argument("--model", help="run only this model id")
    ap.add_argument("--providers", help="comma-separated provider filter")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a (model,task,repeat) cell already recorded in the run dir")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue an existing run instead of creating a new one "
                         "(implies --skip-existing; reuses its manifest/settings)")
    ap.add_argument("--include-broken", action="store_true", help="include models with skip_reason set")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each (model,task) N times (statistical protocol, idea 16)")
    ap.add_argument("--warmup", action="store_true",
                    help="one discarded call per model before scored cells (warms connection/model)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for randomized (model,task,repeat) interleaving")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    env = load_vault()
    models_data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    tasks_data = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    models = models_data["models"]
    tasks = tasks_data["tasks"]

    # Honour tasks.json `_meta.max_output_tokens` as the per-request token budget.
    # `_meta` is not a task (the loop below already iterates `tasks` only).
    global MAX_TOKENS
    MAX_TOKENS = tasks_data.get("_meta", {}).get("max_output_tokens", MAX_TOKENS)

    # Endpoint overrides for self-hosted providers (kept out of the committed
    # models.json, which ships localhost defaults). Set OLLAMA_BASE_URL /
    # CLAUDE_AGENT_BASE_URL in the environment or in secrets/vault.env to point
    # at your own hosts.
    _overrides = {
        "ollama": os.environ.get("OLLAMA_BASE_URL") or env.get("OLLAMA_BASE_URL"),
        "claude_agent": os.environ.get("CLAUDE_AGENT_BASE_URL") or env.get("CLAUDE_AGENT_BASE_URL"),
    }
    for m in models:
        ovr = _overrides.get(m["provider"])
        if ovr:
            m["endpoint"] = ovr

    # Drop skip_reason'd models BEFORE smoke dedup, otherwise smoke can pick a
    # broken model as the provider representative and the whole provider is skipped.
    if not args.include_broken and not args.model:
        models = [m for m in models if not m.get("skip_reason")]

    if args.smoke:
        seen = set()
        smoke_models = []
        for m in models:
            if m["provider"] in seen:
                continue
            seen.add(m["provider"])
            smoke_models.append(m)
        models = smoke_models
        # Use the requested task under smoke; otherwise the general --task filter
        # below would intersect with a hardcoded T1 and yield zero cells.
        smoke_task_id = args.task or "T1_ru_edit_short"
        tasks = [t for t in tasks if t["id"] == smoke_task_id]
        sys.stderr.write(f"SMOKE: {len(models)} models x {len(tasks)} task\n")

    if args.providers:
        keep = set(args.providers.split(","))
        models = [m for m in models if m["provider"] in keep]
    if args.model:
        models = [m for m in models if m["id"] == args.model]
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]

    # === Statistical protocol: randomized (model,task,repeat) interleaving (idea 16) ===
    # Spreading repeats and models across time (rather than looping model-by-model)
    # keeps a transient provider hiccup from biasing one model's whole row.
    cells = [(m, t, rep) for m in models for t in tasks for rep in range(args.repeats)]
    rng = random.Random(args.seed)
    rng.shuffle(cells)

    # === Run directory + manifest with a real lifecycle (idea 15) ===
    # The manifest records status/expected_cells up front and is REWRITTEN with
    # status="completed" + completed_cells at the end. Only then does
    # latest-complete.txt move. Previously the run dir and latest.txt were created
    # before the first cell and never updated, so an interrupted run was
    # indistinguishable from a finished one and judge/report happily built
    # rankings from a partial matrix.
    resume_id = args.resume
    if resume_id:
        run_dir = _store.resolve_run_dir(resume_id)
        if run_dir is None:
            ap.error(f"--resume: unknown run id {resume_id!r}")
        run_id = run_dir.name
        prior = _store.load_manifest(run_dir) or {}
        started_at = prior.get("started_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        args.skip_existing = True
        sys.stderr.write(f"RESUME {run_id} → {run_dir}\n")
    else:
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_dir = _store.RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "status": "started",
        "started_at": started_at,
        "completed_at": None,
        "expected_cells": len(cells),
        "completed_cells": 0,
        "git_sha": _git_sha(),
        "models_json_sha": _sha16(MODELS_JSON.read_bytes()),
        "tasks_json_sha": _sha16(TASKS_JSON.read_bytes()),
        "cli_args": vars(args),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "repeats": args.repeats,
        "seed": args.seed,
        "python_version": platform.python_version(),
    }

    def _write_manifest() -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _write_manifest()
    # latest.txt = "most recently STARTED run" (may be partial). The
    # latest-complete pointer only moves on success, below.
    _store.LATEST_TXT.write_text(run_id, encoding="utf-8")
    sys.stderr.write(f"RUN {run_id} → {run_dir}\n")

    order = ", ".join(f"{m['id']}/{t['id']}#{rep}" for m, t, rep in cells)
    sys.stderr.write(f"ORDER (seed={args.seed}, {len(cells)} cells): {order}\n")

    # Skip-existing works within the run dir: on a fresh invocation the dir starts
    # empty so nothing is skipped; with --resume (or an explicitly reused dir) a
    # cell with a prior non-error record is skipped. Keyed by (task_id, repeat_idx)
    # so resuming a --repeats=3 run doesn't collapse the three repeats into one.
    existing: dict[str, set[tuple[str, int]]] = {}
    if args.skip_existing:
        for m in models:
            f = run_dir / f"{m['id']}.jsonl"
            if not f.exists():
                continue
            latest: dict[tuple[str, int], dict] = {}
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                tid = r.get("task_id")
                if tid is not None:
                    latest[(tid, r.get("repeat_idx") or 0)] = r
            existing[m["id"]] = {k for k, r in latest.items() if not r.get("error")}

    # One reusable HTTP client for every call — persistent connection pool instead
    # of a fresh TCP/TLS handshake per request (idea 16).
    client = httpx.Client(timeout=httpx.Timeout(TIMEOUT_READ, connect=TIMEOUT_CONNECT))
    # Open every per-model handle up front; write as cells complete (interleaved),
    # close all in finally.
    handles: dict[str, Any] = {}
    total_cells = len(cells)
    cell = 0
    done_cells = 0
    try:
        # Warmup: one discarded call per model to warm the connection / model
        # weights before the scored cells (idea 16). Not written to results.
        if args.warmup and tasks:
            for m in models:
                sys.stderr.write(f"WARMUP {m['id']} (discarded)... ")
                sys.stderr.flush()
                wr = run_one(client, m, tasks[0], env)
                sys.stderr.write(f"{'ok' if not wr['error'] else wr['error'][:40]}\n")

        for m, t, rep in cells:
            cell += 1
            mid = m["id"]
            if args.skip_existing and (t["id"], rep) in existing.get(mid, set()):
                sys.stderr.write(f"[{cell}/{total_cells}] SKIP {mid} / {t['id']} #{rep} (exists)\n")
                done_cells += 1
                continue
            sys.stderr.write(f"[{cell}/{total_cells}] {mid} / {t['id']} #{rep}... ")
            sys.stderr.flush()
            res = run_one(client, m, t, env)
            ttft = f"{res['ttft_s']:.2f}s" if res["ttft_s"] is not None else "—"
            ttftr = f"{res.get('ttft_reasoning_s'):.2f}s" if res.get("ttft_reasoning_s") is not None else "—"
            total = f"{res['total_s']:.2f}s"
            rlen = len(res.get("reasoning_text") or "")
            tlen = len(res.get("text") or "")
            err = res["error"] or "ok"
            sys.stderr.write(
                f"ttft={ttft} ttftR={ttftr} total={total} "
                f"txt={tlen}b rsn={rlen}b status={res['http_status']} {err[:50]}\n"
            )
            record = {
                "run_id": run_id,
                "model_id": mid,
                "task_id": t["id"],
                "repeat_idx": rep,
                # First repeat of a (model,task) is the cold call (no warm cache /
                # connection reuse for that pair); the rest are warm.
                "cold": rep == 0,
                "ttft_s": res["ttft_s"],
                "ttft_reasoning_s": res.get("ttft_reasoning_s"),
                "total_s": res["total_s"],
                "tok_out": res["tok_out"],
                "http_status": res["http_status"],
                "streaming": res["streaming"],
                "error": res["error"],
                "text": res["text"],
                "reasoning_text": res.get("reasoning_text", ""),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            fh = handles.get(mid)
            if fh is None:
                fh = (run_dir / f"{mid}.jsonl").open("a", encoding="utf-8")
                handles[mid] = fh
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            done_cells += 1
        completed = True
    except BaseException:
        # Ctrl-C, a crash, anything: the run did NOT finish, so it must never be
        # published as the complete one.
        completed = False
        raise
    finally:
        for fh in handles.values():
            fh.close()
        client.close()
        manifest["completed_cells"] = done_cells
        manifest["completed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["status"] = "completed" if completed and done_cells >= total_cells else "failed"
        _write_manifest()
        if manifest["status"] == "completed":
            # Only NOW does the complete-pointer move — judge.py / report.py
            # default to it, so an interrupted run can't be silently ranked.
            _store.LATEST_COMPLETE_TXT.write_text(run_id, encoding="utf-8")
            sys.stderr.write(f"RUN {run_id} completed ({done_cells}/{total_cells} cells)\n")
        else:
            sys.stderr.write(
                f"RUN {run_id} INCOMPLETE ({done_cells}/{total_cells} cells) — "
                f"resume with: python run.py --resume {run_id}\n"
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
