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
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

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
    for line in VAULT.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ============================================================================
# Per-API streaming implementations. All return (ttft_s, total_s, text, tok_out, err)
# ============================================================================

def call_openai(endpoint: str, model: str, system: str, user: str, api_key: str | None) -> dict:
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
        with httpx.stream(
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
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
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
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_reasoning,
            "total_s": total, "text": text, "reasoning_text": reasoning,
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": None,
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


def call_gemini(endpoint: str, model: str, system: str, user: str, api_key: str) -> dict:
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
    try:
        with httpx.stream(
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
        return {
            "ttft_s": ttft, "ttft_reasoning_s": None,
            "total_s": total, "text": text, "reasoning_text": "",
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": None,
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


def call_ollama(endpoint: str, model: str, system: str, user: str) -> dict:
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
    try:
        with httpx.stream(
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
                    tok_out = chunk.get("eval_count")
                    break
        total = time.perf_counter() - t0
        text = "".join(buf)
        thinking_text = "".join(thinking_buf)
        if tok_out is None and text:
            tok_out = max(1, len(text) // 4)
        return {
            "ttft_s": ttft, "ttft_reasoning_s": ttft_thinking,
            "total_s": total, "text": text, "reasoning_text": thinking_text,
            "tok_out": tok_out, "http_status": http_status,
            "streaming": True, "error": None,
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


def run_one(model_cfg: dict, task: dict, env: dict[str, str]) -> dict:
    api = model_cfg.get("api", "openai")
    auth_env = model_cfg.get("auth_env")
    api_key = env.get(auth_env) if auth_env else None
    if api == "openai":
        return call_openai(model_cfg["endpoint"], model_cfg["model"], task["system"], task["user"], api_key)
    if api == "gemini":
        if not api_key:
            return {"ttft_s": None, "ttft_reasoning_s": None, "total_s": 0,
                    "text": "", "reasoning_text": "", "tok_out": None,
                    "http_status": 0, "streaming": False, "error": "no api key"}
        return call_gemini(model_cfg["endpoint"], model_cfg["model"], task["system"], task["user"], api_key)
    if api == "ollama":
        return call_ollama(model_cfg["endpoint"], model_cfg["model"], task["system"], task["user"])
    return {"ttft_s": None, "ttft_reasoning_s": None, "total_s": 0,
            "text": "", "reasoning_text": "", "tok_out": None,
            "http_status": 0, "streaming": False, "error": f"unknown api {api}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 task x 1 model per provider")
    ap.add_argument("--task", help="run only this task id")
    ap.add_argument("--model", help="run only this model id")
    ap.add_argument("--providers", help="comma-separated provider filter")
    ap.add_argument("--skip-existing", action="store_true", help="skip (model,task) if already in results")
    ap.add_argument("--include-broken", action="store_true", help="include models with skip_reason set")
    args = ap.parse_args()

    env = load_vault()
    models_data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    tasks_data = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    models = models_data["models"]
    tasks = tasks_data["tasks"]

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

    if args.smoke:
        seen = set()
        smoke_models = []
        for m in models:
            if m["provider"] in seen:
                continue
            seen.add(m["provider"])
            smoke_models.append(m)
        models = smoke_models
        tasks = [t for t in tasks if t["id"] == "T1_ru_edit_short"]
        sys.stderr.write(f"SMOKE: {len(models)} models x {len(tasks)} task\n")

    if not args.include_broken and not args.model:
        models = [m for m in models if not m.get("skip_reason")]
    if args.providers:
        keep = set(args.providers.split(","))
        models = [m for m in models if m["provider"] in keep]
    if args.model:
        models = [m for m in models if m["id"] == args.model]
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]

    total_cells = len(models) * len(tasks)
    cell = 0
    for m in models:
        out_file = RESULTS / f"{m['id']}.jsonl"
        existing: set[str] = set()
        if args.skip_existing and out_file.exists():
            for line in out_file.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    existing.add(json.loads(line)["task_id"])
                except Exception:
                    pass
        # Open the per-model jsonl once for the whole task loop. Reduces I/O
        # overhead and removes the open/close-per-task window during which a
        # second run.py for the same model could race the existence check.
        # Concurrent runs of run.py for the SAME model are still unsupported
        # (would need filelock) — single-runner is the only supported mode.
        with out_file.open("a", encoding="utf-8") as out_f:
            for t in tasks:
                cell += 1
                if t["id"] in existing:
                    sys.stderr.write(f"[{cell}/{total_cells}] SKIP {m['id']} / {t['id']} (exists)\n")
                    continue
                sys.stderr.write(f"[{cell}/{total_cells}] {m['id']} / {t['id']}... ")
                sys.stderr.flush()
                res = run_one(m, t, env)
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
                    "model_id": m["id"],
                    "task_id": t["id"],
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
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
