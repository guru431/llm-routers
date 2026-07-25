"""JSONL logger for mcp-council. Writes metadata per-call + full dump for analysis.

Everything written here is REDACTED first (retention.redact over the serialized
JSON): a dump carries the whole question, the context excerpts and every
provider body, so a key pasted into a question would otherwise sit on disk in
clear text for the whole retention window. Set COUNCIL_LOG_REDACT=0 to write raw
(debugging only — it defeats the point of the redaction promise in
retention.py's docstring).
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from retention import redact

LOG_DIR = Path(__file__).parent / "logs"
CALLS_DIR = LOG_DIR / "calls"

_REDACT_ENV = "COUNCIL_LOG_REDACT"


def _redaction_enabled() -> bool:
    return os.environ.get(_REDACT_ENV, "1").strip().lower() not in ("0", "false", "no")


def _redacted_json(payload: dict, **dumps_kwargs) -> str:
    """Serialize `payload` and mask credential-shaped tokens in the result.

    Redacting the SERIALIZED text (rather than walking the structure) covers
    secrets wherever they sit — nested provider bodies, answer prose, file
    excerpts — with one pass and no schema assumptions. JSON string escaping
    can't hide a token from the patterns: the credential shapes are ASCII and
    survive `json.dumps` unchanged."""
    text = json.dumps(payload, ensure_ascii=False, **dumps_kwargs)
    return redact(text) if _redaction_enabled() else text


def _new_call_id() -> str:
    # 48 bits of uuid4 entropy, not 16 bits of token_hex(2): a single async fan-out
    # can allocate several call_ids inside one second, and 16 bits collide at the
    # birthday bound (~256 ids) — an overwrite of another call's dump file. The
    # timestamp prefix stays for human-readable sort order.
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{uuid.uuid4().hex[:12]}"


def write_full_dump(call_id: str, dump: dict) -> Path:
    """Write the full per-call dump (question + stage1 answers + stage2 rankings) to disk.

    Used for offline analysis of council quality. Returns the relative path used in the
    summary JSONL record.
    """
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    path = CALLS_DIR / f"{call_id}.json"
    # Atomic write (tmp + replace) so a crash mid-write can't leave a truncated
    # dump that later fails to parse as JSON.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(_redacted_json(dump, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def log_call(
    *,
    call_id: str,
    members_total: int,
    members_ok_stage1: int,
    members_ok_stage2: int,
    prompt_size_bytes: int,
    total_latency_ms: int,
    status: str,
    log_dump: str | None,
    tool: str = "council_ask",
    web_search: dict | None = None,
) -> None:
    """Append one JSONL summary record to logs/council_YYYY-MM-DD.log.

    status = "ok" | "error: <message>".
    log_dump = relative path to the full dump (or None on hard failure before any dump).
    tool = which MCP tool produced this record ("council_ask" default; callers
        pass "model_ask" etc. so log analysis can split by tool).
    web_search = optional {calls, ok, blocked, cost_usd} aggregate. `model_ask`
        writes no full dump, so without this its Exa spend and DLP blocks were
        recorded nowhere at all.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"council_{datetime.now().strftime('%Y-%m-%d')}.log"

    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "call_id": call_id,
        "tool": tool,
        "members_total": members_total,
        "members_ok_stage1": members_ok_stage1,
        "members_ok_stage2": members_ok_stage2,
        "prompt_size_bytes": prompt_size_bytes,
        "total_latency_ms": total_latency_ms,
        "status": status,
        "log_dump": log_dump,
    }
    if web_search is not None:
        record["web_search"] = web_search

    with log_path.open("a", encoding="utf-8") as f:
        # `status` carries provider error text, which can echo a key-bearing URL.
        f.write(_redacted_json(record) + "\n")
