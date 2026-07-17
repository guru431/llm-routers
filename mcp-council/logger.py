"""JSONL logger for mcp-council. Writes metadata per-call + full dump for analysis."""

import json
import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
CALLS_DIR = LOG_DIR / "calls"


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
    tmp.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
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
) -> None:
    """Append one JSONL summary record to logs/council_YYYY-MM-DD.log.

    status = "ok" | "error: <message>".
    log_dump = relative path to the full dump (or None on hard failure before any dump).
    tool = which MCP tool produced this record ("council_ask" default; callers
        pass "model_ask" etc. so log analysis can split by tool).
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

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
