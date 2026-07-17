#!/usr/bin/env python3
"""Private backlog lifecycle manifest + stale reminder.

FINDINGS.md and IDEAS.md are gitignored (private working notes); FINDINGS-archive.md
(and IDEAS-archive.md) are the tracked audit trail. Nothing tracked the LIFECYCLE
of the open items — which are still open, and which have gone stale. This script
scans those files, builds a status manifest, and flags any still-open item older
than the stale horizon (default 90 days) so it doesn't silently rot.

It writes `backlog/manifest.json` (gitignored — it echoes private titles) and
prints a summary + the stale list. Run it from a cron / MonthlyStratReview, or
locally: `python scripts/backlog_manifest.py [--stale-days N] [--json-only]`.

Exit code is 0 normally, or 2 (with --fail-on-stale) when stale items exist — so
a scheduled job can alert on them.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STALE_DAYS_DEFAULT = 90

# A backlog entry header: "## 2026-07-13 · Title [P2]".
_ENTRY = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s+·\s+(.+?)\s*(?:\[(P\d)\])?\s*$", re.MULTILINE)
_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)

# Files that make up the backlog, with whether their entries are "open by default".
_SOURCES = {
    "FINDINGS.md": {"open_default": "open"},
    "IDEAS.md": {"open_default": "proposed"},
    "FINDINGS-archive.md": {"open_default": "done"},
    "IDEAS-archive.md": {"open_default": "done"},
}

# Statuses that mean "still needs attention" (eligible to go stale).
_OPEN_STATUSES = {"open", "proposed", "accepted"}


def _parse_file(path: Path, open_default: str) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entries: list[dict] = []
    matches = list(_ENTRY.finditer(text))
    for i, m in enumerate(matches):
        block = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        sm = _STATUS.search(block)
        status = (sm.group(1).lower() if sm else open_default)
        entries.append({
            "file": path.name,
            "date": m.group(1),
            "title": m.group(2).strip(),
            "priority": m.group(3),
            "status": status,
        })
    return entries


def build_manifest(root: Path, stale_days: int) -> dict:
    today = _dt.date.today()
    horizon = today - _dt.timedelta(days=stale_days)
    entries: list[dict] = []
    for name, meta in _SOURCES.items():
        p = root / name
        if p.exists():
            entries.extend(_parse_file(p, meta["open_default"]))

    by_status: dict[str, int] = {}
    stale: list[dict] = []
    for e in entries:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        if e["status"] in _OPEN_STATUSES:
            try:
                d = _dt.date.fromisoformat(e["date"])
            except ValueError:
                continue
            if d < horizon:
                age = (today - d).days
                stale.append({**e, "age_days": age})
    stale.sort(key=lambda x: -x["age_days"])
    return {
        "generated": today.isoformat(),
        "stale_horizon_days": stale_days,
        "total_entries": len(entries),
        "by_status": by_status,
        "open_count": sum(by_status.get(s, 0) for s in _OPEN_STATUSES),
        "stale_count": len(stale),
        "stale": stale,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-days", type=int, default=STALE_DAYS_DEFAULT)
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--fail-on-stale", action="store_true",
                    help="exit 2 if any open item is stale (for scheduled alerts)")
    args = ap.parse_args()

    manifest = build_manifest(ROOT, args.stale_days)
    out_dir = ROOT / "backlog"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.json_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"Backlog manifest ({manifest['generated']}): "
              f"{manifest['total_entries']} entries, "
              f"{manifest['open_count']} open, {manifest['stale_count']} stale "
              f"(>{args.stale_days}d).")
        print("by status:", ", ".join(f"{k}={v}" for k, v in sorted(manifest["by_status"].items())))
        for s in manifest["stale"]:
            print(f"  STALE {s['age_days']}d · {s['file']} · {s['date']} · {s['title']}")

    if args.fail_on_stale and manifest["stale_count"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
