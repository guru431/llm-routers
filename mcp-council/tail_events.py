"""Line-buffered tail for the council event-stream JSONL files.

Why this exists: on Git Bash / MSYS2 / Windows-cmd, `tail -F file | grep
--line-buffered ...` does not reliably propagate line-buffering to stdout —
the Claude Code `Monitor` tool ends up not receiving events in real-time.
This Python tail explicitly flushes after every line it writes, which keeps
the parent Monitor watching us happy.

Usage:
    python tail_events.py <path-to-jsonl> [--events <type1,type2,...>]

By default emits every line. With --events, only emits events whose top-level
"event" field matches one of the comma-separated types.

The tail exits naturally when it sees a `result_ready` event (with --until-done),
otherwise it polls until killed (Monitor timeout / TaskStop).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Poll interval — short because each council member can resolve in seconds
# during the busy stage1_member burst.
POLL_INTERVAL = 0.25


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the JSONL event log to tail")
    parser.add_argument(
        "--events",
        default=None,
        help="Comma-separated event types to keep (default: all)",
    )
    parser.add_argument(
        "--until-done",
        action="store_true",
        help="Exit after seeing a `result_ready` event",
    )
    args = parser.parse_args()

    wanted: set[str] | None = None
    if args.events:
        wanted = {e.strip() for e in args.events.split(",") if e.strip()}
        # With --until-done, the terminal event must survive the filter or the
        # exit check below never sees it and the tail polls forever.
        if args.until_done:
            wanted.add("result_ready")

    path = Path(args.path)
    # Wait for the file to appear — useful when this tail is started before
    # the first event has been written.
    while not path.exists():
        time.sleep(POLL_INTERVAL)

    # Read from the start: callers may attach mid-job and want backfill.
    fh = path.open("r", encoding="utf-8")
    try:
        while True:
            line = fh.readline()
            if not line:
                time.sleep(POLL_INTERVAL)
                continue
            line = line.rstrip("\r\n")
            if not line:
                continue
            # Parse JSON authoritatively rather than rely on whitespace-
            # sensitive substring matching (json.dumps adds a space by default
            # between key and value, so `"event":"phase"` substring matches
            # would silently miss every legitimately-encoded line).
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = obj.get("event")
            if wanted is not None and event_type not in wanted:
                continue
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            if args.until_done and event_type == "result_ready":
                return 0
    finally:
        fh.close()


if __name__ == "__main__":
    sys.exit(main())
