"""Benchmark diff / dashboard between two immutable runs (idea 18).

Compares a baseline run against a current run on quality (per-model & per-task),
latency (median TTFT / total), failure rate and output size (tok_out), flags
regressions by the shared thresholds, and exports Markdown (default), CSV or a
self-contained HTML dashboard (no external CDN).

Usage:
    python report_diff.py <baseline> <current>              # markdown to stdout
    python report_diff.py <baseline> <current> --md diff.md
    python report_diff.py <baseline> <current> --csv diff.csv
    python report_diff.py <baseline> <current> --html diff.html

<baseline>/<current> may each be a run id under results/runs/, a path to a run
dir, or a path to a manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
from pathlib import Path

import _store
from report import (
    LATENCY_REGRESSION_PCT,
    QUALITY_REGRESSION_DELTA,
    bootstrap_ci,
    load_run,
)

ROOT = Path(__file__).resolve().parent
MODELS_JSON = ROOT / "models.json"
TASKS_JSON = ROOT / "prompts" / "tasks.json"


def _median(xs):
    return statistics.median(xs) if xs else None


def compute_metrics(run_arg: str) -> dict:
    """Per-model aggregate metrics for one run (id/dir/manifest path)."""
    records_by_model, last_by_cell, judges, manifest, run_dir = load_run(run_arg)
    if run_dir is None and not records_by_model:
        raise SystemExit(f"no run found for: {run_arg!r}")
    repeats = int(manifest.get("repeats", 1)) if manifest else 1
    task_ids = [t["id"] for t in json.loads(TASKS_JSON.read_text(encoding="utf-8"))["tasks"]]

    per_model: dict[str, dict] = {}
    for mid, recs in records_by_model.items():
        ttfts, totals, toks = [], [], []
        errors = 0
        attempts = 0
        for r in recs:
            attempts += 1
            if r.get("error"):
                errors += 1
                continue
            if not (r.get("text") or "").strip():
                continue
            if r.get("ttft_s") is not None:
                ttfts.append(r["ttft_s"])
            if r.get("total_s") is not None:
                totals.append(r["total_s"])
            if r.get("tok_out") is not None:
                toks.append(r["tok_out"])
        scores = [j["score"] for (jm, jt), j in judges.items()
                  if jm == mid and j.get("score") is not None]
        per_task_q = {jt: j["score"] for (jm, jt), j in judges.items()
                      if jm == mid and j.get("score") is not None}
        per_model[mid] = {
            "quality": statistics.mean(scores) if scores else None,
            "quality_ci": bootstrap_ci(scores, repeats),
            "per_task_q": per_task_q,
            "total_p50": _median(totals),
            "total_ci": bootstrap_ci(totals, repeats),
            "ttft_p50": _median(ttfts),
            "tok_out_p50": _median(toks),
            "fail_rate": (errors / attempts) if attempts else None,
            "attempts": attempts,
        }
    return {
        "run_dir": run_dir,
        "run_id": manifest.get("run_id") if manifest else (run_dir.name if run_dir else "flat"),
        "manifest": manifest,
        "repeats": repeats,
        "per_model": per_model,
        "task_ids": task_ids,
    }


def _pct_delta(cur, base):
    if cur is None or base is None or base == 0:
        return None
    return (cur - base) / base * 100


def diff_runs(base: dict, cur: dict) -> list[dict]:
    """One diff row per model present in either run, with regression flags."""
    mids = sorted(set(base["per_model"]) | set(cur["per_model"]))
    rows = []
    for mid in mids:
        b = base["per_model"].get(mid, {})
        c = cur["per_model"].get(mid, {})
        dq = None
        if c.get("quality") is not None and b.get("quality") is not None:
            dq = c["quality"] - b["quality"]
        dlat = _pct_delta(c.get("total_p50"), b.get("total_p50"))
        dfail = None
        if c.get("fail_rate") is not None and b.get("fail_rate") is not None:
            dfail = c["fail_rate"] - b["fail_rate"]
        dtok = None
        if c.get("tok_out_p50") is not None and b.get("tok_out_p50") is not None:
            dtok = c["tok_out_p50"] - b["tok_out_p50"]
        regressed = bool(
            (dlat is not None and dlat > LATENCY_REGRESSION_PCT)
            or (dq is not None and dq < -QUALITY_REGRESSION_DELTA)
        )
        # Rank by regression severity (worst first): scaled latency growth + quality drop.
        severity = 0.0
        if dlat is not None and dlat > 0:
            severity += dlat / LATENCY_REGRESSION_PCT
        if dq is not None and dq < 0:
            severity += (-dq) / QUALITY_REGRESSION_DELTA
        rows.append({
            "model": mid,
            "base_q": b.get("quality"), "cur_q": c.get("quality"), "dq": dq,
            "cur_q_ci": c.get("quality_ci"),
            "base_total": b.get("total_p50"), "cur_total": c.get("total_p50"), "dlat_pct": dlat,
            "cur_total_ci": c.get("total_ci"),
            "base_ttft": b.get("ttft_p50"), "cur_ttft": c.get("ttft_p50"),
            "base_fail": b.get("fail_rate"), "cur_fail": c.get("fail_rate"), "dfail": dfail,
            "base_tok": b.get("tok_out_p50"), "cur_tok": c.get("tok_out_p50"), "dtok": dtok,
            "regressed": regressed,
            "severity": severity,
        })
    rows.sort(key=lambda r: (-r["severity"], r["model"]))
    return rows


# ---- formatting helpers ----

def _f(v, unit="", pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v:+.0f}%"
    return f"{v:.2f}{unit}"


def _ci_str(ci):
    return f"[{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""


def render_markdown(base: dict, cur: dict, rows: list[dict]) -> str:
    L = []
    L.append(f"# Benchmark diff — `{base['run_id']}` → `{cur['run_id']}`")
    L.append("")
    L.append(f"Пороги регрессии: latency +{LATENCY_REGRESSION_PCT}%, quality −{QUALITY_REGRESSION_DELTA}.")
    L.append("")
    regs = [r for r in rows if r["regressed"]]
    L.append("## ⚠️ Регрессии (worst first)")
    L.append("")
    if regs:
        for r in regs:
            bits = []
            if r["dlat_pct"] is not None and r["dlat_pct"] > LATENCY_REGRESSION_PCT:
                bits.append(f"latency {_f(r['base_total'],'s')} → {_f(r['cur_total'],'s')} ({_f(r['dlat_pct'],pct=True)})")
            if r["dq"] is not None and r["dq"] < -QUALITY_REGRESSION_DELTA:
                bits.append(f"Q {_f(r['base_q'])} → {_f(r['cur_q'])} ({r['dq']:+.2f})")
            L.append(f"- `{r['model']}` — " + "; ".join(bits))
    else:
        L.append("_Регрессий не обнаружено._")
    L.append("")
    L.append("## Полная таблица")
    L.append("")
    L.append("| Модель | Q base | Q cur | ΔQ | Total base | Total cur | ΔLat | Fail base | Fail cur | tok base | tok cur | Δtok |")
    L.append("|--------|--------|-------|----|-----------|-----------|------|-----------|----------|----------|---------|------|")
    for r in rows:
        flag = " ⚠️" if r["regressed"] else ""
        cur_q = _f(r["cur_q"])
        if r["cur_q_ci"]:
            cur_q += " " + _ci_str(r["cur_q_ci"])
        cur_total = _f(r["cur_total"], "s")
        if r["cur_total_ci"]:
            cur_total += " " + _ci_str(r["cur_total_ci"])
        dfail = "—" if r["dfail"] is None else f"{r['dfail']:+.2f}"
        dtok = "—" if r["dtok"] is None else f"{r['dtok']:+.0f}"
        dq = "—" if r["dq"] is None else f"{r['dq']:+.2f}"
        L.append(
            f"| `{r['model']}`{flag} | {_f(r['base_q'])} | {cur_q} | {dq} | "
            f"{_f(r['base_total'],'s')} | {cur_total} | {_f(r['dlat_pct'],pct=True)} | "
            f"{_f(r['base_fail'])} | {_f(r['cur_fail'])} | "
            f"{'—' if r['base_tok'] is None else f'{r['base_tok']:.0f}'} | "
            f"{'—' if r['cur_tok'] is None else f'{r['cur_tok']:.0f}'} | {dtok} |"
        )
    L.append("")
    return "\n".join(L)


CSV_FIELDS = [
    "model", "regressed",
    "base_q", "cur_q", "dq",
    "base_total", "cur_total", "dlat_pct",
    "base_ttft", "cur_ttft",
    "base_fail", "cur_fail", "dfail",
    "base_tok", "cur_tok", "dtok",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def write_html(path: Path, base: dict, cur: dict, rows: list[dict]) -> None:
    def td(v, unit="", pct=False):
        return f"<td>{html.escape(_f(v, unit, pct))}</td>"

    body = []
    body.append("<tr><th>Модель</th><th>Q base</th><th>Q cur</th><th>ΔQ</th>"
                "<th>Total base</th><th>Total cur</th><th>ΔLat</th>"
                "<th>Fail base</th><th>Fail cur</th><th>tok base</th><th>tok cur</th></tr>")
    for r in rows:
        cls = ' class="reg"' if r["regressed"] else ""
        dq = "—" if r["dq"] is None else f"{r['dq']:+.2f}"
        body.append(
            f"<tr{cls}><td>{html.escape(r['model'])}</td>"
            f"{td(r['base_q'])}{td(r['cur_q'])}<td>{html.escape(dq)}</td>"
            f"{td(r['base_total'],'s')}{td(r['cur_total'],'s')}{td(r['dlat_pct'],pct=True)}"
            f"{td(r['base_fail'])}{td(r['cur_fail'])}"
            f"<td>{'—' if r['base_tok'] is None else f'{r['base_tok']:.0f}'}</td>"
            f"<td>{'—' if r['cur_tok'] is None else f'{r['cur_tok']:.0f}'}</td></tr>"
        )
    n_reg = sum(1 for r in rows if r["regressed"])
    doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Benchmark diff {html.escape(base['run_id'])} to {html.escape(cur['run_id'])}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111; }}
 h1 {{ font-size: 1.2rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; font-family: monospace; }}
 tr.reg {{ background: #fde8e8; }}
 tr.reg td:first-child::after {{ content: " ⚠"; }}
 .meta {{ color: #555; font-size: 0.85rem; margin-bottom: 1rem; }}
</style></head><body>
<h1>Benchmark diff: {html.escape(base['run_id'])} → {html.escape(cur['run_id'])}</h1>
<p class="meta">Пороги: latency +{LATENCY_REGRESSION_PCT}%, quality −{QUALITY_REGRESSION_DELTA}. Регрессий: {n_reg}.</p>
<table>{''.join(body)}</table>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Diff two immutable benchmark runs.")
    ap.add_argument("baseline", help="baseline run id / dir / manifest.json")
    ap.add_argument("current", help="current run id / dir / manifest.json")
    ap.add_argument("--md", help="write markdown to this path (default: stdout)")
    ap.add_argument("--csv", help="write CSV to this path")
    ap.add_argument("--html", help="write self-contained HTML to this path")
    args = ap.parse_args()

    base = compute_metrics(args.baseline)
    cur = compute_metrics(args.current)
    rows = diff_runs(base, cur)

    md = render_markdown(base, cur, rows)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
        print(f"Markdown written: {args.md}")
    if args.csv:
        write_csv(Path(args.csv), rows)
        print(f"CSV written: {args.csv}")
    if args.html:
        write_html(Path(args.html), base, cur, rows)
        print(f"HTML written: {args.html}")
    if not (args.md or args.csv or args.html):
        print(md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
