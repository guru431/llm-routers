"""MCP server: mcp-council.

Exposes two flavours of the council deliberation:

  * `council_ask` (sync) — blocks until the full council finishes (2-8 min).
  * `council_ask_async` + `council_status` / `council_result` / `council_cancel`
    / `council_list_jobs` — start in background, poll progress, fetch result.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from models import COUNCIL_DEFAULT, resolve_member, resolve_members, resolve_preset
from council import _aggregate as _aggregate_helper  # noqa: F401 — re-exported for tests
from council import run_council
from council import MAX_ROUNDS as COUNCIL_MAX_ROUNDS
from single_call import run_single
from logger import _new_call_id, log_call, write_full_dump
from sandbox import SandboxError, read_files_with_limit, resolve_and_validate
import sandbox
import circuit_breaker
from healthcheck import healthcheck_models
import event_log
import state as job_state

LOGS_DIR = Path(__file__).parent / "logs"

MAX_RESPONSE_TOKENS_HARD_CAP = 16384

mcp = FastMCP("mcp-council")


def _resolve_models_arg(
    models: list[str] | None, models_preset: str | None
) -> list[str] | None:
    """Resolve the effective model-id list from `models` / `models_preset`.

    At most one may be set. Returns list[str] | None (None → default council).
    Raises RuntimeError if both are set, UnknownPresetError on a bad name.
    """
    if models_preset is not None:
        if models is not None:
            raise RuntimeError("pass either models or models_preset, not both")
        return resolve_preset(models_preset)
    return models


def _build_files_section(files: list[tuple[Path, str]]) -> str:
    if not files:
        return ""
    parts = ["=== CONTEXT FILES ==="]
    for path, content in files:
        parts.append(f"=== FILE: {path} ===\n{content}\n")
    return "\n".join(parts)


def _clamp_tokens(n: int) -> int:
    return min(max(n, 1), MAX_RESPONSE_TOKENS_HARD_CAP)


def _format_analysis_lines(analysis: dict) -> list[str]:
    """Render the chairman's structured analysis. Blind spots first — the
    highest-value 'what did everyone miss' signal. Empty categories are skipped."""
    lines: list[str] = ["## Cross-cutting analysis", ""]
    bs = analysis.get("blind_spots") or []
    if bs:
        lines.append("**Blind spots (no member addressed):**")
        lines.extend(f"- {x}" for x in bs)
        lines.append("")
    contr = analysis.get("contradictions") or []
    if contr:
        lines.append("**Contradictions:**")
        for c in contr:
            if not isinstance(c, dict):
                lines.append(f"- {c}")
                continue
            topic = c.get("topic", "?")
            stances = "; ".join(
                f"{s.get('model', '?')}: {s.get('stance', '')}"
                for s in (c.get("stances") or []) if isinstance(s, dict)
            )
            lines.append(f"- {topic} — {stances}" if stances else f"- {topic}")
        lines.append("")
    cons = analysis.get("consensus") or []
    if cons:
        lines.append("**Consensus:**")
        lines.extend(f"- {x}" for x in cons)
        lines.append("")
    pc = analysis.get("partial_coverage") or []
    if pc:
        lines.append("**Partial coverage:**")
        for p in pc:
            if isinstance(p, dict):
                models = ", ".join(p.get("models") or [])
                lines.append(f"- [{models}] {p.get('point', '')}")
            else:
                lines.append(f"- {p}")
        lines.append("")
    ui = analysis.get("unique_insights") or []
    if ui:
        lines.append("**Unique insights:**")
        for u in ui:
            if isinstance(u, dict):
                lines.append(f"- {u.get('model', '?')}: {u.get('insight', '')}")
            else:
                lines.append(f"- {u}")
        lines.append("")
    return lines


def format_markdown(question: str, result: dict) -> str:
    """Render stage1+stage2+aggregate (and optional stage 3 synthesis) into a
    markdown brief for the chairman (Claude in-session, or whoever consumes it)."""
    stage1 = result["stage1"]
    stage2 = result["stage2"]
    aggregate = result["aggregate"]
    stage3 = result.get("stage3")
    notes = result["notes"]
    rounds_detail = result.get("rounds_detail") or []

    # Build a global pseudonym mapping for display: stable letter per member_id,
    # in stage1 order (so reading is consistent). Stage 2 rankers used their own
    # randomized mapping internally; for display we de-anonymize anyway.
    display_letter: dict[str, str] = {}
    letters = "ABCDEFGHIJ"
    for i, s in enumerate(stage1):
        display_letter[s["id"]] = letters[i]

    lines: list[str] = []
    lines.append("# Council deliberation")
    lines.append("")
    lines.append("## Question")
    lines.append(question)
    lines.append("")

    # Stage 3 synthesis goes first when present — it is the headline answer.
    if stage3 is not None:
        if stage3["status"] == "ok":
            chairman_label = f"{stage3['chairman_model']} ({stage3['chairman_id']})"
            latency_s = stage3["latency_ms"] / 1000.0
            lines.append(f"## Final Synthesis — by chairman {chairman_label}, {latency_s:.0f}s")
            lines.append("")
            lines.append(stage3["synthesis"])
            lines.append("")
            analysis = stage3.get("analysis")
            if analysis:
                lines.extend(_format_analysis_lines(analysis))
        else:
            lines.append(
                f"## Final Synthesis — FAILED (chairman {stage3['chairman_model']}: {stage3['error']})"
            )
            lines.append("")
            lines.append(
                "_(Synthesis attempt failed; fall back to stage 1 / stage 2 materials below.)_"
            )
            lines.append("")

    # Multi-round progression: the stage1/stage2/aggregate blocks below show only
    # the FINAL round, so for rounds>=2 the early critique rounds would be visible
    # nowhere. Render a compact per-round digest (who answered + aggregate order)
    # before the detailed final-round materials.
    if len(rounds_detail) > 1:
        lines.append("## Round-by-round progression (compact)")
        lines.append("")
        for ri, rd in enumerate(rounds_detail, 1):
            rd_stage1 = rd.get("stage1") or []
            model_by_id = {s["id"]: s["model"] for s in rd_stage1}
            revised = [s["model"] for s in rd_stage1 if s["status"] == "ok"]
            revised_str = ", ".join(revised) if revised else "(none)"
            lines.append(f"- Round {ri}: answered — {revised_str}")
            order = [
                f"{model_by_id.get(mid, mid)} {mean:.2f}"
                for mid, mean, _n in (rd.get("aggregate") or [])
            ]
            if order:
                lines.append(f"  - aggregate order: {'; '.join(order)}")
        lines.append("")

    lines.append("## Stage 1: Independent answers")
    lines.append("")
    for s in stage1:
        letter = display_letter[s["id"]]
        latency_s = s["latency_ms"] / 1000.0
        if s["status"] == "ok":
            lines.append(f"### Member {letter} ({s['model']}) — ok, {latency_s:.0f}s")
            lines.append("")
            lines.append(s["answer"])
            lines.append("")
        else:
            lines.append(f"### Member {letter} ({s['model']}) — error: {s['error']}")
            lines.append("")
            lines.append("_(no answer)_")
            lines.append("")

    lines.append("## Stage 2: Peer rankings (anonymized to each ranker, de-anonymized here)")
    lines.append("")
    if not stage2:
        lines.append("_(stage 2 skipped — not enough surviving members)_")
        lines.append("")
    else:
        for s in stage2:
            ranker_letter = display_letter.get(s["ranker_id"], "?")
            if s["status"] != "ok":
                lines.append(
                    f"### Member {ranker_letter} ({s['ranker_id']}) — error: {s['error']}"
                )
                lines.append("")
                continue
            conf = s.get("confidence")
            conf_str = f" (self-conf {conf}/10)" if conf is not None else ""
            lines.append(f"### Member {ranker_letter} ({s['ranker_id']}) ranked{conf_str}:")
            for r in sorted(s["rankings"], key=lambda x: -x["score"]):
                target_letter = display_letter.get(r["ranked_id"], "?")
                reasoning = r["reasoning"] or ""
                lines.append(
                    f"- {target_letter} ({r['ranked_id']}): {r['score']}/10 — \"{reasoning}\""
                )
            lines.append("")

    lines.append(
        "## Aggregate scores (confidence-weighted mean across rankers, excluding self)"
    )
    lines.append("")
    if not aggregate:
        lines.append("_(no aggregate — no successful rankings)_")
    else:
        for i, (mid, mean, n) in enumerate(aggregate, 1):
            letter = display_letter.get(mid, "?")
            # Find the model name from stage1
            model = next((s["model"] for s in stage1 if s["id"] == mid), mid)
            lines.append(f"{i}. Member {letter} ({model}): {mean:.2f} (n={n})")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    else:
        lines.append("- all members completed both stages successfully")
    lines.append("")

    summary = result.get("summary")
    usage = result.get("usage")
    if summary or usage:
        lines.append("## Verdict & usage")
        lines.append("")
        if summary:
            win = summary.get("winner_model") or "—"
            mean = summary.get("winner_mean_score")
            mean_str = f" (mean {mean})" if mean is not None else ""
            lines.append(f"- Winner: **{win}**{mean_str} · confidence: {summary.get('confidence')}")
            failed = summary.get("failed_models") or []
            if failed:
                lines.append(
                    "- Failed: " + ", ".join(
                        f"{f['model']} ({f['stage']}: {f.get('failure_reason', 'error')})"
                        for f in failed
                    )
                )
            dis = summary.get("top_disagreements") or []
            if dis:
                lines.append(
                    "- Top disagreement: " + ", ".join(
                        f"{d['model']} (spread {d['spread']})" for d in dis
                    )
                )
            lines.append(f"- Next: {summary.get('recommended_next_action')}")
        if usage:
            lines.append(
                f"- Usage: {usage.get('llm_calls')} LLM calls, "
                f"{usage.get('tokens_in')}→{usage.get('tokens_out')} tokens, "
                f"{usage.get('web_search_calls')} web searches, "
                f"{usage.get('retries')} retries"
            )
        lines.append("")

    lines.append("---")
    if stage3 is not None and stage3["status"] == "ok":
        lines.append(
            "Synthesis above was produced by the council chairman. Cross-check "
            "against stage 1 / stage 2 materials for blind spots before relaying."
        )
    else:
        lines.append("Now synthesize the final answer based on these materials.")
    return "\n".join(lines)


async def _do_council_ask_async(
    question: str,
    context_paths: list[str],
    max_response_tokens: int,
    synthesis: bool = False,
    rounds: int = 1,
    web_search: bool = False,
    models: list[str] | None = None,
    context_in_stage2: bool = True,
) -> str:
    """Validate paths, read files, run council, log, return markdown brief.

    Async core. Use this from MCP-tool (already inside a running event loop).
    For sync callers (tests, CLI) use the `_do_council_ask` wrapper below.
    """
    start = time.monotonic()
    call_id = _new_call_id()
    prompt_size = 0
    members_ok_stage1 = 0
    members_ok_stage2 = 0
    log_dump_rel: str | None = None

    # Resolve member subset before touching the sandbox. Validation errors here
    # are immediate — no half-started runs.
    if models is not None and len(set(models)) < 2:
        raise RuntimeError(
            "council_ask requires at least 2 distinct models; "
            "use model_ask for single-model"
        )
    # Validate rounds with the same RuntimeError the async tool raises. Without
    # this a bad rounds reaches run_council as a ValueError, which is not caught
    # by the except (SandboxError, RuntimeError) below (no audit log, worse error
    # than council_ask_async).
    if not (1 <= rounds <= COUNCIL_MAX_ROUNDS):
        raise RuntimeError(f"rounds must be in [1, {COUNCIL_MAX_ROUNDS}], got {rounds}")
    members = resolve_members(models)

    try:
        max_tokens = _clamp_tokens(max_response_tokens)
        files_section: str | None = None
        if context_paths:
            validated = resolve_and_validate(context_paths)
            files = read_files_with_limit(validated)
            files_section = _build_files_section(files)
        prompt_for_size = (files_section or "") + question
        prompt_size = len(prompt_for_size.encode("utf-8"))

        result = await run_council(
            question=question,
            files_section=files_section,
            max_response_tokens=max_tokens,
            synthesis=synthesis,
            rounds=rounds,
            web_search=web_search,
            members=members,
            context_in_stage2=context_in_stage2,
        )
    except SandboxError as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_call(
            call_id=call_id,
            members_total=len(members),
            members_ok_stage1=0,
            members_ok_stage2=0,
            prompt_size_bytes=prompt_size,
            total_latency_ms=latency_ms,
            status=f"error: sandbox — {e}",
            log_dump=None,
        )
        raise RuntimeError(f"sandbox: {e}") from e
    except RuntimeError as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_call(
            call_id=call_id,
            members_total=len(members),
            members_ok_stage1=0,
            members_ok_stage2=0,
            prompt_size_bytes=prompt_size,
            total_latency_ms=latency_ms,
            status=f"error: {e}",
            log_dump=None,
        )
        raise

    members_ok_stage1 = sum(1 for s in result["stage1"] if s["status"] == "ok")
    members_ok_stage2 = sum(1 for s in result["stage2"] if s["status"] == "ok")

    dump = {
        "call_id": call_id,
        "question": question,
        "context_paths": list(context_paths),
        "stage1": result["stage1"],
        "stage2": result["stage2"],
        "aggregate": result["aggregate"],
        "rounds_detail": result.get("rounds_detail"),
        "stage3": result.get("stage3"),
        "notes": result["notes"],
        "usage": result.get("usage"),
        "summary": result.get("summary"),
    }
    dump_path = write_full_dump(call_id, dump)
    log_dump_rel = str(dump_path.relative_to(Path(__file__).parent))

    latency_ms = int((time.monotonic() - start) * 1000)
    log_call(
        call_id=call_id,
        members_total=len(members),
        members_ok_stage1=members_ok_stage1,
        members_ok_stage2=members_ok_stage2,
        prompt_size_bytes=prompt_size,
        total_latency_ms=latency_ms,
        status="ok",
        log_dump=log_dump_rel,
    )

    return format_markdown(question, result)


def _do_council_ask(
    question: str,
    context_paths: list[str],
    max_response_tokens: int,
    synthesis: bool = False,
    rounds: int = 1,
    web_search: bool = False,
    models: list[str] | None = None,
    context_in_stage2: bool = True,
) -> str:
    """Sync wrapper around `_do_council_ask_async` for tests and CLI use.

    Do NOT call from within a running asyncio event loop (e.g. MCP tool handler);
    use `_do_council_ask_async` directly with `await` there.
    """
    return asyncio.run(
        _do_council_ask_async(
            question, context_paths, max_response_tokens, synthesis, rounds,
            web_search, models, context_in_stage2,
        )
    )


@mcp.tool()
async def council_ask(
    question: str,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 8192,
    synthesis: bool = False,
    rounds: int = 1,
    web_search: bool = False,
    models: list[str] | None = None,
    models_preset: str | None = None,
    context_in_stage2: bool = True,
) -> str:
    """Спросить council по методу Karpathy: independent answers → anonymized
    peer-ranking → optional stage 3 synthesis. Synthesis off by default —
    пусть Claude в сессии делает финальный синтез с полным контекстом.

    По умолчанию совет = 7 моделей (GLM, Kimi, DeepSeek-Pro, Qwen, MiniMax,
    Gemini, Codex). Через `models=[...]` можно вызвать подмножество — минимум 2
    модели. Для одной модели используй `model_ask`.

    Используй когда: архитектурное решение, спорный технический вопрос, важный
    code review, разбор сложного бага. НЕ используй для рутины (быстрых
    вопросов, шаблонной генерации) — это дорого и медленно (~2-4 минуты).

    Parameters:
      models — list[str] | None. Список model_id из CATALOG (например
        ["glm","kimi","deepseek-pro"]). None → все 7 default-членов. ≥2.
      models_preset — str | None. Удобная альтернатива ручному `models`:
        "best" (все 7), "balanced" (3 модели), "cheap" (2 дешёвых). Нельзя
        задавать вместе с `models`.
      context_paths — опциональные файлы, прокидываются всем участникам (sandbox).
      synthesis — если True, добавляется stage 3 (auto-synthesis by chairman).
        Chairman дополнительно отдаёт структурный analysis (consensus /
        contradictions / partial_coverage / unique_insights / blind_spots),
        который попадает в summary.analysis (machine-readable). Если False,
        возвращаются только материалы stage1+stage2.
      rounds — 1..3. 2+ = multi-round debate с критикой между раундами.
      web_search — если True, каждая модель в stage 1 получает tool
        `web_search(query)` через Exa.ai (per-model exploration, не shared
        context). Stage 2 без поиска; при synthesis=True chairman тоже получает
        web_search для фактчека спорных claim'ов. Добавляет 30-90s к каждому
        stage 1 вызову и расход на Exa API.

    Note: блокирующий вызов; для long-running неблокирующего паттерна
    используй council_ask_async / council_status / council_result.
    """
    models = _resolve_models_arg(models, models_preset)
    return await _do_council_ask_async(
        question, context_paths or [], max_response_tokens, synthesis, rounds,
        web_search, models, context_in_stage2,
    )


# ---------------------------------------------------------------------------
# Async-job pattern: council_ask_async + council_status/result/cancel/list_jobs
# ---------------------------------------------------------------------------


def _make_progress_callback(state: job_state.JobState):
    """Return an on_progress function bound to `state` for run_council.

    Side effects: (1) updates in-memory JobState for `council_status` polling,
    (2) appends each event as JSONL line to logs/events/<job_id>.jsonl for
    Monitor-friendly real-time consumption.
    """
    writer = event_log.open_writer(state.job_id, LOGS_DIR)

    def progress(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "phase":
            phase = payload.get("phase")
            if phase:
                job_state.mark_phase(state, phase)
        elif event_type == "stage1_member":
            job_state.update_member_stage1(
                state,
                id=payload["id"],
                model=payload["model"],
                status=payload["status"],
                error=payload.get("error"),
                latency_ms=payload.get("latency_ms"),
            )
        elif event_type == "stage2_ranker":
            job_state.update_member_stage2(
                state,
                id=payload["id"],
                model=payload["model"],
                status=payload["status"],
                error=payload.get("error"),
                latency_ms=payload.get("latency_ms"),
            )
        elif event_type == "stage3":
            job_state.update_stage3(
                state,
                id=payload["id"],
                model=payload["model"],
                status=payload["status"],
                error=payload.get("error"),
                latency_ms=payload.get("latency_ms"),
            )
        # tool_call events have no state mirror — they're purely observability.
        # Mirror everything to the event log regardless of type so consumers
        # see the full timeline.
        try:
            writer.write(event_type, payload)
        except Exception:
            # Event log is best-effort: failure here must not break the run.
            pass

    return progress


async def _run_job(
    state: job_state.JobState,
    question: str,
    context_paths: list[str],
    max_response_tokens: int,
    synthesis: bool,
    rounds: int,
    web_search: bool,
    members: list[dict],
    context_in_stage2: bool = True,
) -> None:
    """Background entry point — runs the council and stores the result on state."""
    start = time.monotonic()
    call_id = _new_call_id()
    prompt_size = 0
    log_dump_rel: str | None = None
    on_progress = _make_progress_callback(state)
    try:
        try:
            max_tokens = _clamp_tokens(max_response_tokens)
            files_section: str | None = None
            if context_paths:
                validated = resolve_and_validate(context_paths)
                files = read_files_with_limit(validated)
                files_section = _build_files_section(files)
            prompt_for_size = (files_section or "") + question
            prompt_size = len(prompt_for_size.encode("utf-8"))

            result = await run_council(
                question=question,
                files_section=files_section,
                max_response_tokens=max_tokens,
                synthesis=synthesis,
                rounds=rounds,
                web_search=web_search,
                members=members,
                on_progress=on_progress,
                context_in_stage2=context_in_stage2,
            )
        except asyncio.CancelledError:
            # cancel_job intentionally leaves phase alone now (it used to set
            # phase='cancelled' eagerly and could overwrite an in-flight
            # mark_phase('done')). We own the transition here.
            job_state.mark_phase(state, "cancelled")
            on_progress("result_ready", {"status": "cancelled"})
            raise
        except SandboxError as e:
            state.error = f"sandbox: {e}"
            job_state.mark_phase(state, "error")
            on_progress("result_ready", {"status": "error", "error": state.error})
            latency_ms = int((time.monotonic() - start) * 1000)
            log_call(
                call_id=call_id, members_total=len(members),
                members_ok_stage1=0, members_ok_stage2=0,
                prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
                status=f"error: sandbox — {e}", log_dump=None,
            )
            return
        except RuntimeError as e:
            state.error = str(e)
            job_state.mark_phase(state, "error")
            on_progress("result_ready", {"status": "error", "error": str(e)})
            latency_ms = int((time.monotonic() - start) * 1000)
            log_call(
                call_id=call_id, members_total=len(members),
                members_ok_stage1=0, members_ok_stage2=0,
                prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
                status=f"error: {e}", log_dump=None,
            )
            return

        members_ok_stage1 = sum(1 for s in result["stage1"] if s["status"] == "ok")
        members_ok_stage2 = sum(1 for s in result["stage2"] if s["status"] == "ok")
        dump = {
            "call_id": call_id, "question": question, "context_paths": list(context_paths),
            "stage1": result["stage1"], "stage2": result["stage2"],
            "aggregate": result["aggregate"], "rounds_detail": result.get("rounds_detail"),
            "stage3": result.get("stage3"),
            "notes": result["notes"],
            "usage": result.get("usage"), "summary": result.get("summary"),
        }
        dump_path = write_full_dump(call_id, dump)
        log_dump_rel = str(dump_path.relative_to(Path(__file__).parent))
        state.dump_path = log_dump_rel
        state.usage = result.get("usage")
        state.summary = result.get("summary")
        state.result_markdown = format_markdown(question, result)
        job_state.mark_phase(state, "done")
        # Emit a terminal event with a stable string so Monitor consumers can
        # match on `"event": "result_ready"` and know the run is consumable.
        on_progress("result_ready", {
            "status": "ok",
            "members_ok_stage1": members_ok_stage1,
            "members_ok_stage2": members_ok_stage2,
            "dump_path": log_dump_rel,
        })

        latency_ms = int((time.monotonic() - start) * 1000)
        log_call(
            call_id=call_id, members_total=len(members),
            members_ok_stage1=members_ok_stage1, members_ok_stage2=members_ok_stage2,
            prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
            status="ok", log_dump=log_dump_rel,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Catch-all so an unexpected error never leaves the job stuck in a
        # non-terminal phase forever, holding one of the MAX_ACTIVE_JOBS slots.
        # Reachable via e.g. a context file deleted between validation and read
        # (FileNotFoundError), or write_full_dump hitting OSError after the
        # council already succeeded. CancelledError is re-raised above so cancel
        # semantics (handled inside the inner try) are preserved.
        if state.phase not in job_state.TERMINAL_PHASES:
            state.error = state.error or f"{type(e).__name__}: {e}"
            job_state.mark_phase(state, "error")
            try:
                on_progress("result_ready", {"status": "error", "error": state.error})
            except Exception:
                pass
            latency_ms = int((time.monotonic() - start) * 1000)
            log_call(
                call_id=call_id, members_total=len(members),
                members_ok_stage1=0, members_ok_stage2=0,
                prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
                status=f"error: {type(e).__name__} — {e}", log_dump=None,
            )
    finally:
        # Always close the event log so the tail -F consumer sees EOF cleanly.
        event_log.close_writer(state.job_id)


@mcp.tool()
async def council_ask_async(
    question: str,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 8192,
    synthesis: bool = False,
    rounds: int = 1,
    web_search: bool = False,
    models: list[str] | None = None,
    models_preset: str | None = None,
    context_in_stage2: bool = True,
) -> dict:
    """Start a council deliberation in the background and return a job_id
    immediately (within ~50ms). Poll progress with `council_status(job_id)`
    and fetch the final markdown with `council_result(job_id)` once
    `phase == "done"`.

    Use this when the caller (you, Claude in-session) wants to remain
    responsive to the user while the 2-8 minute deliberation runs.

    `rounds` — 1 (default) for single-pass Karpathy, 2+ for multi-round debate
    where surviving members rewrite their answers after seeing peer critique.
    Each extra round adds 2-8 minutes of wall-time.

    `models` — list[str] | None. Subset of CATALOG ids (≥2). None → default 7.
    `models_preset` — str | None. "best" | "balanced" | "cheap" instead of a
        hand-listed `models` (mutually exclusive with it).
    """
    # Validate + resolve BEFORE creating job state, so bad inputs fail fast.
    models = _resolve_models_arg(models, models_preset)
    if models is not None and len(set(models)) < 2:
        raise RuntimeError(
            "council_ask_async requires at least 2 distinct models; "
            "use model_ask for single-model"
        )
    # Validate rounds BEFORE create_job. run_council raises ValueError on a bad
    # rounds value; if that fired inside the background task it would leave the
    # job stuck non-terminal (the catch-all now also guards this, but failing
    # fast here gives the caller a clean error instead of a dead job_id).
    if not (1 <= rounds <= COUNCIL_MAX_ROUNDS):
        raise RuntimeError(f"rounds must be in [1, {COUNCIL_MAX_ROUNDS}], got {rounds}")
    members = resolve_members(models)

    state = await job_state.create_job(
        question_preview=question,
        synthesis=synthesis,
        rounds=rounds,
    )
    task = asyncio.create_task(
        _run_job(
            state, question, context_paths or [], max_response_tokens,
            synthesis, rounds, web_search, members, context_in_stage2,
        )
    )
    job_state.attach_task(state, task)
    return {
        "job_id": state.job_id,
        "phase": state.phase,
        "expected_members": [m["id"] for m in members],
        "synthesis_requested": synthesis,
        "rounds_requested": rounds,
        "web_search_enabled": web_search,
        "event_log": str(
            Path(__file__).parent / "logs" / "events" / f"{state.job_id}.jsonl"
        ),
        "hint": (
            "Poll council_status(job_id). When phase=='done', call "
            "council_result(job_id). For real-time monitoring tail -F the "
            "event_log file (JSONL, one event per line)."
        ),
    }


@mcp.tool()
async def council_status(job_id: str) -> dict:
    """Return current snapshot of a job: phase, per-member progress, elapsed
    time. Does NOT block — safe to poll often. Returns {error: ...} if the
    job_id is unknown.
    """
    state = await job_state.get_job(job_id)
    if state is None:
        return {"error": f"unknown job_id: {job_id}"}
    snap = job_state.snapshot(state)
    # Surface the global active-jobs budget so callers can see headroom before
    # firing more council_ask_async calls (cap enforced in state.create_job).
    snap["active_jobs"] = await job_state.active_job_count()
    snap["max_active_jobs"] = job_state.MAX_ACTIVE_JOBS
    return snap


@mcp.tool()
async def council_result(job_id: str) -> dict:
    """Fetch the final markdown for a completed job. Returns the markdown
    inline plus a `dump_path` (relative to the mcp-council/ folder) where the
    full JSON dump lives. If the job is not yet done, returns the current
    phase and asks the caller to poll again.
    """
    state = await job_state.get_job(job_id)
    if state is None:
        return {"error": f"unknown job_id: {job_id}"}
    if state.phase != "done":
        interrupted = state.phase == "interrupted"
        return {
            "ready": False,
            "phase": state.phase,
            "elapsed_ms": (
                int((time.time() - state.started_at) * 1000)
                if state.started_at else 0
            ),
            "usage": state.usage,
            "summary": state.summary,
            "hint": (
                "Job was interrupted by a server restart and is not resumable. "
                "Call council_status(job_id) for the partial per-stage progress, "
                "then re-run council_ask_async if you need a complete result."
                if interrupted
                else "Call council_status(job_id) for live progress, retry later."
            ),
        }
    return {
        "ready": True,
        "phase": state.phase,
        "result_markdown": state.result_markdown,
        "dump_path": state.dump_path,
        "usage": state.usage,
        "summary": state.summary,
    }


@mcp.tool()
async def council_cancel(job_id: str) -> dict:
    """Cancel a running job. No-op if the job is already done/errored."""
    ok = await job_state.cancel_job(job_id)
    return {"cancelled": ok}


@mcp.tool()
async def council_list_jobs(limit: int = 20) -> list[dict]:
    """List most-recent jobs (default last 20) — useful when the caller forgot
    the job_id from a previous turn."""
    jobs = await job_state.list_jobs(limit=limit)
    return [job_state.snapshot(j) for j in jobs]


@mcp.tool()
async def model_healthcheck(models: list[str] | None = None) -> dict:
    """Ping every CATALOG model (or a subset) with a trivial prompt and report
    per-model health: key present, HTTP status class, latency, empty-response.

    Use this BEFORE a council run when something looks off, or to debug a member
    that keeps erroring. Each model gets one cheap call (~"pong"); disabled
    models are reported as status="disabled" (not called). `status` per model is
    one of: ok | disabled | no_key | auth | insufficient_balance | rate_limited
    | timeout | empty_response | network | error.

    Also surfaces whether the COUNCIL_CONTEXT_ROOTS guardrail is configured.
    """
    rows = await healthcheck_models(models)
    ok = sum(1 for r in rows if r["ok"])
    return {
        "checked": len(rows),
        "ok": ok,
        "failed": len(rows) - ok,
        "context_roots_configured": sandbox.context_roots_configured(),
        "circuit_breakers": circuit_breaker.snapshot(),
        "models": rows,
    }


# ---------------------------------------------------------------------------
# model_ask: one-shot single-model call (replaces deepseek_read/draft + minimax_*)
# ---------------------------------------------------------------------------


def _build_files_sections(
    context_files: list[tuple[Path, str]],
    example_files: list[tuple[Path, str]],
) -> str:
    """Build CONTEXT FILES + STYLE EXAMPLES sections. Empty sections are skipped."""
    parts: list[str] = []
    if context_files:
        ctx = ["=== CONTEXT FILES ==="]
        for path, content in context_files:
            ctx.append(f"=== FILE: {path} ===\n{content}\n")
        parts.append("\n".join(ctx))
    if example_files:
        ex = ["=== STYLE EXAMPLES ==="]
        for path, content in example_files:
            ex.append(f"=== FILE: {path} ===\n{content}\n")
        parts.append("\n".join(ex))
    return "\n\n".join(parts)


@mcp.tool()
async def model_ask(
    model_id: str,
    prompt: str,
    context_paths: list[str] | None = None,
    example_paths: list[str] | None = None,
    max_response_tokens: int = 4096,
    web_search: bool = False,
) -> str:
    """Дёрнуть ОДНУ конкретную модель из CATALOG напрямую (без council deliberation).

    Заменяет deepseek_read/draft и minimax_read/draft из старых пакетов.

    Используй когда: тяжёлая суммаризация (большие логи, JSONL-транскрипты,
    объёмные конфиги), QA по файлам, шаблонная генерация черновиков кода/доков,
    переводы — задачи, не требующие сложного рассуждения или совещания.
    НЕ используй для архитектурных решений (для них — council_ask).

    Parameters:
      model_id — id из models.CATALOG. Доступные: glm, kimi, deepseek-pro, qwen,
        minimax, gemini, deepseek-flash. (minimax-direct — disabled, billing off.)
      prompt — собственно вопрос / задача.
      context_paths — sandbox-файлы, прокидываются как CONTEXT FILES.
      example_paths — sandbox-файлы стиля, прокидываются как STYLE EXAMPLES.
      max_response_tokens — default 4096, hard cap 16384.
      web_search — если True, даёт модели Exa-based web_search(query) tool.
    """
    start = time.monotonic()
    call_id = _new_call_id()
    prompt_size = 0

    try:
        cfg = resolve_member(model_id)
        max_tokens = _clamp_tokens(max_response_tokens)

        # Enforce the 50-file / 500 KB sandbox limit across context + example
        # COMBINED, not per-list (reading each list independently doubled the
        # documented budget). resolve_and_validate caps count per call, so add
        # an explicit combined count check, then read both lists through one
        # byte-budgeted pass and split the result back by count.
        validated_ctx = resolve_and_validate(context_paths) if context_paths else []
        validated_ex = resolve_and_validate(example_paths) if example_paths else []
        if len(validated_ctx) + len(validated_ex) > sandbox.MAX_FILE_COUNT:
            raise SandboxError(
                f"file count limit exceeded: "
                f"{len(validated_ctx) + len(validated_ex)} > {sandbox.MAX_FILE_COUNT}"
            )
        all_files = read_files_with_limit(validated_ctx + validated_ex)
        ctx_files = all_files[: len(validated_ctx)]
        ex_files = all_files[len(validated_ctx):]

        files_section = _build_files_sections(ctx_files, ex_files)
        full_prompt_parts: list[str] = []
        if files_section:
            full_prompt_parts.append(files_section)
        full_prompt_parts.append(f"=== TASK ===\n{prompt}")
        full_prompt = "\n\n".join(full_prompt_parts)
        prompt_size = len(full_prompt.encode("utf-8"))

        answer = await run_single(
            cfg,
            prompt=full_prompt,
            max_tokens=max_tokens,
            web_search=web_search,
        )
    except SandboxError as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_call(
            call_id=call_id, members_total=1,
            members_ok_stage1=0, members_ok_stage2=0,
            prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
            status=f"error: sandbox — {e}", log_dump=None, tool="model_ask",
        )
        raise RuntimeError(f"sandbox: {e}") from e
    except RuntimeError as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_call(
            call_id=call_id, members_total=1,
            members_ok_stage1=0, members_ok_stage2=0,
            prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
            status=f"error: {e}", log_dump=None, tool="model_ask",
        )
        raise

    latency_ms = int((time.monotonic() - start) * 1000)
    log_call(
        call_id=call_id, members_total=1,
        members_ok_stage1=1, members_ok_stage2=0,
        prompt_size_bytes=prompt_size, total_latency_ms=latency_ms,
        status="ok", log_dump=None, tool="model_ask",
    )
    return answer


# ---------------------------------------------------------------------------
# Dialogue tools — model_debate / model_panel / model_socratic
# ---------------------------------------------------------------------------

from dialogue import state as dialogue_state
from dialogue.debate import run_debate
from dialogue.panel import run_panel
from dialogue.socratic import run_socratic
from dialogue.render import format_dialogue_markdown
from dialogue.engine import write_dump

DIALOGUE_DUMP_DIR = Path(__file__).parent / "logs" / "dialogues"
DIALOGUE_ROUNDS_MAX = 20
DIALOGUE_ROUNDS_MIN = 1
DEFAULT_DEBATE_PARTICIPANTS = ["glm", "kimi", "codex"]
DEFAULT_PANEL_PARTICIPANTS = ["glm", "kimi", "deepseek-pro", "qwen", "minimax", "gemini", "codex"]
DEFAULT_SOCRATIC_QUESTIONER = "deepseek-pro"
DEFAULT_SOCRATIC_RESPONDENT = "glm"
DEFAULT_MODERATOR = "deepseek-flash"
DEFAULT_PANEL_MIN_PARTICIPANTS = 4
DEFAULT_DEBATE_MIN_PARTICIPANTS = 2


def _validate_rounds(rounds: int) -> int:
    if not (DIALOGUE_ROUNDS_MIN <= rounds <= DIALOGUE_ROUNDS_MAX):
        raise RuntimeError(
            f"rounds must be in [{DIALOGUE_ROUNDS_MIN}, {DIALOGUE_ROUNDS_MAX}], got {rounds}"
        )
    return rounds


def _resolve_engine_cfg(model_id: str) -> dict:
    """Resolve a model id to the engine-cfg shape (id, model, base_url, env_key,
    plus optional extra/min_max_tokens)."""
    return resolve_member(model_id)


async def _build_files_section_or_none(context_paths: list[str] | None) -> str | None:
    if not context_paths:
        return None
    validated = resolve_and_validate(context_paths)
    files = read_files_with_limit(validated)
    return _build_files_section(files) or None


async def _dialogue_runner_guard(state, runner_coro_factory) -> None:
    """Run a dialogue runner, owning the terminal-phase transitions. Shared by
    the 3 starter tools and dialogue_continue so cancel/error handling and the
    error-path dump live in exactly one place (they used to be copy-pasted)."""
    try:
        await runner_coro_factory(state)
    except asyncio.CancelledError:
        # cancel_session no longer flips phase eagerly — we own the transition
        # here so a near-done task isn't overwritten. Guard against a cancel that
        # lands AFTER the runner already reached a terminal phase: debate/socratic/
        # panel call mark_phase("done") BEFORE the final `await write_dump`, so a
        # cancel during that await must not clobber the completed run's phase
        # (which would also wrongly block dialogue_continue). Mirrors the council
        # guard's terminal-phase check.
        if state.phase not in dialogue_state.TERMINAL_PHASES:
            dialogue_state.mark_phase(state, "cancelled")
        raise
    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        dialogue_state.mark_phase(state, "error")
        try:
            state.dump_path = str(
                await asyncio.to_thread(write_dump, state, base_dir=DIALOGUE_DUMP_DIR)
            )
        except Exception:
            pass


async def _start_dialogue_session(
    *,
    mode: str,
    question_preview: str,
    total_rounds: int,
    runner_coro_factory,
    participants: list[dict],
    moderator: dict | None,
    web_search: bool = False,
    max_tokens: int = 4096,
    context_paths: list[str] | None = None,
) -> dict:
    """Common shape for the 3 mode tools: create session, kick off the background
    task, return the immediate response dict."""
    state = await dialogue_state.create_session(
        mode=mode, question_preview=question_preview, total_rounds=total_rounds,
        web_search=web_search, max_tokens=max_tokens, context_paths=context_paths,
    )
    state.participants = participants
    state.moderator = moderator

    task = asyncio.create_task(_dialogue_runner_guard(state, runner_coro_factory))
    dialogue_state.attach_task(state, task)

    return {
        "session_id": state.session_id,
        "mode": state.mode,
        "phase": state.phase,
        "total_rounds": state.total_rounds,
        "participants": list(state.participants),
        "moderator": state.moderator,
        "hint": (
            "Poll dialogue_status(session_id). When phase=='done', call "
            "dialogue_result(session_id). Full transcript ends up in "
            f"logs/dialogues/{state.session_id}.json."
        ),
    }


@mcp.tool()
async def model_debate(
    question: str,
    participants: list[str] | None = None,
    moderator: str | None = None,
    rounds: int = 5,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 4096,
    web_search: bool = False,
) -> dict:
    """Запустить debate из 2+ моделей с противоположными позициями.

    Модератор автоматически разбивает question на N противоположных тезисов и
    назначает их участникам в порядке declared. Каждый участник жёстко защищает
    свою позицию N раундов с critique-phase. В финале модератор пишет summary.

    Возвращает session_id (~50ms); для прогресса — dialogue_status, для
    результата — dialogue_result.

    Parameters:
      participants — list[str] | None. Минимум 2 distinct id из CATALOG.
        Default: ["glm", "kimi", "codex"].
      moderator — str | None. Default: "deepseek-flash" (дешёвая модель для
        разбиения вопроса и summary).
      rounds — 1..20. Default 5.
    """
    rounds = _validate_rounds(rounds)
    ids = participants or DEFAULT_DEBATE_PARTICIPANTS
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"model_debate participants must be distinct, got duplicates: {ids}")
    if len(set(ids)) < DEFAULT_DEBATE_MIN_PARTICIPANTS:
        raise RuntimeError(
            f"model_debate requires at least {DEFAULT_DEBATE_MIN_PARTICIPANTS} distinct participants, got {ids}"
        )
    part_cfgs = [_resolve_engine_cfg(i) for i in ids]
    mod_cfg = _resolve_engine_cfg(moderator or DEFAULT_MODERATOR)
    max_tokens = _clamp_tokens(max_response_tokens)
    files_section = await _build_files_section_or_none(context_paths)

    participants_seed = [
        {"id": c["id"], "model": c["model"], "position": None, "role": None}
        for c in part_cfgs
    ]
    moderator_seed = {"id": mod_cfg["id"], "model": mod_cfg["model"]}

    async def runner(state):
        await run_debate(
            state=state, question=question, participant_cfgs=part_cfgs,
            moderator_cfg=mod_cfg, rounds=rounds, max_tokens=max_tokens,
            web_search=web_search, files_section=files_section,
        )

    return await _start_dialogue_session(
        mode="debate", question_preview=question, total_rounds=rounds,
        runner_coro_factory=runner,
        participants=participants_seed, moderator=moderator_seed,
        web_search=web_search, max_tokens=max_tokens,
        context_paths=context_paths,
    )


@mcp.tool()
async def model_panel(
    question: str,
    participants: list[str] | None = None,
    roles: list[str] | None = None,
    diversity_monitor: bool = True,
    diversity_threshold: int = 7,
    devils_advocate_rotation: bool = True,
    monitor_model: str | None = None,
    rounds: int = 5,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 4096,
    web_search: bool = False,
) -> dict:
    """Запустить panel discussion: 4+ моделей свободно обсуждают тему.

    Anti-convergence: devil's advocate ротация (каждый раунд один участник
    обязан возражать) + diversity monitor (cheap LLM-вызов проверяет similarity,
    при score > threshold re-prompt согласившимся).

    Default participants = DEFAULT_PANEL_PARTICIPANTS (7 моделей, вкл. codex). Min 4 distinct.
    """
    rounds = _validate_rounds(rounds)
    ids = participants or DEFAULT_PANEL_PARTICIPANTS
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"model_panel participants must be distinct, got duplicates: {ids}")
    if len(set(ids)) < DEFAULT_PANEL_MIN_PARTICIPANTS:
        raise RuntimeError(
            f"model_panel requires at least {DEFAULT_PANEL_MIN_PARTICIPANTS} distinct participants, got {ids}"
        )
    if roles is not None and len(roles) != len(ids):
        raise RuntimeError(
            f"roles must match participants length; got {len(roles)} roles for {len(ids)} participants"
        )
    part_cfgs = [_resolve_engine_cfg(i) for i in ids]
    mon_cfg = _resolve_engine_cfg(monitor_model or DEFAULT_MODERATOR)
    max_tokens = _clamp_tokens(max_response_tokens)
    files_section = await _build_files_section_or_none(context_paths)

    participants_seed = [
        {"id": c["id"], "model": c["model"], "position": None,
         "role": (roles[i] if roles else None)}
        for i, c in enumerate(part_cfgs)
    ]
    moderator_seed = {"id": mon_cfg["id"], "model": mon_cfg["model"]}

    async def runner(state):
        state.diversity_monitor = diversity_monitor
        state.diversity_threshold = diversity_threshold
        state.devils_advocate_rotation = devils_advocate_rotation
        await run_panel(
            state=state, question=question, participant_cfgs=part_cfgs,
            monitor_cfg=mon_cfg, rounds=rounds, max_tokens=max_tokens,
            web_search=web_search, files_section=files_section, roles=roles,
            diversity_monitor=diversity_monitor,
            diversity_threshold=diversity_threshold,
            devils_advocate_rotation=devils_advocate_rotation,
        )

    return await _start_dialogue_session(
        mode="panel", question_preview=question, total_rounds=rounds,
        runner_coro_factory=runner,
        participants=participants_seed, moderator=moderator_seed,
        web_search=web_search, max_tokens=max_tokens,
        context_paths=context_paths,
    )


@mcp.tool()
async def model_socratic(
    topic: str,
    questioner: str | None = None,
    respondent: str | None = None,
    moderator: str | None = None,
    rounds: int = 5,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 4096,
    web_search: bool = False,
) -> dict:
    """Запустить Socratic dialogue: questioner задаёт углубляющие вопросы,
    respondent отвечает. Optional moderator пишет note после каждого раунда
    и финальный summary.

    Default: questioner=deepseek-pro, respondent=glm.
    """
    rounds = _validate_rounds(rounds)
    q_id = questioner or DEFAULT_SOCRATIC_QUESTIONER
    r_id = respondent or DEFAULT_SOCRATIC_RESPONDENT
    if q_id == r_id:
        raise RuntimeError(
            f"questioner and respondent must be distinct, both are '{q_id}'"
        )
    q_cfg = _resolve_engine_cfg(q_id)
    r_cfg = _resolve_engine_cfg(r_id)
    m_cfg = _resolve_engine_cfg(moderator) if moderator else None
    max_tokens = _clamp_tokens(max_response_tokens)
    files_section = await _build_files_section_or_none(context_paths)

    participants_seed = [
        {"id": q_cfg["id"], "model": q_cfg["model"], "position": None, "role": "questioner"},
        {"id": r_cfg["id"], "model": r_cfg["model"], "position": None, "role": "respondent"},
    ]
    moderator_seed = {"id": m_cfg["id"], "model": m_cfg["model"]} if m_cfg else None

    async def runner(state):
        await run_socratic(
            state=state, topic=topic, questioner_cfg=q_cfg, respondent_cfg=r_cfg,
            moderator_cfg=m_cfg, rounds=rounds, max_tokens=max_tokens,
            web_search=web_search, files_section=files_section,
        )

    return await _start_dialogue_session(
        mode="socratic", question_preview=topic, total_rounds=rounds,
        runner_coro_factory=runner,
        participants=participants_seed, moderator=moderator_seed,
        web_search=web_search, max_tokens=max_tokens,
        context_paths=context_paths,
    )


@mcp.tool()
async def dialogue_status(session_id: str) -> dict:
    """Live snapshot of a dialogue session: phase, current_round, elapsed_ms.
    Safe to poll often. Returns {error: ...} if session_id is unknown."""
    state = await dialogue_state.get_session(session_id)
    if state is None:
        return {"error": f"unknown session_id: {session_id}"}
    return dialogue_state.snapshot(state)


@mcp.tool()
async def dialogue_result(session_id: str) -> dict:
    """Fetch the final markdown for a completed dialogue session.

    If the session is not yet done, returns {ready: False, phase, hint}. If the
    session is done, returns {ready: True, phase, result_markdown, dump_path}.
    Errored/cancelled sessions return ready=True with the partial markdown and
    the error message."""
    state = await dialogue_state.get_session(session_id)
    if state is None:
        return {"error": f"unknown session_id: {session_id}"}
    if state.phase not in {"done", "error", "cancelled"}:
        return {
            "ready": False,
            "phase": state.phase,
            "current_round": state.current_round,
            "elapsed_ms": (
                int((time.time() - state.started_at) * 1000)
                if state.started_at else 0
            ),
            "hint": "Call dialogue_status(session_id) for progress, retry later.",
        }
    if state.result_markdown is None and state.history:
        state.result_markdown = format_dialogue_markdown(state, state.question)
    return {
        "ready": True,
        "phase": state.phase,
        "result_markdown": state.result_markdown or "(empty — no history)",
        "dump_path": state.dump_path,
        "error": state.error,
    }


@mcp.tool()
async def dialogue_cancel(session_id: str) -> dict:
    """Cancel a running dialogue session. No-op if already terminal."""
    ok = await dialogue_state.cancel_session(session_id)
    return {"cancelled": ok}


@mcp.tool()
async def dialogue_list_sessions(limit: int = 20) -> list[dict]:
    """List most-recent dialogue sessions (default last 20)."""
    sessions = await dialogue_state.list_sessions(limit=limit)
    return [dialogue_state.snapshot(s) for s in sessions]


DIRECTIVE_INJECTION_TEMPLATE = (
    "НОВАЯ ВВОДНАЯ ОТ МОДЕРАТОРА (применяется со следующего раунда): {directive}"
)


@mcp.tool()
async def dialogue_continue(
    session_id: str,
    directive: str,
    rounds: int = 3,
) -> dict:
    """Продолжить завершённую или прерванную сессию ещё N раундов с user-directive.

    Directive вшивается в историю как entry с phase='directive' от модератора,
    участники видят её в DIALOGUE HISTORY следующего раунда. Под капотом
    переиспользуются те же оркестраторы (run_debate/run_panel/run_socratic) с
    resume=True — отдельной копии round-loop'ов больше нет.

    Errors:
      - unknown session_id
      - session not in phase 'done'/'interrupted' (finish/cancel the run first)
      - total_rounds + rounds > DIALOGUE_ROUNDS_MAX
    """
    state = await dialogue_state.get_session(session_id)
    if state is None:
        raise RuntimeError(f"unknown session_id: {session_id}")
    # 'interrupted' = a run that died on a server restart; its full history and
    # params were persisted, so it can be resumed just like a finished one.
    if state.phase not in ("done", "interrupted"):
        raise RuntimeError(
            f"dialogue_continue requires phase 'done' or 'interrupted', got "
            f"'{state.phase}' (cancel/wait the current run first)"
        )
    new_total = state.total_rounds + rounds
    if new_total > DIALOGUE_ROUNDS_MAX:
        raise RuntimeError(
            f"total rounds would be {new_total}, exceeds max {DIALOGUE_ROUNDS_MAX}"
        )
    if rounds < 1:
        raise RuntimeError(f"rounds must be >= 1, got {rounds}")

    # Pre-flight resolves that can raise (a model removed from CATALOG, a context
    # file deleted/blocked) run BEFORE any state mutation, so a failure can't
    # leave a half-mutated zombie session stuck non-terminal forever.
    part_cfgs = [_resolve_engine_cfg(p["id"]) for p in state.participants]
    mod_cfg = _resolve_engine_cfg(state.moderator["id"]) if state.moderator else None
    files_section = await _build_files_section_or_none(state.context_paths or None)
    web_search = state.web_search
    max_tokens = state.max_tokens

    # create_session is the only gate for MAX_ACTIVE_SESSIONS; reactivating a
    # terminal session here would bypass it. Re-check the cap (same RuntimeError)
    # before any mutation so a failure can't leave a half-mutated session.
    await dialogue_state.reserve_active_slot()

    # All pre-flight passed — now mutate.
    # Strip terminal artifacts before resuming: a phase=='summary' entry has no
    # branch in format_history_section, so it would render as a plain participant
    # reply in the next round's history and leak the verdict to every model,
    # biasing the continuation toward the stated conclusion (breaks
    # anti-convergence). The renderer recreates the summary from summary_entries.
    state.history = [h for h in state.history if h["phase"] != "summary"]
    mod_id = (state.moderator or {}).get("id", "moderator")
    state.history.append({
        "round": state.current_round,
        "phase": "directive",
        "id": mod_id,
        "text": DIRECTIVE_INJECTION_TEMPLATE.format(directive=directive),
        "latency_ms": 0,
        "status": "ok",
    })
    state.total_rounds = new_total
    state.error = None
    # Reset so dialogue_status' elapsed_ms tracks the continuation, not a value
    # frozen at the first run's duration.
    state.finished_at = None
    dialogue_state.mark_phase(state, "starting")

    if state.mode == "debate":
        async def runner(s):
            await run_debate(
                state=s, question=s.question, participant_cfgs=part_cfgs,
                moderator_cfg=mod_cfg, rounds=s.total_rounds, max_tokens=max_tokens,
                web_search=web_search, files_section=files_section, resume=True,
            )
    elif state.mode == "panel":
        async def runner(s):
            await run_panel(
                state=s, question=s.question, participant_cfgs=part_cfgs,
                monitor_cfg=mod_cfg, rounds=s.total_rounds, max_tokens=max_tokens,
                web_search=web_search, files_section=files_section, roles=None,
                diversity_monitor=s.diversity_monitor,
                diversity_threshold=s.diversity_threshold,
                devils_advocate_rotation=s.devils_advocate_rotation, resume=True,
            )
    elif state.mode == "socratic":
        async def runner(s):
            await run_socratic(
                state=s, topic=s.question, questioner_cfg=part_cfgs[0],
                respondent_cfg=part_cfgs[1], moderator_cfg=mod_cfg,
                rounds=s.total_rounds, max_tokens=max_tokens, web_search=web_search,
                files_section=files_section, resume=True,
            )
    else:
        raise RuntimeError(f"unknown mode {state.mode!r}")

    task = asyncio.create_task(_dialogue_runner_guard(state, runner))
    dialogue_state.attach_task(state, task)
    return {
        "session_id": state.session_id,
        "mode": state.mode,
        "phase": state.phase,
        "total_rounds": state.total_rounds,
        "participants": list(state.participants),
        "hint": "Poll dialogue_status(session_id). When phase=='done', call dialogue_result(session_id).",
    }


def _warn_if_context_roots_unset() -> None:
    """Emit a startup guardrail warning to stderr when COUNCIL_CONTEXT_ROOTS is
    unset. Stdout is the MCP transport — diagnostics must go to stderr."""
    if not sandbox.context_roots_configured():
        print(
            "[mcp-council] WARNING: COUNCIL_CONTEXT_ROOTS is not set — context_paths "
            "run deny-list-only. A prompt-injected path can exfiltrate any "
            "non-blacklisted file to a third-party LLM. Set it to your "
            "repo/workspace dir(s) to require every context file to resolve inside.",
            file=sys.stderr,
        )


def _run_startup_recovery() -> None:
    """Warn about an unset context-roots allow-list and reload persisted job /
    dialogue snapshots, marking still-running ones 'interrupted'."""
    _warn_if_context_roots_unset()
    n_jobs = job_state.load_persisted_jobs()
    n_dlg = dialogue_state.load_persisted_dialogues()
    if n_jobs or n_dlg:
        print(
            f"[mcp-council] recovered {n_jobs} persisted job(s) and "
            f"{n_dlg} dialogue session(s); non-terminal ones marked 'interrupted' "
            "(partial result via council_result / dialogue_result).",
            file=sys.stderr,
        )


# Run recovery for every launch path (`python server.py`, `fastmcp run server.py`,
# `mcp dev server.py`) — not just __main__, which a non-direct launcher never
# triggers. Skipped under pytest so importing the module for tests has no disk
# side effects on the real logs/ directories.
if "pytest" not in sys.modules:
    _run_startup_recovery()

if __name__ == "__main__":
    mcp.run()
