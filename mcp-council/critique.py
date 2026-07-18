"""Adversarial critique orchestrator: independent lensed critics → cross-lens
dedup → adversarial verification → severity-ranked verdict.

How this differs from `council.run_council` (and why it exists alongside it):

  council_ask   — N models answer the SAME question with the SAME mandate, then
                  peer-RANK each other. Optimised for "what is the best answer?".
                  Diversity comes only from the models being different.
  council_critique — N critics get DIFFERENT mandates (lenses.LENSES) and hunt for
                  defects independently, then each surviving finding is attacked
                  by verifiers whose job is to REFUTE it. Optimised for "what is
                  wrong with this, and which of those claims survive scrutiny?".

The two failure modes this is built against:
  1. Correlated critics — N models with one prompt converge on the same shallow
     list. Countered by disjoint lens mandates (each lens has an explicit
     out_of_scope) and by spreading lenses across provider domains.
  2. Plausible-but-wrong findings — a confident model invents a bug that does not
     exist. Countered by the verification stage: verifiers are told to refute and
     to default to `refuted: true` when uncertain, and a finding is dropped on a
     majority refute. Verifiers are drawn from models that did NOT raise the
     finding, so a model never rubber-stamps its own claim.

Nothing here is a correctness oracle. `verified` means "survived adversarial
review by other models", which is a filter on noise, not evidence of a real bug.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Callable

from budget import RunBudget
from council import CallFn, ProgressFn, _compute_usage, _extract_json, _noop_progress
from dlp import redact_secrets
from healthcheck import _classify_error
from lenses import LENSES, assign_lenses
from models import effective_max_tokens, provider_domain
from openai_client import call_openai_compat
from web_search import WEB_SEARCH_TOOL_SPEC
from web_search_tool import MAX_TOOL_ITERATIONS, RunSearchCache, run_with_tool_loop

# Per-run ceiling on how many merged findings go through verification. Each one
# costs `verifiers_per_finding` LLM calls, so an over-eager critic panel could
# otherwise turn a review into hundreds of calls. Findings are verified in
# severity order; anything past the cap is REPORTED as unverified rather than
# silently dropped (see run_critique notes / summary.unverified_findings).
MAX_VERIFIED_FINDINGS = 24

# Concurrency ceiling for the verification fan-out (findings × verifiers). The
# HTTP layer has its own in-flight semaphore, but that is process-global and
# shared with every other tool — this keeps one critique run from monopolising it.
MAX_VERIFY_CONCURRENCY = 8

MAX_VERIFIERS_PER_FINDING = 5
DEFAULT_VERIFIERS_PER_FINDING = 2

SEVERITIES = ("critical", "high", "medium", "low")
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}

# Cluster two findings when their (title + claim) token sets overlap this much.
# Deliberately conservative: an over-eager merge hides a real second bug behind
# an unrelated one, which is worse than reporting a near-duplicate pair.
_SAME_LOCATION_JACCARD = 0.45
_CROSS_LOCATION_JACCARD = 0.70

# Words that appear in almost every finding and would inflate similarity between
# two unrelated ones ("missing check in handler" vs "missing check in parser").
_STOPWORDS = frozenset(
    "a an the is are was were be being been in on at to of for with without and "
    "or not no if then else this that these those it its can could should would "
    "may might will shall do does did done has have had when while from by as "
    "into over under out up down code function method class file line issue bug "
    "problem error missing must may need needs".split()
)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_CRITIC_SYSTEM_TEMPLATE = """You are ONE critic on a review panel. Other critics \
are reviewing the same material RIGHT NOW under different mandates, independently \
of you. You will never see their output and they will never see yours.

=== YOUR LENS: {title} ===
Hunt ONLY for the following:
{focus}

NOT your job (another critic owns it, and duplicates are discarded): {out_of_scope}.
Reporting an out-of-scope issue costs you nothing but wastes a slot — stay in your lane.

Rules:
1. Report only defects you can point AT — name the file/function/line or quote the
   exact construct. A finding with no location is unusable and will be dropped.
2. Every finding needs a concrete failure_scenario: specific inputs or state that
   lead to the wrong output, crash, or exposure. "Could cause problems" is not a
   failure scenario. If you cannot write one, you do not have a finding.
3. Do NOT pad. An empty findings list is a perfectly good answer and is scored as
   such — inventing a marginal issue to look thorough is the single worst thing
   you can do here, because a later stage will spend real work refuting it.
4. Do not restate the material back, do not summarise, do not praise.
5. `confidence` is YOUR honesty about this specific claim: 9-10 you traced it and
   are certain; 5-6 it looks wrong but you could not verify the surrounding
   context; 1-3 a hunch. Be honest — low confidence is useful, false confidence
   is not.

Return STRICT JSON only. No markdown fence, no prose before or after. Schema:
{{"findings": [{{"title": "short defect name, <=80 chars",
              "severity": "critical|high|medium|low",
              "location": "file:line, function name, or exact quoted construct",
              "claim": "what is wrong, in 1-2 sentences",
              "failure_scenario": "concrete inputs/state -> concrete bad outcome",
              "fix": "the smallest change that resolves it, 1 sentence",
              "confidence": 1-10}}]}}

severity: critical = data loss / security breach / production outage; high = wrong
results or a crash on a reachable path; medium = degraded behaviour or a latent
trap; low = worth fixing, no user-visible impact. Use [] for findings if you found
nothing in your lane."""

_CRITIC_WEB_SEARCH_NOTE = (
    "\n\nYou have a `web_search(query)` tool. Use it ONLY to check an external "
    "fact your finding depends on (an API's documented behaviour, a CVE, a "
    "library's default). Do not research the topic in general — most critiques "
    "need zero searches."
)

# Each verifier attacks from a different direction. Redundant verifiers agree
# with each other for the same reason they agree with the finding; distinct
# attack angles catch failure modes identical refuters cannot.
REFUTE_ANGLES: list[dict] = [
    {
        "id": "does-not-reproduce",
        "instruction": (
            "Attack the FAILURE SCENARIO. Walk the described inputs through the "
            "actual code path step by step. Does the claimed bad outcome really "
            "occur, or does the path diverge somewhere the critic did not check?"
        ),
    },
    {
        "id": "already-handled",
        "instruction": (
            "Assume the defect is already prevented somewhere the critic did not "
            "look — a caller-side guard, a validation layer, a type constraint, an "
            "earlier early-return, a framework default. Find that guard."
        ),
    },
    {
        "id": "misreads-the-code",
        "instruction": (
            "Assume the critic misread something: wrong scope, wrong overload, a "
            "variable that is not what they think, a comment describing old "
            "behaviour. Check the literal text of the cited location."
        ),
    },
    {
        "id": "not-reachable",
        "instruction": (
            "Attack REACHABILITY. Can the required state or input actually occur in "
            "this system, given who calls this and with what? A defect on a "
            "genuinely unreachable path is not a defect."
        ),
    },
    {
        "id": "wrong-severity",
        "instruction": (
            "Grant that something is off, then attack the SEVERITY and the framing: "
            "is the real impact far smaller than claimed, or is this a style "
            "preference dressed up as a defect?"
        ),
    },
]

_REFUTE_SYSTEM_TEMPLATE = """You are an adversarial verifier. Another model claims \
it found a defect. Your job is to REFUTE that claim — not to evaluate it neutrally, \
and definitely not to agree with it.

=== YOUR ATTACK ANGLE ===
{angle_instruction}

Ground rules:
1. The burden of proof is on the CLAIM, not on you. If the material does not
   clearly show the defect is real, the correct verdict is `refuted: true`.
2. Default to `refuted: true` when uncertain. A wrong finding that survives costs
   a human a real debugging session; a correct finding that gets refuted here is
   still visible in the report as contested. Asymmetric — act accordingly.
3. Do NOT refute on vibes. Your `reasoning` must cite the specific code, guard, or
   call path that defeats the claim. "Seems fine" is not a refutation and will be
   read as a failed verification.
4. If the claim is unambiguously correct and you cannot find a defeater, say so
   honestly: `refuted: false`. Confirming a real defect is a valid outcome.
5. Judge ONLY this claim. Not the critic's other findings, not the overall code.

Return STRICT JSON only. No markdown fence, no prose. Schema:
{{"refuted": true|false,
 "confidence": 1-10,
 "reasoning": "the specific evidence for your verdict, 2-4 sentences",
 "strongest_counterpoint": "the best argument AGAINST your own verdict, 1 sentence"}}

`confidence` is how sure you are of YOUR verdict. `strongest_counterpoint` is
mandatory — state the best case against yourself even when you are confident."""


def _build_critic_user(subject: str, files_section: str | None) -> str:
    parts: list[str] = []
    if files_section:
        parts.append(files_section)
    parts.append("=== MATERIAL UNDER REVIEW ===")
    parts.append(subject)
    parts.append("")
    parts.append("Now return the STRICT JSON findings object for your lens.")
    return "\n".join(parts)


def _build_refute_user(
    subject: str, finding: dict, files_section: str | None
) -> str:
    parts: list[str] = []
    if files_section:
        parts.append(files_section)
    parts.append("=== MATERIAL UNDER REVIEW ===")
    parts.append(subject)
    parts.append("")
    parts.append("=== THE CLAIM YOU MUST TRY TO REFUTE ===")
    parts.append(f"Title: {finding['title']}")
    parts.append(f"Claimed severity: {finding['severity']}")
    parts.append(f"Location: {finding['location']}")
    parts.append(f"Claim: {finding['claim']}")
    parts.append(f"Claimed failure scenario: {finding['failure_scenario']}")
    parts.append("")
    parts.append("Return the STRICT JSON verdict object.")
    return "\n".join(parts)


def _build_repair_user(error: str, schema_hint: str) -> str:
    return (
        f"Your previous reply could not be parsed (error: {error}).\n"
        "Reply AGAIN with STRICT JSON ONLY — no markdown fence, no prose before "
        f"or after. Exactly this schema:\n{schema_hint}"
    )


_CRITIC_SCHEMA_HINT = (
    '{"findings": [{"title": "...", "severity": "critical|high|medium|low", '
    '"location": "...", "claim": "...", "failure_scenario": "...", '
    '"fix": "...", "confidence": 1-10}]}'
)
_REFUTE_SCHEMA_HINT = (
    '{"refuted": true|false, "confidence": 1-10, "reasoning": "...", '
    '"strongest_counterpoint": "..."}'
)


# --------------------------------------------------------------------------
# Parsing / normalisation
# --------------------------------------------------------------------------


class CritiqueParseError(ValueError):
    """JSON was present but unusable (wrong shape / no salvageable entries).

    Distinguished from a raw json failure so the caller knows a single repair
    retry with a strict-JSON demand is worth one attempt — same contract as
    council._Stage2ParseError.
    """


def _norm_severity(value: object) -> str:
    s = str(value or "").strip().lower()
    return s if s in _SEVERITY_RANK else "medium"


def _norm_confidence(value: object) -> int | None:
    try:
        c = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return c if 1 <= c <= 10 else None


def normalize_findings(content: str | None) -> list[dict]:
    """Parse a critic reply into a clean findings list.

    An EMPTY list is a valid result ("nothing in my lane") and is returned as
    such — only a structurally unusable reply raises. Entries missing a location
    or a failure_scenario are dropped: the critic prompt makes both mandatory
    precisely because an unlocatable claim cannot be verified or acted on.
    """
    parsed = _extract_json(content or "")  # ValueError when there is no object
    raw = parsed.get("findings", [])
    if not isinstance(raw, list):
        raise CritiqueParseError("findings is not a list")

    clean: list[dict] = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        title = str(f.get("title", "")).strip()[:200]
        location = str(f.get("location", "")).strip()[:300]
        claim = str(f.get("claim", "")).strip()[:1500]
        scenario = str(f.get("failure_scenario", "")).strip()[:1500]
        if not title or not location or not scenario:
            continue
        clean.append({
            "title": title,
            "severity": _norm_severity(f.get("severity")),
            "location": location,
            "claim": claim,
            "failure_scenario": scenario,
            "fix": str(f.get("fix", "")).strip()[:800],
            "confidence": _norm_confidence(f.get("confidence")),
        })
    return clean


def normalize_verdict(content: str | None) -> dict:
    """Parse a verifier reply into {refuted, confidence, reasoning, counterpoint}.

    `refuted` must be an explicit boolean. A missing or non-boolean value raises
    rather than defaulting: silently reading a malformed verdict as "not refuted"
    would let a broken verifier confirm findings for free.
    """
    parsed = _extract_json(content or "")
    refuted = parsed.get("refuted")
    if isinstance(refuted, str):
        low = refuted.strip().lower()
        if low in ("true", "yes"):
            refuted = True
        elif low in ("false", "no"):
            refuted = False
    if not isinstance(refuted, bool):
        raise CritiqueParseError(f"'refuted' is not a boolean: {parsed.get('refuted')!r}")
    return {
        "refuted": refuted,
        "confidence": _norm_confidence(parsed.get("confidence")),
        "reasoning": str(parsed.get("reasoning", "")).strip()[:1200],
        "strongest_counterpoint": str(parsed.get("strongest_counterpoint", "")).strip()[:600],
    }


# --------------------------------------------------------------------------
# Shared single-call helper
# --------------------------------------------------------------------------


async def _json_call(
    member: dict,
    system: str,
    user: str,
    max_response_tokens: int,
    call_fn: CallFn,
    parse: Callable[[str | None], Any],
    schema_hint: str,
    *,
    web_search: bool = False,
    search_cache: RunSearchCache | None = None,
) -> dict:
    """One LLM call that must return JSON, with a single repair retry.

    Always returns a record dict (never raises for a provider failure) shaped so
    council._compute_usage can account it: id / tokens_in / tokens_out / attempts
    / tool_calls_log. On success `parsed` holds the parse() result.
    """
    start = time.monotonic()
    mid = member["id"]

    def _rec(**kw: Any) -> dict:
        base = {
            "id": mid,
            "model": member["model"],
            "latency_ms": int((time.monotonic() - start) * 1000),
            "tokens_in": None,
            "tokens_out": None,
            "attempts": None,
            "tool_calls_log": [],
            "parsed": None,
            "error": None,
        }
        base.update(kw)
        return base

    api_key = os.environ.get(member["env_key"])
    if not api_key:
        return _rec(status="error", error=f"env var {member['env_key']} not set", latency_ms=0)

    max_tokens = effective_max_tokens(max_response_tokens, member)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    tools = [WEB_SEARCH_TOOL_SPEC] if web_search else None

    async def _call(msgs: list[dict], *, force_json: bool) -> tuple[dict, list]:
        if tools and not force_json:
            # The repair pass never needs tools — by then the model has all the
            # material it asked for and only has to re-emit valid JSON.
            return await run_with_tool_loop(
                member=member, api_key=api_key, messages=msgs,
                max_tokens=max_tokens, call_fn=call_fn, tools=tools,
                search_cache=search_cache,
            )
        result = await call_fn(
            base_url=member["base_url"],
            api_key=api_key,
            model=member["model"],
            messages=msgs,
            max_tokens=max_tokens,
            extra_payload=member.get("extra"),
            response_format={"type": "json_object"} if force_json else None,
        )
        return result, []

    try:
        result, tool_log = await _call(messages, force_json=False)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Mirrors council._run_member_stage1: any non-cancellation failure becomes
        # a record so asyncio.gather never aborts a whole fan-out, and the message
        # is redacted because it travels back to the MCP client.
        return _rec(
            status="error", error=redact_secrets(str(e)),
            attempts=getattr(e, "attempts", None),
        )

    tin = result.get("tokens_in") or 0
    tout = result.get("tokens_out") or 0
    attempts = result.get("attempts")
    loop_keys = {
        k: result[k] for k in
        ("loop_calls", "loop_tokens_in", "loop_tokens_out", "loop_attempts")
        if k in result
    }

    if not result.get("content"):
        return _rec(
            status="error",
            error=(
                f"no final content after {MAX_TOOL_ITERATIONS} tool iterations "
                f"(finish_reason={result.get('finish_reason')})"
            ),
            tokens_in=tin, tokens_out=tout, attempts=attempts,
            tool_calls_log=tool_log, **loop_keys,
        )

    try:
        parsed = parse(result.get("content"))
    except (CritiqueParseError, ValueError, TypeError, KeyError) as first_err:
        repair = messages + [
            {"role": "assistant", "content": result.get("content") or ""},
            {"role": "user", "content": _build_repair_user(str(first_err), schema_hint)},
        ]
        try:
            result2, _ = await _call(repair, force_json=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            result2 = None
        if result2 is None:
            return _rec(
                status="error", error=f"invalid_json: {first_err}",
                tokens_in=tin, tokens_out=tout, attempts=attempts,
                tool_calls_log=tool_log, **loop_keys,
            )
        tin += result2.get("tokens_in") or 0
        tout += result2.get("tokens_out") or 0
        attempts = (attempts or 0) + (result2.get("attempts") or 0)
        try:
            parsed = parse(result2.get("content"))
        except (CritiqueParseError, ValueError, TypeError, KeyError) as second_err:
            return _rec(
                status="error",
                error=f"invalid_json: {second_err} (after repair retry)",
                tokens_in=tin, tokens_out=tout, attempts=attempts,
                tool_calls_log=tool_log, repaired=True, **loop_keys,
            )
        return _rec(
            status="ok", parsed=parsed, tokens_in=tin, tokens_out=tout,
            attempts=attempts, tool_calls_log=tool_log, repaired=True, **loop_keys,
        )

    return _rec(
        status="ok", parsed=parsed, tokens_in=tin, tokens_out=tout,
        attempts=attempts, tool_calls_log=tool_log, **loop_keys,
    )


# --------------------------------------------------------------------------
# Stage 1 — independent lensed critics
# --------------------------------------------------------------------------


async def _run_critic(
    assignment: dict,
    subject: str,
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
    *,
    web_search: bool,
    search_cache: RunSearchCache | None,
) -> dict:
    lens_id = assignment["lens"]
    lens = LENSES[lens_id]
    member = assignment["member"]
    system = _CRITIC_SYSTEM_TEMPLATE.format(
        title=lens["title"], focus=lens["focus"], out_of_scope=lens["out_of_scope"]
    )
    if web_search:
        system += _CRITIC_WEB_SEARCH_NOTE
    rec = await _json_call(
        member=member, system=system,
        user=_build_critic_user(subject, files_section),
        max_response_tokens=max_response_tokens, call_fn=call_fn,
        parse=normalize_findings, schema_hint=_CRITIC_SCHEMA_HINT,
        web_search=web_search, search_cache=search_cache,
    )
    rec["lens"] = lens_id
    rec["lens_title"] = lens["title"]
    rec["findings"] = rec.pop("parsed") or []
    return rec


# --------------------------------------------------------------------------
# Stage 2 — cross-lens dedup (pure, no LLM)
# --------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    """Comparable token set for similarity. Stopwords out, short noise words out
    — but DIGITS are kept even when short: an index, port, status code, or
    version is exactly the discriminator that tells two otherwise
    identically-worded findings apart."""
    words = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return {
        w for w in words
        if w not in _STOPWORDS and (len(w) > 2 or w.isdigit())
    }


def _norm_location(loc: str) -> str:
    """Collapse a location to a comparable key: lowercase, line numbers dropped.

    'server.py:412' and 'server.py:418' are almost always the same site seen by
    two critics who counted lines differently, so the line number is noise here.
    """
    low = (loc or "").lower().strip()
    low = re.sub(r":\s*\d+(-\d+)?", "", low)
    low = re.sub(r"\s+", " ", low)
    return low


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _same_finding(f1: dict, f2: dict) -> bool:
    t1, t2 = f1["_tokens"], f2["_tokens"]
    sim = _jaccard(t1, t2)
    l1, l2 = f1["_loc"], f2["_loc"]
    location_matches = bool(l1) and bool(l2) and (l1 == l2 or l1 in l2 or l2 in l1)
    if location_matches:
        return sim >= _SAME_LOCATION_JACCARD
    return sim >= _CROSS_LOCATION_JACCARD


def cluster_findings(critic_records: list[dict]) -> list[dict]:
    """Merge near-duplicate findings raised by different critics.

    Two critics reaching the same defect through different lenses is the single
    strongest signal this whole mode produces — but only if the duplicates are
    actually merged, otherwise it reads as two separate weak findings. Pure
    function (no LLM): an LLM deduper would itself need verifying.

    Returns merged findings sorted by (severity, corroboration, confidence), each
    with `raised_by` = [{model, lens, severity, confidence}, ...].
    """
    flat: list[dict] = []
    for rec in critic_records:
        if rec.get("status") != "ok":
            continue
        for f in rec.get("findings") or []:
            item = dict(f)
            item["_tokens"] = _tokens(f["title"] + " " + f["claim"])
            item["_loc"] = _norm_location(f["location"])
            item["_by"] = {
                "model": rec["id"],
                "lens": rec["lens"],
                "severity": f["severity"],
                "confidence": f["confidence"],
            }
            flat.append(item)

    clusters: list[list[dict]] = []
    for item in flat:
        for cluster in clusters:
            if any(_same_finding(item, existing) for existing in cluster):
                cluster.append(item)
                break
        else:
            clusters.append([item])

    merged: list[dict] = []
    for cluster in clusters:
        # Representative = most severe, then most confident: the strongest
        # statement of the defect, not an averaged-out one.
        rep = min(
            cluster,
            key=lambda f: (_SEVERITY_RANK[f["severity"]], -(f["confidence"] or 0)),
        )
        raised_by = [f["_by"] for f in cluster]
        merged.append({
            "title": rep["title"],
            "severity": rep["severity"],
            "location": rep["location"],
            "claim": rep["claim"],
            "failure_scenario": rep["failure_scenario"],
            "fix": rep["fix"],
            "confidence": rep["confidence"],
            "raised_by": raised_by,
            "lenses": sorted({b["lens"] for b in raised_by}),
            "raised_by_models": sorted({b["model"] for b in raised_by}),
            # Independent corroboration at the RAISING stage: two lenses on one
            # model share a model's blind spots, two provider domains do not.
            "raised_by_domains": sorted({provider_domain(b["model"]) for b in raised_by}),
            # Alternative phrasings from the other critics in the cluster —
            # sometimes a weaker-severity duplicate explains the defect better.
            "duplicates": [
                {"model": f["_by"]["model"], "lens": f["_by"]["lens"],
                 "title": f["title"], "claim": f["claim"]}
                for f in cluster if f is not rep
            ],
        })

    merged.sort(key=lambda f: (
        _SEVERITY_RANK[f["severity"]],
        -len(f["raised_by_domains"]),
        -len(f["raised_by"]),
        -(f["confidence"] or 0),
    ))
    return merged


# --------------------------------------------------------------------------
# Stage 3 — adversarial verification
# --------------------------------------------------------------------------


def pick_verifiers(
    finding: dict, members: list[dict], k: int
) -> list[tuple[dict, dict]]:
    """Choose k (member, angle) pairs to attack `finding`.

    Preference order:
      1. models that did NOT raise it (no self-review),
      2. provider domains not already represented among the pick,
      3. domains not already represented among the RAISERS — a verifier sharing
         the raiser's gateway/credential shares its correlated blind spots.
    Falls back to raisers only when there are not enough other members; those
    picks are flagged `self_review` in the record so the report never presents a
    model confirming its own claim as independent corroboration.
    """
    raiser_ids = set(finding["raised_by_models"])
    raiser_domains = set(finding["raised_by_domains"])
    others = [m for m in members if m["id"] not in raiser_ids]
    fallback = [m for m in members if m["id"] in raiser_ids]

    picked: list[dict] = []
    used_domains: set[str] = set()

    def _take(pool: list[dict]) -> None:
        # Two passes: fresh domains first, then anything left in the pool.
        for prefer_fresh_vs_raisers in (True, False):
            for m in pool:
                if len(picked) >= k:
                    return
                if m in picked:
                    continue
                d = provider_domain(m["id"])
                if d in used_domains:
                    continue
                if prefer_fresh_vs_raisers and d in raiser_domains:
                    continue
                picked.append(m)
                used_domains.add(d)
        for m in pool:  # domains exhausted — allow a repeat domain
            if len(picked) >= k:
                return
            if m not in picked:
                picked.append(m)

    _take(others)
    if len(picked) < k:
        _take(fallback)

    return [
        (m, REFUTE_ANGLES[i % len(REFUTE_ANGLES)])
        for i, m in enumerate(picked[:k])
    ]


async def _run_verifier(
    member: dict,
    angle: dict,
    finding: dict,
    finding_index: int,
    subject: str,
    files_section: str | None,
    max_response_tokens: int,
    call_fn: CallFn,
) -> dict:
    rec = await _json_call(
        member=member,
        system=_REFUTE_SYSTEM_TEMPLATE.format(angle_instruction=angle["instruction"]),
        user=_build_refute_user(subject, finding, files_section),
        max_response_tokens=max_response_tokens, call_fn=call_fn,
        parse=normalize_verdict, schema_hint=_REFUTE_SCHEMA_HINT,
    )
    verdict = rec.pop("parsed") or {}
    rec.update({
        "angle": angle["id"],
        "finding_index": finding_index,
        "self_review": member["id"] in finding["raised_by_models"],
        "refuted": verdict.get("refuted"),
        "verdict_confidence": verdict.get("confidence"),
        "reasoning": verdict.get("reasoning"),
        "strongest_counterpoint": verdict.get("strongest_counterpoint"),
    })
    return rec


def _apply_verdicts(finding: dict, verdicts: list[dict]) -> dict:
    """Fold verifier verdicts into the finding: status + corroboration signals.

    status:
      confirmed — every valid verifier failed to refute it;
      contested — refuted by a minority (the finding stands, but flagged);
      refuted   — refuted by half or more of the valid verifiers, so it is
                  dropped from the main report. Half counts as refuted because
                  verifiers are instructed to default to `refuted` only when the
                  claim is NOT clearly supported — a split panel means the claim
                  never cleared that bar.
      unverified — no verifier returned a usable verdict (all errored).
    """
    valid = [v for v in verdicts if v.get("status") == "ok" and v.get("refuted") is not None]
    refuted = [v for v in valid if v["refuted"]]
    independent = [v for v in valid if not v.get("self_review")]
    domains = sorted({provider_domain(v["id"]) for v in independent})

    if not valid:
        status = "unverified"
    elif not refuted:
        status = "confirmed"
    elif len(refuted) * 2 >= len(valid):
        status = "refuted"
    else:
        status = "contested"

    out = dict(finding)
    out.update({
        "status": status,
        "verdicts": verdicts,
        "verifier_count": len(valid),
        "refuted_count": len(refuted),
        # Independent = verifiers that did NOT raise this finding. A finding
        # "confirmed" only by its own raiser is not corroborated, and this pair of
        # fields is what the summary gates the strong wording on.
        "independent_verifiers": len(independent),
        "verifier_domains": domains,
        "verification_quorum_ok": len(independent) >= 2 and len(domains) >= 2,
    })
    return out


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def _failure_reason(error: str | None) -> str:
    """Same coarse enum council uses, so automation branches on a stable code
    instead of substring-matching a human-readable message."""
    if not error:
        return "error"
    low = error.lower()
    if "env var" in low and "not set" in low:
        return "no_key"
    return _classify_error(error)


def build_critique_summary(
    critics: list[dict],
    verified: list[dict],
    unverified_overflow: list[dict],
    lens_ids: list[str],
) -> dict:
    """Machine-readable verdict for automation: counts by status/severity, which
    lenses actually produced signal, whether the run is trustworthy at all."""
    ok_critics = [c for c in critics if c.get("status") == "ok"]
    failed = [
        {"id": c["id"], "model": c["model"], "lens": c.get("lens"),
         "error": c.get("error"), "failure_reason": _failure_reason(c.get("error"))}
        for c in critics if c.get("status") != "ok"
    ]

    kept = [f for f in verified if f["status"] in ("confirmed", "contested", "unverified")]
    dropped = [f for f in verified if f["status"] == "refuted"]

    by_severity: dict[str, int] = {s: 0 for s in SEVERITIES}
    for f in kept:
        by_severity[f["severity"]] += 1
    by_status: dict[str, int] = {}
    for f in verified:
        by_status[f["status"]] = by_status.get(f["status"], 0) + 1

    domains = sorted({provider_domain(c["id"]) for c in ok_critics})
    # Corroboration for the RUN as a whole. One provider domain means every
    # critic shares a gateway and a credential — they fail together and agree for
    # correlated reasons, so no finding from such a run is independently sourced.
    panel_quorum_ok = len(ok_critics) >= 2 and len(domains) >= 2

    top = [
        f for f in kept
        if f["severity"] in ("critical", "high") and f["status"] == "confirmed"
    ]
    corroborated_top = [f for f in top if f.get("verification_quorum_ok")]

    if not ok_critics:
        next_action = "Every critic failed — run model_healthcheck and retry; no review happened."
    elif not panel_quorum_ok:
        next_action = (
            f"Panel ran on {len(ok_critics)} critic(s) across {len(domains)} provider "
            "domain(s) — findings are one correlated opinion, not an independent "
            "review. Re-run with models spanning ≥2 domains before acting."
        )
    elif not kept and dropped:
        # Raised-then-all-refuted is a DIFFERENT outcome from found-nothing: the
        # critics did engage with the material, and the refutations themselves are
        # the useful artifact (they explain why each suspicion was wrong).
        next_action = (
            f"All {len(dropped)} finding(s) were refuted by verification — the "
            "critics engaged but nothing held up. Skim the Refuted section: a "
            "wrongly-refuted real defect is the main failure mode here."
        )
    elif not kept:
        next_action = (
            "No critic raised anything in its lane. That is a real signal only if "
            "the lenses matched the material — check lenses_with_findings against "
            "lenses_used before reading it as a clean bill of health."
        )
    elif corroborated_top:
        next_action = (
            f"{len(corroborated_top)} confirmed critical/high finding(s) with "
            "independent verification — fix these first, verifying each against "
            "the code yourself before changing anything."
        )
    elif top:
        next_action = (
            f"{len(top)} confirmed critical/high finding(s), but none cleared the "
            "independent-verification quorum — treat as leads to check by hand, "
            "not as established defects."
        )
    else:
        next_action = (
            "Only medium/low or contested findings — triage by hand; nothing here "
            "warrants an urgent fix."
        )

    return {
        "lenses_used": lens_ids,
        # Which lenses RAISED anything (before verification). This is the check
        # behind "did the lenses match the material at all?" — computing it from
        # surviving findings instead would report an empty list for a run where
        # every lens engaged and every claim was then refuted.
        "lenses_with_findings": sorted({
            c["lens"] for c in ok_critics if c.get("findings")
        }),
        "lenses_with_surviving_findings": sorted({
            b["lens"] for f in kept for b in f["raised_by"]
        }),
        "critics_total": len(critics),
        "critics_ok": len(ok_critics),
        "provider_domains": len(domains),
        "single_provider": len(domains) < 2,
        "panel_quorum_ok": panel_quorum_ok,
        "failed_critics": failed,
        "raw_findings": sum(len(c.get("findings") or []) for c in ok_critics),
        "merged_findings": len(verified) + len(unverified_overflow),
        "findings_kept": len(kept),
        "findings_refuted": len(dropped),
        "unverified_findings": len(unverified_overflow),
        "by_status": by_status,
        "by_severity": by_severity,
        "cross_lens_corroborated": sum(1 for f in kept if len(f["lenses"]) > 1),
        # This mode filters noise; it does not establish correctness. Anything it
        # surfaces still needs a human to confirm against the actual code.
        "human_review_required": True,
        "recommended_next_action": next_action,
    }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


async def run_critique(
    subject: str,
    members: list[dict],
    lens_ids: list[str],
    *,
    files_section: str | None = None,
    max_response_tokens: int = 8192,
    verifiers_per_finding: int = DEFAULT_VERIFIERS_PER_FINDING,
    max_verified_findings: int = MAX_VERIFIED_FINDINGS,
    web_search: bool = False,
    call_fn: CallFn | None = None,
    on_progress: ProgressFn | None = None,
    budget: RunBudget | None = None,
) -> dict:
    """Run the full critique: lensed critics → dedup → adversarial verification.

    `verifiers_per_finding=0` skips verification entirely (every merged finding
    comes back with status "unverified") — useful for a cheap first pass, but the
    resulting list is unfiltered model output.

    Raises RuntimeError("critique fully failed") when no critic survived, so the
    caller never renders an empty report as "nothing found".
    """
    call_fn = call_fn or call_openai_compat
    progress = on_progress or _noop_progress
    if not (0 <= verifiers_per_finding <= MAX_VERIFIERS_PER_FINDING):
        raise ValueError(
            f"verifiers_per_finding must be in [0, {MAX_VERIFIERS_PER_FINDING}]"
        )

    notes: list[str] = []
    # One shared Exa cache per run — lensed critics issue overlapping queries and
    # would otherwise pay per duplicate. A budget.max_web_searches lowers the cap.
    search_cache: RunSearchCache | None = None
    if web_search:
        if budget is not None and budget.max_web_searches is not None:
            search_cache = RunSearchCache(max_searches=budget.max_web_searches)
        else:
            search_cache = RunSearchCache()

    # --- Stage 1: independent lensed critics ------------------------------
    assignments = assign_lenses(lens_ids, members)
    progress("phase", {
        "phase": "critique",
        "critics": [f"{a['lens']}@{a['member']['id']}" for a in assignments],
    })

    async def _critic_wrap(a: dict) -> dict:
        r = await _run_critic(
            a, subject, files_section, max_response_tokens, call_fn,
            web_search=web_search, search_cache=search_cache,
        )
        # Reuse the council's stage-1 event type so the existing job-state
        # progress callback mirrors critics live with no changes on its side.
        progress("stage1_member", {
            "id": f"{r['lens']}@{r['id']}", "model": r["model"],
            "status": r["status"], "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
            "findings": len(r.get("findings") or []),
        })
        return r

    critics = await asyncio.gather(*(_critic_wrap(a) for a in assignments))
    ok_critics = [c for c in critics if c["status"] == "ok"]
    if not ok_critics:
        progress("phase", {"phase": "error", "error": "critique fully failed"})
        raise RuntimeError("critique fully failed")
    for c in critics:
        if c["status"] != "ok":
            notes.append(
                f"{c['id']} ({c['model']}) on lens '{c['lens']}': {c['error']}; "
                "that lens produced no findings this run"
            )

    # --- Stage 2: cross-lens dedup (pure) ---------------------------------
    merged = cluster_findings(critics)
    progress("phase", {"phase": "dedup", "merged_findings": len(merged)})

    # --- Stage 3: adversarial verification --------------------------------
    to_verify = merged[:max_verified_findings]
    overflow = merged[max_verified_findings:]
    if overflow:
        # Never let a cap read as "that was everything".
        notes.append(
            f"{len(overflow)} lower-severity finding(s) exceeded the "
            f"max_verified_findings={max_verified_findings} cap and were NOT "
            "verified; they are listed separately as unverified."
        )

    budget_stop = (
        budget.check(_compute_usage([{"stage1": critics, "stage2": []}], None, search_cache))
        if budget is not None else None
    )
    if verifiers_per_finding == 0:
        notes.append("verification skipped (verifiers_per_finding=0) — findings are unfiltered.")
        verifiers: list[dict] = []
        verified = [_apply_verdicts(f, []) for f in to_verify]
    elif budget_stop:
        notes.append(f"verification skipped: budget — {budget_stop}")
        verifiers = []
        verified = [_apply_verdicts(f, []) for f in to_verify]
    else:
        progress("phase", {
            "phase": "verify",
            "findings": len(to_verify),
            "verifiers_per_finding": verifiers_per_finding,
        })
        sem = asyncio.Semaphore(MAX_VERIFY_CONCURRENCY)

        async def _verify_wrap(idx: int, finding: dict, member: dict, angle: dict) -> dict:
            async with sem:
                r = await _run_verifier(
                    member, angle, finding, idx, subject, files_section,
                    max_response_tokens, call_fn,
                )
            progress("stage2_ranker", {
                "id": f"verify{idx}:{angle['id']}@{r['id']}", "model": r["model"],
                "status": r["status"], "error": r.get("error"),
                "latency_ms": r.get("latency_ms"), "refuted": r.get("refuted"),
            })
            return r

        tasks = []
        for idx, finding in enumerate(to_verify):
            for member, angle in pick_verifiers(finding, members, verifiers_per_finding):
                tasks.append(_verify_wrap(idx, finding, member, angle))
        verifiers = list(await asyncio.gather(*tasks)) if tasks else []

        by_finding: dict[int, list[dict]] = {}
        for v in verifiers:
            by_finding.setdefault(v["finding_index"], []).append(v)
        verified = [
            _apply_verdicts(f, by_finding.get(i, []))
            for i, f in enumerate(to_verify)
        ]

    unverified_overflow = [_apply_verdicts(f, []) for f in overflow]
    progress("phase", {"phase": "done"})

    return {
        "subject_preview": subject[:200],
        "critics": critics,
        "verifiers": verifiers,
        "findings": verified,
        "unverified_findings": unverified_overflow,
        "notes": notes,
        # Reuses the council accountant: critic records go in the stage1 slot,
        # verifier records in stage2. Both carry the id/tokens/attempts/tool-log
        # keys it reads, so llm_calls / cost / web_search accounting is identical.
        "usage": _compute_usage(
            [{"stage1": critics, "stage2": verifiers}], None, search_cache
        ),
        "summary": build_critique_summary(critics, verified, unverified_overflow, lens_ids),
        "budget": budget.as_dict() if budget is not None else None,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_STATUS_MARK = {
    "confirmed": "✅ confirmed",
    "contested": "⚠️ contested",
    "unverified": "· unverified",
    "refuted": "❌ refuted",
}


def _render_finding(f: dict, n: int) -> list[str]:
    lines = [
        f"### {n}. [{f['severity'].upper()}] {f['title']}",
        "",
        f"**Status:** {_STATUS_MARK.get(f['status'], f['status'])}"
        + (
            f" ({f['verifier_count'] - f['refuted_count']}/{f['verifier_count']} verifiers failed to refute)"
            if f["verifier_count"] else ""
        ),
        f"**Location:** {f['location']}",
        f"**Raised by:** "
        + ", ".join(f"{b['model']} ({b['lens']})" for b in f["raised_by"])
        + (
            f" — corroborated across {len(f['lenses'])} lenses / "
            f"{len(f['raised_by_domains'])} provider domains"
            if len(f["lenses"]) > 1 else ""
        ),
        "",
        f"{f['claim']}",
        "",
        f"**Failure scenario:** {f['failure_scenario']}",
    ]
    if f.get("fix"):
        lines.append(f"**Proposed fix:** {f['fix']}")
    if f.get("duplicates"):
        lines.append("")
        lines.append("<details><summary>Also raised as</summary>")
        lines.append("")
        for d in f["duplicates"]:
            lines.append(f"- *{d['model']} ({d['lens']})* — {d['title']}: {d['claim']}")
        lines.append("")
        lines.append("</details>")
    valid = [v for v in f.get("verdicts", []) if v.get("status") == "ok"]
    if valid:
        lines.append("")
        lines.append("<details><summary>Verification</summary>")
        lines.append("")
        for v in valid:
            mark = "refutes" if v["refuted"] else "fails to refute"
            self_note = " *(self-review)*" if v.get("self_review") else ""
            conf = f", conf {v['verdict_confidence']}/10" if v.get("verdict_confidence") else ""
            lines.append(
                f"- **{v['id']}** via `{v['angle']}`{self_note} — {mark}{conf}: "
                f"{v.get('reasoning') or ''}"
            )
            if v.get("strongest_counterpoint"):
                lines.append(f"  - counterpoint: {v['strongest_counterpoint']}")
        lines.append("")
        lines.append("</details>")
    lines.append("")
    return lines


def format_critique_markdown(subject: str, result: dict) -> str:
    """Render a critique run into the markdown brief the MCP tool returns."""
    s = result["summary"]
    usage = result.get("usage") or {}
    lines: list[str] = ["# Adversarial critique", ""]

    lines.append(f"**Subject:** {subject[:300]}{'…' if len(subject) > 300 else ''}")
    lines.append(
        f"**Panel:** {s['critics_ok']}/{s['critics_total']} critics across "
        f"{s['provider_domains']} provider domain(s) · lenses: "
        + ", ".join(s["lenses_used"])
    )
    lines.append(
        f"**Findings:** {s['raw_findings']} raw → {s['merged_findings']} merged → "
        f"{s['findings_kept']} kept ({s['findings_refuted']} refuted by verification)"
    )
    if not s["panel_quorum_ok"]:
        lines.append("")
        lines.append(
            "> ⚠️ **Not an independent review.** The surviving critics span fewer "
            "than 2 provider domains, so they share a gateway/credential and fail "
            "(and agree) for correlated reasons."
        )
    lines.append("")

    sev = s["by_severity"]
    lines.append(
        "| critical | high | medium | low |\n|---|---|---|---|\n"
        f"| {sev['critical']} | {sev['high']} | {sev['medium']} | {sev['low']} |"
    )
    lines.append("")
    lines.append(f"**Next action:** {s['recommended_next_action']}")
    lines.append("")

    kept = [f for f in result["findings"] if f["status"] != "refuted"]
    if kept:
        lines.append("## Findings")
        lines.append("")
        for i, f in enumerate(kept, 1):
            lines.extend(_render_finding(f, i))
    else:
        lines.append("## Findings")
        lines.append("")
        lines.append("_Nothing survived verification._")
        lines.append("")

    overflow = result.get("unverified_findings") or []
    if overflow:
        lines.append(f"## Unverified (over the cap — {len(overflow)})")
        lines.append("")
        for f in overflow:
            lines.append(
                f"- **[{f['severity'].upper()}] {f['title']}** — {f['location']}: {f['claim']}"
            )
        lines.append("")

    refuted = [f for f in result["findings"] if f["status"] == "refuted"]
    if refuted:
        lines.append(f"## Refuted ({len(refuted)})")
        lines.append("")
        lines.append("_Dropped by adversarial verification. Listed so a real defect "
                     "wrongly refuted is still visible._")
        lines.append("")
        for f in refuted:
            why = next(
                (v.get("reasoning") for v in f.get("verdicts", [])
                 if v.get("status") == "ok" and v.get("refuted")),
                "",
            )
            lines.append(f"- **{f['title']}** ({f['location']}) — {why}")
        lines.append("")

    if s["failed_critics"]:
        lines.append("## Failed critics")
        lines.append("")
        for c in s["failed_critics"]:
            lines.append(
                f"- `{c['lens']}` on {c['model']} ({c['id']}) — "
                f"{c['failure_reason']}: {c['error']}"
            )
        lines.append("")

    if result.get("notes"):
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {n}" for n in result["notes"])
        lines.append("")

    lines.append("---")
    lines.append(
        f"_{usage.get('llm_calls', 0)} LLM calls · "
        f"{usage.get('tokens_in', 0)}→{usage.get('tokens_out', 0)} tokens · "
        f"{usage.get('web_search_calls', 0)} web searches "
        f"(${usage.get('web_search_cost_usd', 0)})._ "
        "Verification filters model noise; it does not establish correctness — "
        "confirm every finding against the code before acting."
    )
    return "\n".join(lines)
