"""Critique lens registry — the "different angles" half of adversarial critique.

A lens is a REVIEW MANDATE, not a personality. Each entry carries a `focus`
(what this critic must hunt) and an explicit `out_of_scope` (what it must leave
to the other critics). The out_of_scope half is load-bearing: without it every
critic drifts to the same generic "here are some bugs and also consider adding
tests" answer, and N critics collapse into one opinion with N latencies. Keeping
the mandates disjoint is what makes the fan-out actually independent.

Lens ≠ model. `assign_lenses` pairs lenses with council members so that, as far
as the member list allows, each lens sits on a DIFFERENT provider domain —
correlated failure (one gateway, one key) is the thing that quietly turns an
"independent" panel into a single source (see models.provider_domain).
"""

from __future__ import annotations

from models import provider_domain

# `focus` — imperative bullets, injected verbatim into the critic system prompt.
# `out_of_scope` — what this lens must NOT report (another lens owns it).
LENSES: dict[str, dict] = {
    "correctness": {
        "title": "Correctness",
        "focus": (
            "- Logic that produces a wrong result for some reachable input.\n"
            "- Off-by-one, inverted conditions, wrong operator, wrong default.\n"
            "- Unhandled branches: None/null, empty collection, zero, negative.\n"
            "- State that can go stale, or an invariant the code assumes but never enforces.\n"
            "- Error paths that swallow a failure and continue with bad data."
        ),
        "out_of_scope": (
            "security, performance, concurrency, test coverage, style/naming"
        ),
    },
    "security": {
        "title": "Security",
        "focus": (
            "- Untrusted input reaching a sink: shell, SQL, path, deserialization, template.\n"
            "- Secrets/credentials in logs, error strings, dumps, or outbound payloads.\n"
            "- Missing or bypassable authz/authn checks; a boundary that is documented\n"
            "  but not enforced in code.\n"
            "- Fail-OPEN defaults where the safe default is fail-closed.\n"
            "- Prompt injection: attacker-controlled text treated as instructions."
        ),
        "out_of_scope": (
            "general correctness bugs with no attacker, performance, style"
        ),
    },
    "concurrency": {
        "title": "Concurrency & state",
        "focus": (
            "- Shared mutable state touched from more than one task/thread.\n"
            "- Check-then-act races; state read outside the lock that guards it.\n"
            "- await points that split what the code assumes is one atomic step.\n"
            "- Cancellation: work left half-done, locks/handles leaked on CancelledError.\n"
            "- Unbounded queues/fan-out, missing backpressure, deadlock-capable lock order."
        ),
        "out_of_scope": "single-threaded logic bugs, security, style",
    },
    "failure-modes": {
        "title": "Failure modes & blast radius",
        "focus": (
            "- What happens when a dependency is slow, down, or returns garbage.\n"
            "- Missing timeout, unbounded retry, retry of a non-idempotent operation.\n"
            "- Partial failure: the operation half-applied, no rollback, no reconciliation.\n"
            "- Irreversible actions (delete/overwrite/deploy) without a guard.\n"
            "- Recovery after restart: in-flight work silently lost or double-executed."
        ),
        "out_of_scope": "happy-path logic bugs, micro-performance, style",
    },
    "performance": {
        "title": "Performance & resource use",
        "focus": (
            "- Work that grows superlinearly with input the caller controls.\n"
            "- Repeated I/O inside a loop where one batched call would do.\n"
            "- Data read fully into memory when it can be arbitrarily large.\n"
            "- Caches with no bound or no eviction; leaks of connections/handles/tasks.\n"
            "- Hot-path allocation or serialization that is trivially avoidable."
        ),
        "out_of_scope": "correctness, security, style, test coverage",
    },
    "data-integrity": {
        "title": "Data integrity & migration",
        "focus": (
            "- Writes that can leave persisted state inconsistent if interrupted.\n"
            "- Ordering bugs where a delete/overwrite precedes the durable write.\n"
            "- Schema/format changes without a read path for the old shape.\n"
            "- Silent truncation, lossy coercion, encoding assumptions.\n"
            "- Idempotency: replaying the same input produces different stored state."
        ),
        "out_of_scope": "in-memory-only logic, performance, style",
    },
    "api-contract": {
        "title": "API contract & compatibility",
        "focus": (
            "- Behaviour that contradicts the docstring, README, or type signature.\n"
            "- A breaking change to a signature, return shape, or error type that\n"
            "  existing callers depend on.\n"
            "- Error contract: raising where callers expect a value, or vice versa.\n"
            "- Defaults that change semantics for callers who pass nothing.\n"
            "- Names that promise something the implementation does not do."
        ),
        "out_of_scope": "internal implementation quality, performance, security",
    },
    "simplicity": {
        "title": "Simplicity & over-engineering",
        "focus": (
            "- Abstraction, indirection, or configurability with exactly one use.\n"
            "- Code paths that cannot be reached, or handle impossible states.\n"
            "- Duplicated logic that already exists elsewhere in the codebase.\n"
            "- A simpler formulation that is strictly shorter AND clearer — say what.\n"
            "- Changes beyond what the stated goal required."
        ),
        "out_of_scope": "bugs, security, performance numbers, formatting preferences",
    },
    "testing": {
        "title": "Test coverage & verifiability",
        "focus": (
            "- Behaviour a test asserts nothing about, where a regression would be silent.\n"
            "- Tests that pass for the wrong reason (over-mocked, tautological assertion).\n"
            "- Missing cases for the boundaries the code itself branches on.\n"
            "- Behaviour that cannot be tested at all without a refactor — say which.\n"
            "- Flakiness: reliance on wall-clock time, ordering, network, or shared disk."
        ),
        "out_of_scope": "production-code style, performance, security",
    },
    "observability": {
        "title": "Observability & operability",
        "focus": (
            "- A failure that would leave no trace an operator could find.\n"
            "- Errors logged without the identifiers needed to act on them.\n"
            "- Log/metric volume that would be unusable (per-item spam) or absent.\n"
            "- Sensitive data written to logs or error text.\n"
            "- No way to tell, from outside, whether the component is healthy."
        ),
        "out_of_scope": "correctness, performance, style",
    },
}

# Named bundles. Explicit lists (not computed) so editing one never silently
# reshuffles another — same rationale as models.PRESETS.
LENS_PRESETS: dict[str, list[str]] = {
    "code-review": [
        "correctness", "security", "concurrency",
        "failure-modes", "simplicity", "testing",
    ],
    "security-audit": ["security", "data-integrity", "failure-modes", "api-contract"],
    "design-review": [
        "correctness", "simplicity", "api-contract", "failure-modes", "observability",
    ],
    "reliability": ["failure-modes", "concurrency", "data-integrity", "observability"],
    "fast-3": ["correctness", "security", "failure-modes"],
}

DEFAULT_LENS_PRESET = "code-review"

# Two critics is the floor: one "critic" is just model_ask with a themed prompt,
# and there is nothing for the cross-lens dedup/verify machinery to do.
MIN_LENSES = 2


class UnknownLensError(RuntimeError):
    """Raised when a lens id is not present in LENSES."""


class UnknownLensPresetError(RuntimeError):
    """Raised when a lens preset name is not in LENS_PRESETS."""


def resolve_lenses(lenses: list[str] | None, preset: str | None) -> list[str]:
    """Resolve the effective lens list. At most one of `lenses`/`preset`.

    None/None → DEFAULT_LENS_PRESET. Duplicates are dropped (a lens twice would
    double-count as two "independent" critics raising the same finding).
    """
    if preset is not None:
        if lenses is not None:
            raise RuntimeError("pass either lenses or lenses_preset, not both")
        if preset not in LENS_PRESETS:
            raise UnknownLensPresetError(
                f"unknown lenses_preset: '{preset}'. Available: {sorted(LENS_PRESETS)}"
            )
        names = list(LENS_PRESETS[preset])
    elif lenses is not None:
        names = lenses
    else:
        names = list(LENS_PRESETS[DEFAULT_LENS_PRESET])

    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in LENSES:
            raise UnknownLensError(
                f"unknown lens: '{n}'. Available: {sorted(LENSES)}"
            )
        if n not in seen:
            seen.add(n)
            unique.append(n)
    if len(unique) < MIN_LENSES:
        raise RuntimeError(
            f"critique requires at least {MIN_LENSES} distinct lenses, got {unique}"
        )
    return unique


def _domain_interleaved(members: list[dict]) -> list[dict]:
    """Reorder members so consecutive entries come from different provider
    domains where possible (round-robin across domain buckets, preserving the
    caller's order inside each bucket).

    With 5 OCG members and 1 Gemini, the naive order hands lenses 1-5 to a single
    credential domain; interleaving puts the cross-domain member second, so a
    2-lens critique already spans two independent failure domains.
    """
    buckets: dict[str, list[dict]] = {}
    for m in members:
        buckets.setdefault(provider_domain(m["id"]), []).append(m)
    order = list(buckets)  # dict preserves first-seen order → deterministic
    out: list[dict] = []
    while any(buckets[d] for d in order):
        for d in order:
            if buckets[d]:
                out.append(buckets[d].pop(0))
    return out


def assign_lenses(lens_ids: list[str], members: list[dict]) -> list[dict]:
    """Pair each lens with a member: [{"lens": id, "member": cfg}, ...].

    Round-robin over the domain-interleaved member list, so lenses are spread
    across provider domains before any domain is reused. When lenses outnumber
    members a model legitimately carries more than one lens — the pairing is
    still unique per (lens, model), which is what the dedup/verify stages key on.
    """
    if not members:
        raise RuntimeError("assign_lenses: no members")
    ordered = _domain_interleaved(members)
    return [
        {"lens": lens, "member": ordered[i % len(ordered)]}
        for i, lens in enumerate(lens_ids)
    ]
