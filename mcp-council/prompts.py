"""System prompts for council stage 1 (independent) and stage 2 (peer-ranking)."""

STAGE1_SYSTEM = (
    "You are a senior engineer reviewing a technical question. "
    "Answer as thoroughly and concretely as you can. "
    "Other independent experts are answering the same question separately; "
    "do not collaborate or reference them. "
    "Cite specific files, functions, or trade-offs where relevant. "
    "If you are uncertain, say so explicitly rather than guess."
)

STAGE2_SYSTEM = (
    "You are reviewing answers from anonymous experts (Member A, B, C, …) "
    "to a technical question. Rank them by accuracy, depth, and actionable insight.\n"
    "\n"
    "Calibration — use the full 1-10 scale, do NOT default to 8-10:\n"
    "  1-3  fundamentally wrong, misses the question, or contradicts the evidence\n"
    "  4-5  partially correct but shallow, generic, or missing critical aspects\n"
    "  6-7  solid answer that addresses the question well, with minor gaps\n"
    "  8    strong answer — accurate, deep, actionable\n"
    "  9    exceptional — covers something the others missed\n"
    "  10   reserved for genuinely outstanding work; rarely warranted\n"
    "If two answers seem equally good, differentiate them anyway — give one a "
    "lower score and explain what tipped you.\n"
    "\n"
    'reasoning: 3-5 sentences. COMPARE this member to the others where possible '
    '("X covers Y more concretely than Member Z"). State both strengths and a '
    'concrete weakness. Cite specific claims from the answer. Do NOT just say '
    '"comprehensive and well-structured" — that is filler.\n'
    "\n"
    "Return STRICT JSON only, no markdown, no commentary. Schema:\n"
    '{"confidence": 1-10, "rankings": [{"member": "A", "score": 1-10, "reasoning": "..."}, ...]}\n'
    "Include every member you see in the list (you will not see your own answer). "
    "Use integer scores 1-10.\n"
    "\n"
    'confidence (integer 1-10) is YOUR self-rated confidence in this ranking '
    "as a whole — how sure are you that your scores reflect real quality? Low "
    "confidence (1-4) if the question is far outside your expertise or all "
    "answers seem equally good/bad and you're guessing. High confidence (8-10) "
    "only if you genuinely understand the domain and the differences are clear. "
    "This weights your vote in aggregation — be honest, not modest."
)


def build_stage1_user(question: str, files_section: str | None) -> str:
    """Stage 1 user message: optional file context + the question."""
    if files_section:
        return f"{files_section}\n=== QUESTION ===\n{question}"
    return f"=== QUESTION ===\n{question}"


STAGE3_SYSTEM = (
    "You are the chairman of an expert council. Several independent experts "
    "answered a question, then peer-reviewed each other anonymously. Your job "
    "is to produce ONE final answer that combines the strongest, most accurate "
    "points across all the answers — not a summary, but a synthesized "
    "recommendation a user can act on.\n"
    "\n"
    "Rules:\n"
    "1. Lead with the answer itself. No meta-commentary about the council "
    "process unless directly load-bearing.\n"
    "2. Where members agreed on something concrete and correct, state it as "
    "consensus. Where they disagreed, pick a side and say why (cite which "
    "member's reasoning you followed and what tipped the call).\n"
    "3. Where members covered different ground, integrate the complementary "
    "pieces — don't drop high-value insights just because only one member "
    "raised them.\n"
    "4. Where a member was wrong or shallow, do NOT incorporate their point. "
    "The peer rankings are a signal; do not blindly trust your own ranking but "
    "use it as input.\n"
    "5. End with a concrete next-action / prioritized sequence when the "
    "question asks for one.\n"
    "\n"
    "Write as a senior engineer talking to another senior engineer. No filler."
)


def build_stage2_user(
    question: str,
    other_answers: list[tuple[str, str]],
    files_section: str | None,
) -> str:
    """Stage 2 user message: question + anonymized peer answers.

    other_answers = [(pseudonym_letter, answer_text), ...] — answers from peers,
    excluding self. Pseudonym letters vary across rankers (anti-positional-bias).
    """
    parts: list[str] = []
    if files_section:
        parts.append(files_section)
    parts.append("=== ORIGINAL QUESTION ===")
    parts.append(question)
    parts.append("")
    parts.append("=== ANSWERS TO RANK ===")
    for letter, answer in other_answers:
        parts.append(f"\n--- Member {letter} ---")
        parts.append(answer)
    parts.append("")
    parts.append(
        "Now return STRICT JSON ranking these members. Do not include yourself "
        "(your answer is not in the list above)."
    )
    return "\n".join(parts)


STAGE1_ROUND_N_SYSTEM = (
    "You already answered this question in the previous round, then anonymous "
    "experts peer-reviewed all answers. Below are: your previous answer, the "
    "other answers, and the peer-review digest.\n"
    "\n"
    "Write an IMPROVED answer that:\n"
    " - keeps what you got right in round 1;\n"
    " - integrates insights from other members where they were stronger than yours;\n"
    " - addresses concrete criticism from the peer-review digest if it applies "
    "to your answer;\n"
    " - does NOT just concede to the highest-scored answer — re-evaluate based "
    "on substance, the peer scores are imperfect.\n"
    "\n"
    "Stay grounded in the original question and any context files. Be concrete: "
    "name files, functions, trade-offs."
)


def build_stage1_round_n_user(
    question: str,
    own_previous_answer: str,
    other_answers: list[tuple[str, str]],
    rankings_digest: str,
    files_section: str | None,
) -> str:
    """Round-2+ user message for stage 1: prior round outputs + critique digest.

    `own_previous_answer` is the model's own round-(N-1) answer.
    `other_answers` is [(model_label, answer_text), …] from other members in
    round (N-1). Pseudonyms are NOT used here — by this point we want each
    model to consider authorship (e.g. "deepseek already covered X").
    """
    parts: list[str] = []
    if files_section:
        parts.append(files_section)
    parts.append("=== ORIGINAL QUESTION ===")
    parts.append(question)
    parts.append("")
    parts.append("=== YOUR PREVIOUS ANSWER ===")
    parts.append(own_previous_answer)
    parts.append("")
    parts.append("=== OTHER MEMBERS' PREVIOUS ANSWERS ===")
    for label, ans in other_answers:
        parts.append(f"\n--- {label} ---")
        parts.append(ans)
    parts.append("")
    parts.append("=== PEER-REVIEW DIGEST ===")
    parts.append(rankings_digest)
    parts.append("")
    parts.append("Now write your improved answer.")
    return "\n".join(parts)


def build_stage3_user(
    question: str,
    answers: list[tuple[str, str]],
    rankings_summary: str,
    files_section: str | None,
) -> str:
    """Stage 3 user message for the chairman: original question + all answers
    (with model labels for clarity) + a digest of peer rankings.

    answers = [(model_label, answer_text), ...] — typically all stage 1 survivors,
    labelled with their model name for the chairman's benefit (no anonymization
    here — the chairman benefits from knowing which model said what).
    rankings_summary = a short text block summarising the peer-review aggregate.
    """
    parts: list[str] = []
    if files_section:
        parts.append(files_section)
    parts.append("=== ORIGINAL QUESTION ===")
    parts.append(question)
    parts.append("")
    parts.append("=== COUNCIL ANSWERS ===")
    for label, answer in answers:
        parts.append(f"\n--- {label} ---")
        parts.append(answer)
    parts.append("")
    parts.append("=== PEER RANKINGS DIGEST ===")
    parts.append(rankings_summary)
    parts.append("")
    parts.append(
        "Now produce the synthesized final answer per the chairman rules. "
        "Plain markdown. No JSON wrapping."
    )
    return "\n".join(parts)
