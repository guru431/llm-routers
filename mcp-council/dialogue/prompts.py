"""Prompt templates for dialogue modes (debate, panel, socratic).

All renderers return a single string (the full user-message content for the
LLM call). They are pure functions — no I/O, no LLM calls.

History is always truncated to last HISTORY_TRUNCATE_ROUNDS rounds to keep
prompts bounded.
"""

from __future__ import annotations

HISTORY_TRUNCATE_ROUNDS = 10

# Within the kept round-window, only the most recent rounds are sent verbatim;
# in older rounds each entry's text is capped. Without this, a 7-participant
# panel at round 10 carries ~140 entries of up to max_tokens (4096) each —
# hundreds of thousands of tokens in every per-round prompt, growing quadratically.
RECENT_FULL_ROUNDS = 2
OLD_ROUND_ENTRY_CHAR_CAP = 1000

# Token-aware ceiling on the rendered history. The round-window + per-entry char
# caps above bound a single prompt, but a wide panel (7 participants × several
# entries/round) can still assemble a very large history within the window. This
# is a HARD token budget on the rendered section: oldest rounds beyond the recent
# verbatim window are dropped (with a rolling-summary marker) until the estimate
# fits. Set high so typical/short dialogues are unaffected — it only bites long,
# wide panels that would otherwise overflow a model's context.
HISTORY_TOKEN_BUDGET = 24000


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Dependency-free — good enough to
    bound a prompt without pulling in a tokenizer."""
    return (len(text) + 3) // 4


def _cap_text(text: str, cap: int | None) -> str:
    if cap is None or len(text) <= cap:
        return text
    return text[:cap].rstrip() + f"… [truncated {len(text) - cap} chars]"


def format_history_section(
    history: list[dict],
    *,
    recent_full_rounds: int = RECENT_FULL_ROUNDS,
    old_entry_char_cap: int | None = OLD_ROUND_ENTRY_CHAR_CAP,
    max_history_tokens: int | None = HISTORY_TOKEN_BUDGET,
) -> str:
    """Group history entries by round and render as readable text.

    Only the last HISTORY_TRUNCATE_ROUNDS rounds are included. Within that
    window, the last `recent_full_rounds` rounds are verbatim; older rounds
    have each entry capped to `old_entry_char_cap` chars (None = no cap).
    Within a round, critiques come before responses (chronological).

    `max_history_tokens` (None = unbounded): a HARD token budget on the rendered
    section. When exceeded, the OLDEST rounds outside the recent verbatim window
    are dropped and replaced by a one-line rolling-summary marker until the
    estimate fits — the recent verbatim rounds are always kept so the live
    argument is never truncated away.
    """
    if not history:
        return ""
    rounds_seen = sorted({h["round"] for h in history})
    if len(rounds_seen) > HISTORY_TRUNCATE_ROUNDS:
        keep = set(rounds_seen[-HISTORY_TRUNCATE_ROUNDS:])
        history = [h for h in history if h["round"] in keep]
        rounds_seen = sorted(keep)

    full_rounds = (
        set(rounds_seen[-recent_full_rounds:]) if recent_full_rounds else set(rounds_seen)
    )

    phase_order = {"critique": 0, "response": 1, "question": 0, "answer": 1}

    def _render_round(rn: int) -> list[str]:
        out = [f"ROUND {rn}:"]
        round_items = [h for h in history if h["round"] == rn]
        round_items.sort(key=lambda h: (phase_order.get(h["phase"], 99), h["id"]))
        for h in round_items:
            phase_marker = ""
            if h["phase"] == "critique":
                phase_marker = " (critique)"
            elif h["phase"] == "response":
                phase_marker = ""
            elif h["phase"] == "question":
                phase_marker = " (question)"
            elif h["phase"] == "answer":
                phase_marker = " (answer)"
            elif h["phase"] == "directive":
                phase_marker = " (DIRECTIVE)"
            elif h["phase"] == "reprompt":
                phase_marker = " (reprompt)"
            elif h["phase"] == "moderator_note":
                phase_marker = " (moderator note)"
            text = h["text"] if rn in full_rounds else _cap_text(h["text"], old_entry_char_cap)
            out.append(f"  [{h['id']}]{phase_marker}: {text}")
        return out

    per_round = {rn: _render_round(rn) for rn in rounds_seen}

    # Token-aware trimming: keep from NEWEST backward while the estimate fits.
    # Rounds inside the recent verbatim window are always kept (the live argument
    # must survive); only older rounds are eligible to drop.
    if max_history_tokens is not None:
        kept_rounds: list[int] = []
        running = 0
        dropped_old = 0
        for rn in reversed(rounds_seen):
            block = "\n".join(per_round[rn])
            cost = estimate_tokens(block)
            if running + cost > max_history_tokens and rn not in full_rounds and kept_rounds:
                dropped_old += 1
                continue
            running += cost
            kept_rounds.append(rn)
        kept_rounds.sort()
        lines: list[str] = []
        if dropped_old:
            lines.append(
                f"[rolling summary] {dropped_old} earlier round(s) omitted to fit "
                "the history token budget; the most recent rounds are shown verbatim below."
            )
        for rn in kept_rounds:
            lines.extend(per_round[rn])
        return "\n".join(lines)

    lines = []
    for rn in rounds_seen:
        lines.extend(per_round[rn])
    return "\n".join(lines)


def _assemble_prompt(
    *,
    role: str,
    topic: str,
    history: list[dict],
    files_section: str | None,
    task: str,
    anti_agreement_rule: str | None,
) -> str:
    parts: list[str] = []
    parts.append("=== ROLE ===")
    parts.append(role)
    parts.append("")
    parts.append("=== TOPIC ===")
    parts.append(topic)
    parts.append("")
    if files_section:
        parts.append(files_section)
        parts.append("")
    hist_text = format_history_section(history)
    if hist_text:
        parts.append("=== DIALOGUE HISTORY ===")
        parts.append(hist_text)
        parts.append("")
    parts.append("=== YOUR TASK ===")
    parts.append(task)
    if anti_agreement_rule:
        parts.append("")
        parts.append("=== ANTI-AGREEMENT RULE ===")
        parts.append(anti_agreement_rule)
    return "\n".join(parts)


def render_critique_prompt(
    *,
    topic: str,
    role_descriptor: str,
    history: list[dict],
    round_n: int,
    files_section: str | None,
    anti_agreement_rule: str | None,
) -> str:
    task = (
        f"You are entering round {round_n}, phase critique.\n"
        "Look at the most recent responses from other participants in the history.\n"
        "Output two short paragraphs (3-6 sentences each):\n"
        "1. Pick ONE other participant whose argument has the weakest point. "
        "Name them by id and explain precisely what is weak.\n"
        "2. Pick ONE other participant whose argument you find most compelling. "
        "Name them and explain what they got right.\n"
        "Be specific. Do not summarize your own position here — that comes in the response phase."
    )
    return _assemble_prompt(
        role=role_descriptor,
        topic=topic,
        history=history,
        files_section=files_section,
        task=task,
        anti_agreement_rule=anti_agreement_rule,
    )


def render_response_prompt(
    *,
    topic: str,
    role_descriptor: str,
    history: list[dict],
    round_n: int,
    files_section: str | None,
    anti_agreement_rule: str | None,
    opening: bool = False,
) -> str:
    if opening:
        # Round 1 (or any round with no preceding critique): there is nothing to
        # "address" yet, so don't instruct the model to respond to critique it
        # never saw — ask for a clear opening statement instead.
        task = (
            f"You are entering round {round_n}, the OPENING round.\n"
            "No one has spoken yet. State your initial position on the topic in "
            "1-3 short paragraphs: make your strongest concrete argument for your "
            "assigned role/position. Be specific; avoid hedging."
        )
    else:
        task = (
            f"You are entering round {round_n}, phase response.\n"
            "You have just seen critiques from other participants (see history above, "
            "phase=critique entries for the current round).\n"
            "Write your updated position in 1-3 short paragraphs. You MUST:\n"
            "- Address the critique aimed at you (defend or concede a specific point).\n"
            "- Advance your argument — do not merely restate it.\n"
            "Stay in your assigned role/position."
        )
    return _assemble_prompt(
        role=role_descriptor,
        topic=topic,
        history=history,
        files_section=files_section,
        task=task,
        anti_agreement_rule=anti_agreement_rule,
    )


def render_position_split_prompt(*, question: str, n: int) -> str:
    """Ask a moderator model to split `question` into N opposing theses."""
    return (
        "You are a debate moderator. Read the question below and generate "
        f"{n} sharply-opposing positions to be defended by separate debaters.\n"
        "Output ONLY a JSON array of strings, one position per element, no commentary. "
        f"The array MUST have exactly {n} elements.\n"
        "Each position is a single declarative sentence (max 25 words) that a debater can defend.\n"
        "Positions must be genuinely opposing — not slight variations of the same idea.\n\n"
        f"=== QUESTION ===\n{question}\n"
    )


def render_summary_prompt(*, topic: str, history: list[dict], mode: str) -> str:
    """Ask the moderator to write a final summary of the dialogue.

    Summarizing needs the full text of each kept round (no per-entry cap), but
    the round-window cap still applies — so for runs longer than
    HISTORY_TRUNCATE_ROUNDS the transcript is the most-recent rounds, not all of
    them. The header says so rather than mislabelling a partial view as 'full'.
    """
    hist_text = format_history_section(
        history, recent_full_rounds=HISTORY_TRUNCATE_ROUNDS, old_entry_char_cap=None
    )
    total_rounds = len({h["round"] for h in history})
    window_note = (
        f" (most recent {HISTORY_TRUNCATE_ROUNDS} of {total_rounds} rounds)"
        if total_rounds > HISTORY_TRUNCATE_ROUNDS else ""
    )
    return (
        f"You are the moderator for a {mode} dialogue. Below is the transcript"
        f"{window_note}. "
        "Write a final summary in 3-6 short paragraphs covering:\n"
        "1. The strongest 1-2 points each participant made.\n"
        "2. Areas where participants converged (if any).\n"
        "3. Areas that remain unresolved or in genuine disagreement.\n"
        "4. What a reader should take away.\n\n"
        f"=== TOPIC ===\n{topic}\n\n"
        f"=== TRANSCRIPT{window_note} ===\n{hist_text}\n"
    )


def render_diversity_monitor_prompt(*, responses: dict[str, str]) -> str:
    """Ask a cheap model to score how similar the current-round responses are."""
    lines = ["You are a diversity monitor for a multi-model panel discussion."]
    lines.append("Below are responses from this round, one per participant.")
    lines.append("Rate their similarity on a scale of 0-10:")
    lines.append("- 0 = completely different perspectives, no shared conclusions")
    lines.append("- 10 = essentially saying the same thing, just paraphrased")
    lines.append("")
    lines.append("Output a JSON object EXACTLY in this shape:")
    lines.append('{"score": <int 0-10>, "agreers": [<participant_id>, ...], "uncertainty": <float 0-1>, "reasoning": "<one sentence>"}')
    lines.append('Where "agreers" is the list of participant ids that converged into the same view (empty if all distinct).')
    lines.append('"uncertainty" (0-1) is how unsure YOU are about this similarity call — 0 = confident, 1 = guessing.')
    lines.append("")
    lines.append("=== RESPONSES ===")
    for pid, text in responses.items():
        lines.append(f"--- {pid} ---")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def render_socratic_questioner_prompt(
    *,
    topic: str,
    history: list[dict],
    round_n: int,
    files_section: str | None,
) -> str:
    task = (
        f"You are entering round {round_n} as the questioner.\n"
        "Read what the respondent has said (history, phase=answer).\n"
        "Ask ONE deepening question that probes a weak spot, an assumption, or "
        "an interesting consequence of their last answer.\n"
        "Output ONLY the question. No preamble, no commentary. 1-3 sentences."
    )
    return _assemble_prompt(
        role="You are a Socratic questioner. Your job is to deepen understanding by asking sharp, specific questions.",
        topic=topic,
        history=history,
        files_section=files_section,
        task=task,
        anti_agreement_rule=None,
    )


def render_socratic_respondent_prompt(
    *,
    topic: str,
    history: list[dict],
    round_n: int,
    files_section: str | None,
) -> str:
    task = (
        f"You are entering round {round_n} as the respondent.\n"
        "Read the questioner's most recent question (history, phase=question, current round).\n"
        "Answer it directly and substantively in 1-3 short paragraphs. If you cannot "
        "answer with confidence, say so explicitly and explain why."
    )
    return _assemble_prompt(
        role="You are the respondent in a Socratic dialogue. Answer the questioner's questions directly and substantively.",
        topic=topic,
        history=history,
        files_section=files_section,
        task=task,
        anti_agreement_rule=None,
    )
