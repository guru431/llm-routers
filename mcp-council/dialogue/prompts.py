"""Prompt templates for dialogue modes (debate, panel, socratic).

All renderers return a single string (the full user-message content for the
LLM call). They are pure functions — no I/O, no LLM calls.

History is always truncated to last HISTORY_TRUNCATE_ROUNDS rounds to keep
prompts bounded.
"""

from __future__ import annotations

HISTORY_TRUNCATE_ROUNDS = 10


def format_history_section(history: list[dict]) -> str:
    """Group history entries by round and render as readable text.

    Only the last HISTORY_TRUNCATE_ROUNDS rounds are included. Within a round,
    critiques come before responses (chronological).
    """
    if not history:
        return ""
    rounds_seen = sorted({h["round"] for h in history})
    if len(rounds_seen) > HISTORY_TRUNCATE_ROUNDS:
        keep = set(rounds_seen[-HISTORY_TRUNCATE_ROUNDS:])
        history = [h for h in history if h["round"] in keep]
        rounds_seen = sorted(keep)

    phase_order = {"critique": 0, "response": 1, "question": 0, "answer": 1}
    lines: list[str] = []
    for rn in rounds_seen:
        lines.append(f"ROUND {rn}:")
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
            lines.append(f"  [{h['id']}]{phase_marker}: {h['text']}")
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
) -> str:
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
    """Ask the moderator to write a final summary of the dialogue."""
    hist_text = format_history_section(history)
    return (
        f"You are the moderator for a {mode} dialogue. Below is the full transcript. "
        "Write a final summary in 3-6 short paragraphs covering:\n"
        "1. The strongest 1-2 points each participant made.\n"
        "2. Areas where participants converged (if any).\n"
        "3. Areas that remain unresolved or in genuine disagreement.\n"
        "4. What a reader should take away.\n\n"
        f"=== TOPIC ===\n{topic}\n\n"
        f"=== FULL TRANSCRIPT ===\n{hist_text}\n"
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
    lines.append('{"score": <int 0-10>, "agreers": [<participant_id>, ...], "reasoning": "<one sentence>"}')
    lines.append('Where "agreers" is the list of participant ids that converged into the same view (empty if all distinct).')
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
