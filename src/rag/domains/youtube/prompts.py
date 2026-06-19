"""prompts.py -- Claude system/user prompts for the youtube domain.

Three jobs:
  SINGLE / LONGCTX synthesis -- grounded, cited answer from transcript excerpts.
  MAP extraction             -- pull every instance of X from ONE transcript.
  REDUCE                     -- dedupe/organize per-video extractions into an answer.

Citations use [N] mapped to a transcript excerpt; the model is told to name the
video + timestamp it relies on so the UI can render a deep-link.
"""

from __future__ import annotations

REFUSAL_SENTENCE = "The transcripts don't cover that."

ANSWER_SYSTEM = (
    "You answer questions using ONLY the provided excerpts from YouTube video "
    "transcripts.\n\n"
    "RULES:\n"
    "1. Ground every claim in the excerpts. Do not add facts from outside them.\n"
    "2. Follow each claim with a citation in square brackets like [1] or [2,3].\n"
    "3. Each excerpt is labeled with its channel, video title, and timestamp. "
    "When you cite, name the video and timestamp in prose, e.g. \"In "
    "'<title>' (12:30) [2], ...\".\n"
    "4. Transcripts are auto-generated and may have errors; read for meaning, "
    "don't over-read a single odd word.\n"
    "5. Be concise and concrete. Quote a short phrase when the exact wording "
    "matters.\n"
    "6. If the excerpts genuinely don't address the question, respond exactly: "
    "\"" + REFUSAL_SENTENCE + "\"\n"
)

ANSWER_USER_TEMPLATE = (
    "Question: {query}\n\n"
    "Transcript excerpts (numbered):\n{chunks_block}\n\n"
    "Answer the question, citing excerpt numbers like [1] or [2,3]."
)

# --- map-reduce (exhaustive queries) ---

MAP_SYSTEM_TEMPLATE = (
    "You extract structured information from ONE YouTube video transcript.\n\n"
    "TASK: Find EVERY instance of: {target}\n\n"
    "Be exhaustive -- scan the entire transcript, do not stop at the first few. "
    "For each instance output an object with:\n"
    "  - \"claim\": a concise statement of the item, in your words\n"
    "  - \"quote\": a short verbatim quote from the transcript supporting it\n"
    "  - \"approx_topic\": 2-5 word topic tag\n"
    "If there are none, return an empty list. Output ONLY a JSON object of the "
    "form {{\"items\": [ ... ]}} -- no prose, no markdown fences."
)

MAP_USER_TEMPLATE = (
    "Video: {title} (channel: {channel})\n"
    "Transcript:\n{transcript}\n\n"
    "Return the JSON object now."
)

REDUCE_SYSTEM_TEMPLATE = (
    "You synthesize a final answer from per-video extractions across many "
    "YouTube videos by the same source.\n\n"
    "The user asked: {query}\n\n"
    "You are given a JSON list of items, each tagged with the video it came "
    "from (title + url + timestamp). Your job:\n"
    "1. Deduplicate items that say the same thing across videos.\n"
    "2. Organize them clearly (group or order as the question implies -- e.g. "
    "chronologically, by theme, or by whether a prediction came true).\n"
    "3. For each item, cite the source video and timestamp as a markdown link "
    "using the provided url.\n"
    "4. Be exhaustive but readable. Do not invent items beyond those provided.\n"
)


def build_chunks_block(chunks: list[dict], label_fn, budget_chars: int = 1800) -> str:
    """Numbered, labeled excerpt block for the synthesis prompt."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        text = " ".join((c.get("text") or "").split())
        if len(text) > budget_chars:
            text = text[: budget_chars - 3].rstrip() + "..."
        lines.append('[%d] %s\n"%s"' % (i, label_fn(c), text))
    return "\n\n".join(lines)
