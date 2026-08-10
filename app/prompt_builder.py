"""
app/prompt_builder.py
─────────────────────
★ THIS IS THE SINGLE PLACE TO EDIT ALL PROMPT LOGIC ★

Two public functions:

  build_query_text(artifact, question)
      → The string that gets embedded for vector search.
        Changing this changes WHAT the DB retrieves.

  build_system_prompt(artifact, question, chunks)
      → The system prompt injected into every LLM call.
        Changing this changes HOW the model answers.
"""

from __future__ import annotations

from app.models import ArtifactContext


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Query text  (embedded → used for vector search)
# ─────────────────────────────────────────────────────────────────────────────

def build_query_text(artifact: ArtifactContext, question: str) -> str:
    """
    Produce the combined text that gets embedded and used to query the
    vector database.

    The artifact fields are prepended so the search is anchored to the
    artifact's historical/cultural context before the question is considered.

    ✏️  Edit this function to change what the vector search looks for.
    """
    parts: list[str] = ["Artifact:"]

    parts.append(f"Name: {artifact.name}")

    if artifact.period:
        parts.append(f"Period: {artifact.period}")

    if artifact.material:
        parts.append(f"Material: {artifact.material}")

    if artifact.place_of_discovery:
        parts.append(f"Place of discovery: {artifact.place_of_discovery}")

    parts.append(f"Question: {question}")

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  System prompt  (injected into the LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    artifact: ArtifactContext,
    question: str,
    chunks: list[dict],
) -> str:
    """
    Build the full system prompt for the LLM.

    Sections:
      • Role definition
      • Artifact under discussion
      • Retrieved knowledge passages (numbered)
      • Answering instructions

    ✏️  Edit this function to change the model's persona, tone, citation
        style, fallback behaviour, or any other answering rules.

    Args:
        artifact:  The ArtifactContext from the request.
        question:  The user's current question (included for context).
        chunks:    List of dicts returned by vector_search() — each has
                   keys: chunk_id, title, text, page_start, page_end, score.
    """

    # ── Role ──────────────────────────────────────────────────────────────────
    role_section = (
        "You are an expert Egyptologist and archaeologist assistant "
        "specializing in Ancient Egyptian and Nubian artifacts. "
        "You have access to a curated knowledge base from the "
        "UCLA Encyclopedia of Egyptology (UEE). "
        "Your answers are scholarly, precise, and grounded strictly in the "
        "provided context passages. You cite passage numbers when relevant."
    )

    # ── Artifact ──────────────────────────────────────────────────────────────
    artifact_lines = [f"- **Name:** {artifact.name}"]

    if artifact.period:
        artifact_lines.append(f"- **Period:** {artifact.period}")

    if artifact.material:
        artifact_lines.append(f"- **Material:** {artifact.material}")

    if artifact.place_of_discovery:
        artifact_lines.append(f"- **Place of discovery:** {artifact.place_of_discovery}")

    artifact_section = "## Artifact Under Discussion\n" + "\n".join(artifact_lines)

    # ── Retrieved passages ────────────────────────────────────────────────────
    if chunks:
        passage_lines: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title") or "Unknown"
            text = (chunk.get("text") or "").strip()
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")

            # Build a compact header for each passage
            header_parts = [f"[{i}] Source: {title}"]
            if page_start is not None:
                if page_end and page_end != page_start:
                    header_parts.append(f"pp. {page_start}–{page_end}")
                else:
                    header_parts.append(f"p. {page_start}")

            passage_lines.append(", ".join(header_parts))
            passage_lines.append(text)
            passage_lines.append("")  # blank line between passages

        passages_section = (
            "## Relevant Knowledge (retrieved from UEE database)\n"
            + "\n".join(passage_lines).strip()
        )
    else:
        passages_section = (
            "## Relevant Knowledge\n"
            "No relevant passages were retrieved from the database for this query."
        )

    # ── Instructions ──────────────────────────────────────────────────────────
    instructions_section = (
        "## Instructions\n"
        "- Answer the user's question using ONLY the passages above as your source.\n"
        "- If the passages do not contain enough information to answer, "
        "say so clearly and honestly — do not fabricate details.\n"
        "- Cite passage numbers (e.g. [1], [3]) when you draw from them.\n"
        "- Keep your answer focused, scholarly, and concise.\n"
        "- Use plain prose; avoid bullet lists unless specifically helpful."
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    return "\n\n".join(
        [
            role_section,
            artifact_section,
            passages_section,
            instructions_section,
        ]
    )
