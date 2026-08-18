"""
app/prompt_builder.py
─────────────────────
★ THIS IS THE SINGLE PLACE TO EDIT ALL PROMPT LOGIC ★

Three public functions:

  build_query_text(artifact, question)
      → The string that gets embedded for vector search.
        Changing this changes WHAT the DB retrieves.

  build_system_prompt()
      → The system prompt that defines the assistant's behavior/tone.
        This is SEPARATE from the artifact data and passages.

  build_user_message(artifact, question, chunks)
      → Builds the final user message containing:
        - The actual question
        - Artifact context
        - Retrieved knowledge passages
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
# 2.  System prompt  (defines behavior and tone - NOT the data)
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """
    Build the system prompt that defines HOW the assistant should behave.
    
    This should contain:
      • Role definition
      • Tone and style instructions
      • General answering guidelines
    
    This should NOT contain:
      • Artifact data (goes in user message)
      • Retrieved passages (goes in user message)
      • The user's question (goes in user message)
    
    ✏️  Edit this function to change the model's persona, tone, and 
        general behavior.
    """
    
    return (
       ''' You are Ruya, an AI cultural heritage assistant.

Your job is to help users learn about Egyptian cultural heritage
and artifacts.

Use the provided context as the primary and authoritative source
for factual answers.

Rules:
- Do not invent facts that are not supported by the provided context.
- If the provided context does not contain enough information to answer
  the question, clearly say that you don't have enough information.
- Do not present assumptions or guesses as facts.
- Answer the user's actual question directly.
- Maintain a friendly, intimate, warm storytelling style, like a nerdy
  friend who is genuinely excited to share interesting historical facts.'''
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  User message content  (question + artifact + passages)
# ─────────────────────────────────────────────────────────────────────────────

def build_user_message(
    artifact: ArtifactContext,
    question: str,
    chunks: list[dict],
) -> str:
    """
    Build the content for the final user message that will be sent to the LLM.
    
    This combines:
      • The user's actual question
      • Artifact metadata
      • Retrieved knowledge passages from the database
    
    This gets placed as the last message in the messages array with role="user".
    
    ✏️  Edit this function to change how the artifact data and retrieved 
        passages are formatted in the user message.
    
    Args:
        artifact:  The ArtifactContext from the request.
        question:  The user's current question.
        chunks:    List of dicts returned by vector_search() — each has
                   keys: chunk_id, title, text, page_start, page_end, score.
    """
    
    parts: list[str] = []
    
    # ── Artifact Context ──────────────────────────────────────────────────────
    artifact_lines = [f"Name: {artifact.name}"]
    
    if artifact.period:
        artifact_lines.append(f"Period: {artifact.period}")
    
    if artifact.material:
        artifact_lines.append(f"Material: {artifact.material}")
    
    if artifact.place_of_discovery:
        artifact_lines.append(f"Place of discovery: {artifact.place_of_discovery}")
    
    parts.append("**Artifact Information:**")
    parts.append("\n".join(artifact_lines))
    parts.append("")  # blank line
    
    # ── Retrieved Knowledge Passages ──────────────────────────────────────────
    if chunks:
        parts.append("**Relevant Knowledge from UEE Database:**")
        parts.append("")
        
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
            
            parts.append(", ".join(header_parts))
            parts.append(text)
            parts.append("")  # blank line between passages
    else:
        parts.append("**Relevant Knowledge:**")
        parts.append("No relevant passages were retrieved from the database for this query.")
        parts.append("")
    
    # ── User's Question ───────────────────────────────────────────────────────
    parts.append("**Question:**")
    parts.append(question)
    
    return "\n".join(parts)
