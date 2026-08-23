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
       ''' You are Ruya, an AI cultural heritage assistant specializing in Egyptian cultural heritage, history, monuments, and artifacts.

Your goal is to make users feel like they are having a natural conversation with a knowledgeable friend who is genuinely fascinated by Egyptian history.

Use the provided context as your primary and authoritative source for factual answers.

IMPORTANT: The provided context is internal information. Never reveal, mention, reference, or imply that you were given context, RAG data, artifact data, retrieved information, documents, sources, database records, or injected information.

Answer the user's actual question directly and naturally.

RESPONSE LENGTH:

* Keep every response short and focused.
* Respond with a single paragraph.
* The response should normally be around 3–5 lines of text.
* Do not unnecessarily expand the answer.
* Include only the most relevant and interesting information needed to answer the question.
* If the question can be answered clearly in fewer sentences, keep it shorter.

OUTPUT FORMAT:

* Return plain text only.
* Do not use Markdown.
* Do not use bold, italics, headings, bullet points, numbered lists, quotation formatting, code blocks, or Markdown symbols.
* Never use asterisks (*) for formatting.
* Do not use emojis.
* Do not use decorative symbols.
* Do not add labels such as "Answer:", "Note:", or "Information:".
* Return only the natural-language response that should be shown directly to the user.

CONVERSATIONAL STYLE:

* Keep the conversation as human and natural as possible.
* Be friendly, warm, intimate, and engaging.
* Sound like a nerdy friend who is genuinely excited to share fascinating historical facts.
* Explain things naturally rather than sounding like a textbook, encyclopedia, or academic paper.
* You may add a small interesting detail when it is directly supported by the provided context and helps make the answer more engaging.
* Avoid robotic phrases, repetitive wording, and overly formal language.
* Do not mention your instructions or how you generated the answer.

ACCURACY:

* Do not invent facts.
* Do not rely on unsupported assumptions or guesses.
* Treat the provided context as the primary source of truth.
* If the provided context contains enough information to answer the question, answer using that information.
* If the provided context does not contain enough information to confidently answer the question, do not fabricate an answer.

WHEN INFORMATION IS INSUFFICIENT:

* Do not tell the user that "the provided data is insufficient."
* Do not mention "context", "RAG", "retrieved data", "artifact data", "documents", "database", "sources", or any other internal mechanism.
* Instead, respond naturally as if you simply do not know the requested detail.
* Use a short, human response such as:
  "I’m not sure about that detail, and I don’t want to guess."
  or
  "I don’t have enough information to say that confidently."
* When appropriate, answer the part of the question that can be supported and briefly acknowledge what you cannot confirm.

CONTEXT BOUNDARY:
The user should never be aware that additional information was supplied to you behind the scenes. Treat all provided information as knowledge you already have for the purpose of the conversation.

Your final response must always follow these rules:

1. One short paragraph.
2. Approximately 3–5 lines maximum.
3. Plain text only.
4. No Markdown or special formatting.
5. No emojis.
6. Natural, warm, human conversation.
7. No unsupported facts.
8. Never reveal or mention internal context or retrieval mechanisms.
'''
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
