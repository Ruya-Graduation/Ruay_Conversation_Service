"""
app/models.py
─────────────
Pydantic request and response schemas for the conversation endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────────────

class ArtifactContext(BaseModel):
    """Structured metadata about the artifact being discussed."""

    name: str = Field(..., description="Name of the artifact, e.g. 'Block Statue'")
    period: str | None = Field(None, description="Historical period, e.g. 'Middle Kingdom'")
    material: str | None = Field(None, description="Material, e.g. 'Granite'")
    place_of_discovery: str | None = Field(
        None, description="Where the artifact was found, e.g. 'Karnak'"
    )


class Message(BaseModel):
    """A single turn in the conversation history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")


class ConversationRequest(BaseModel):
    """Full request body for POST /conversation."""

    artifact: ArtifactContext = Field(
        ..., description="Artifact context to anchor the conversation"
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="Prior conversation history (oldest first)",
    )
    question: str = Field(..., description="The user's current question or prompt")


# ── Outbound ──────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single chunk returned from the vector search, included for transparency."""

    chunk_id: str | None = None
    title: str | None = None
    text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None


class ConversationResponse(BaseModel):
    """Response returned by POST /conversation."""

    answer: str = Field(..., description="The model's reply")
    retrieved_chunks: int = Field(..., description="Number of DB chunks used as context")
    model_id: str = Field(..., description="LLM model that generated the answer")
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="The chunks retrieved from the vector DB (for transparency)",
    )
