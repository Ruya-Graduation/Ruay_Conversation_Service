"""
app/main.py
───────────
FastAPI application entry-point.

Startup (lifespan):
  1. Validate required config values.
  2. Initialize Chonkie SemanticChunker (heavy, shared across requests).
  3. Initialize HuggingFace InferenceClient.
  4. Connect to MongoDB and ensure indexes.

Endpoints:
  POST /ingest        – Accept a PDF upload, extract → embed → store.
  POST /conversation  – Artifact + history + question → RAG → LLM answer.
  GET  /health        – Simple liveness check.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient

# Make repo root importable (extract_uee_articles_v3 lives there).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extract_uee_articles_v3 import create_chonkie_semantic_chunker  # noqa: E402

from app.config import settings
from app.database import (
    check_article_exists,
    create_motor_client,
    ensure_indexes,
    insert_chunks,
    vector_search,
)
from app.embedder import embed_chunks
from app.extractor import extract_chunks
from app.chat_client import call_chat, extract_answer
from app.prompt_builder import build_query_text, build_system_prompt
from app.models import (
    ConversationRequest,
    ConversationResponse,
    RetrievedChunk,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application state (shared across requests) ────────────────────────────────

class AppState:
    chunker: Any = None
    hf_client: InferenceClient | None = None
    mongo_client: Any = None
    collection: Any = None


app_state = AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""

    # ── Validate required secrets ─────────────────────────────────────────────
    missing: list[str] = []
    if not settings.hf_token:
        missing.append("HF_TOKEN")
    if not settings.mongodb_uri:
        missing.append("MONGODB_URI")
    if not settings.sbg_api_key:
        missing.append("SBG_API_KEY")
    if not settings.sbg_base_url:
        missing.append("SBG_BASE_URL")
    if missing:
        logger.critical(
            "Missing required environment variables: %s. "
            "Set them in .env or your environment and restart.",
            ", ".join(missing),
        )
        raise RuntimeError(
            f"Missing required config: {', '.join(missing)}"
        )

    # ── Chonkie SemanticChunker ───────────────────────────────────────────────
    logger.info(
        "Initializing Chonkie SemanticChunker with model '%s'...",
        settings.chunker_embedding_model,
    )
    app_state.chunker = create_chonkie_semantic_chunker(
        embedding_model_name=settings.chunker_embedding_model,
        similarity_level=settings.similarity_level,
        min_chunk_chars=settings.min_chunk_chars,
        max_chunk_chars=settings.max_chunk_chars,
    )
    logger.info("Chonkie SemanticChunker ready.")

    # ── HuggingFace InferenceClient ───────────────────────────────────────────
    logger.info(
        "Initializing HuggingFace InferenceClient (model: %s)...",
        settings.hf_embedding_model,
    )
    app_state.hf_client = InferenceClient(
        provider="hf-inference",
        api_key=settings.hf_token,
    )
    logger.info("HuggingFace InferenceClient ready.")

    # ── MongoDB ───────────────────────────────────────────────────────────────
    logger.info("Connecting to MongoDB...")
    app_state.mongo_client = create_motor_client(settings.mongodb_uri)
    app_state.collection = app_state.mongo_client[settings.mongodb_db][
        settings.mongodb_collection
    ]
    await ensure_indexes(app_state.collection)
    logger.info(
        "MongoDB connected → %s / %s",
        settings.mongodb_db,
        settings.mongodb_collection,
    )

    logger.info("🚀 UEE Ingestion API is ready.")

    yield  # ── app is running ──────────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if app_state.mongo_client:
        app_state.mongo_client.close()
        logger.info("MongoDB connection closed.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="UEE Ingestion & Conversation API",
    description=(
        "Upload PDF articles for ingestion into a RAG knowledge base, "
        "then query that knowledge base with artifact context to get "
        "AI-powered answers grounded in the UEE corpus."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe."""
    return {"status": "healthy"}


@app.post("/ingest", tags=["Ingestion"])
async def ingest(file: UploadFile = File(...)):
    """
    Ingest a single PDF file.

    Steps:
      1. Validate the upload is a PDF.
      2. Check for duplicate article (reject if already in DB).
      3. Extract text + semantic chunks via Chonkie.
      4. Generate HuggingFace embeddings for each chunk (max 4 concurrent calls).
      5. Insert chunks (with embeddings) into MongoDB.

    Returns a summary of what was stored.
    """

    # ── 1. Validate content type ──────────────────────────────────────────────
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Also accept by extension in case content-type is generic.
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are accepted. Please upload a .pdf file.",
            )

    file_name = file.filename or "upload.pdf"
    logger.info("Received upload: '%s'", file_name)

    # ── 2. Read bytes ─────────────────────────────────────────────────────────
    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── 3. Extract chunks (CPU-bound, runs in thread pool) ────────────────────
    try:
        article, chunks = await extract_chunks(
            pdf_bytes=pdf_bytes,
            file_name=file_name,
            chunker=app_state.chunker,
            min_chunk_chars=settings.min_chunk_chars,
            chunk_scope=settings.chunk_scope,
        )
    except Exception as exc:
        logger.error("Extraction failed for '%s': %s", file_name, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF extraction failed: {exc}",
        )

    article_id = article["article_id"]
    logger.info(
        "'%s' → article_id=%s, %d chunks extracted",
        file_name,
        article_id,
        len(chunks),
    )

    # ── 4. Reject duplicates ───────────────────────────────────────────────────
    try:
        already_exists = await check_article_exists(
            app_state.collection, article_id
        )
    except Exception as exc:
        logger.error("MongoDB duplicate check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error during duplicate check: {exc}",
        )

    if already_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Article '{file_name}' (id: {article_id}) has already been ingested. "
                "Duplicate uploads are not allowed. "
                "If you need to re-ingest, delete the existing records first."
            ),
        )

    # ── 5. Embed chunks ────────────────────────────────────────────────────────
    try:
        chunks = await embed_chunks(
            chunks=chunks,
            client=app_state.hf_client,
            model=settings.hf_embedding_model,
            concurrency=settings.hf_embedding_concurrency,
        )
    except Exception as exc:
        logger.error("Embedding failed for '%s': %s", file_name, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"HuggingFace embedding error: {exc}",
        )

    logger.info(
        "Embeddings generated for %d chunks of '%s'.", len(chunks), file_name
    )

    # ── 6. Store in MongoDB ────────────────────────────────────────────────────
    try:
        stored_count = await insert_chunks(app_state.collection, chunks)
    except Exception as exc:
        logger.error("MongoDB insert failed for '%s': %s", file_name, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database insert error: {exc}",
        )

    logger.info(
        "Stored %d / %d chunks for '%s'.", stored_count, len(chunks), file_name
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "file_name": file_name,
            "article_id": article_id,
            "title": article.get("title", ""),
            "page_count": article.get("page_count"),
            "chunks_extracted": len(chunks),
            "chunks_stored": stored_count,
        },
    )


# ───────────────────────────────────────────────────────────────────────────────
# Conversation endpoint
# ───────────────────────────────────────────────────────────────────────────────

@app.post("/conversation", response_model=ConversationResponse, tags=["Conversation"])
async def conversation(body: ConversationRequest):
    """
    Answer a question about an artifact using RAG over the UEE knowledge base.

    Steps:
      1. Build a combined query string from artifact fields + user question.
      2. Embed the query via HuggingFace (same model used at ingest time).
      3. Run MongoDB Atlas $vectorSearch to retrieve top-K relevant chunks.
      4. Build a structured system prompt (see app/prompt_builder.py to edit it).
      5. Call the SBG chat API with the full conversation history + system prompt.
      6. Return the model's answer together with the source chunks.
    """

    # ── 1. Build query text ───────────────────────────────────────────────────────
    query_text = build_query_text(
        artifact=body.artifact,
        question=body.question,
    )
    logger.info(
        "Conversation request | artifact='%s' | query='%s'",
        body.artifact.name,
        query_text[:120],
    )

    # ── 2. Embed the query ───────────────────────────────────────────────────────
    # Reuse the same embedder helper with a single-element list.
    try:
        query_chunk = [{"text": query_text, "chunk_id": "__query__"}]
        await embed_chunks(
            chunks=query_chunk,
            client=app_state.hf_client,
            model=settings.hf_embedding_model,
            concurrency=1,
        )
        query_embedding: list[float] = query_chunk[0].get("embedding", [])
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to embed query: {exc}",
        )

    if not query_embedding:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="HuggingFace returned an empty embedding for the query.",
        )

    # ── 3. Vector search ─────────────────────────────────────────────────────────
    try:
        retrieved = await vector_search(
            collection=app_state.collection,
            query_embedding=query_embedding,
            index_name=settings.mongodb_vector_index,
            top_k=settings.vector_search_top_k,
        )
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search failed: {exc}",
        )

    logger.info("Retrieved %d chunks from vector DB.", len(retrieved))

    # ── 4. Build system prompt ────────────────────────────────────────────────────
    system_prompt = build_system_prompt(
        artifact=body.artifact,
        question=body.question,
        chunks=retrieved,
    )

    # ── 5. Build message list for the LLM ──────────────────────────────────────────
    # Include the full conversation history, then append the current question
    # as the final user turn so the model has complete context.
    llm_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in body.messages
    ]
    llm_messages.append({"role": "user", "content": body.question})

    # ── 6. Call SBG chat API ──────────────────────────────────────────────────────
    raw_response = await call_chat(
        base_url=settings.sbg_base_url,
        api_key=settings.sbg_api_key,
        model_id=settings.llm_model_id,
        system_prompt=system_prompt,
        messages=llm_messages,
        max_tokens=settings.llm_max_tokens,
    )

    answer = extract_answer(raw_response)

    logger.info(
        "Conversation complete | model=%s | chunks_used=%d | answer_chars=%d",
        settings.llm_model_id,
        len(retrieved),
        len(answer),
    )

    # ── 7. Build and return response ───────────────────────────────────────────────
    sources = [
        RetrievedChunk(
            chunk_id=c.get("chunk_id"),
            title=c.get("title"),
            text=c.get("text"),
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            score=c.get("score"),
        )
        for c in retrieved
    ]

    return ConversationResponse(
        answer=answer,
        retrieved_chunks=len(retrieved),
        model_id=settings.llm_model_id,
        sources=sources,
    )
