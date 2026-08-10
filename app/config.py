"""
app/config.py
─────────────
All configuration is read from environment variables (or a .env file).
Never hard-code secrets here; fill in .env and keep it out of version control.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── HuggingFace ────────────────────────────────────────────────────────────
    hf_token: str = ""
    """HuggingFace API key (HF_TOKEN in env)."""

    hf_embedding_model: str = "microsoft/harrier-oss-v1-0.6b"
    """HuggingFace model used for generating chunk embeddings."""

    hf_embedding_concurrency: int = 4
    """Max parallel HuggingFace feature-extraction calls per ingest request."""

    # ── MongoDB ────────────────────────────────────────────────────────────────
    mongodb_uri: str = ""
    """Full MongoDB connection string (MONGODB_URI in env)."""

    mongodb_db: str = "uee_db"
    """Target database name."""

    mongodb_collection: str = "chunks"
    """Target collection name for storing chunks + embeddings."""

    # ── Chonkie / chunker ──────────────────────────────────────────────────────
    chunker_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    """SentenceTransformer model used internally by Chonkie for semantic splitting.
    This is different from the HF embedding model used for final vector storage."""

    similarity_level: str = "high"
    """Chonkie semantic similarity threshold. One of: low | medium | high."""

    min_chunk_chars: int = 100
    """Chunks shorter than this (in characters) are discarded."""

    max_chunk_chars: int = 1500
    """Soft upper bound on chunk length (in characters)."""

    chunk_scope: str = "full"
    """'full' = semantic chunks across the whole article.
    'page' = semantic chunks per page."""

    # ── SBG Chat API ───────────────────────────────────────────────────────────
    sbg_api_key: str = ""
    """Bearer token for the SBG student chat API (SBG_API_KEY in env)."""

    sbg_base_url: str = ""
    """Base URL for the SBG API, e.g. https://api.example.com/api/v1 (SBG_BASE_URL in env)."""

    llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    """LLM model identifier sent in the chat payload (LLM_MODEL_ID in env)."""

    llm_max_tokens: int = 1024
    """Maximum tokens allowed in the model's reply (LLM_MAX_TOKENS in env)."""

    # ── Vector Search ──────────────────────────────────────────────────────────
    mongodb_vector_index: str = "vector_index"
    """Name of the Atlas Vector Search index on the embedding field (MONGODB_VECTOR_INDEX in env)."""

    vector_search_top_k: int = 10
    """Number of top chunks to retrieve from the vector database (VECTOR_SEARCH_TOP_K in env)."""


# Singleton – import this everywhere instead of re-instantiating.
settings = Settings()
