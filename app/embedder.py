"""
app/embedder.py
───────────────
Generates embeddings for text chunks via the HuggingFace Inference API.

The HF InferenceClient is synchronous, so each call is run inside
asyncio.to_thread.  A semaphore limits the number of concurrent HF calls
to avoid hitting rate limits (default: 4, configured via settings).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _embed_sync(text: str, client: Any, model: str) -> list[float]:
    """
    Call HuggingFace feature_extraction synchronously.
    Returns a flat list[float] regardless of whether the API returns a
    nested list (per-token) or a single vector.
    """
    result = client.feature_extraction(text, model=model)

    # Flatten: some models return shape (1, dim) or (seq_len, dim).
    # We want a single 1-D vector – take the mean over the first axis when needed.
    if hasattr(result, "tolist"):
        result = result.tolist()

    if not result:
        return []

    # Already a flat list of floats?
    if isinstance(result[0], (int, float)):
        return [float(v) for v in result]

    # Nested list → mean-pool over rows (e.g. per-token embeddings)
    if isinstance(result[0], (list, tuple)):
        import statistics

        n_dims = len(result[0])
        pooled = [
            statistics.mean(row[i] for row in result) for i in range(n_dims)
        ]
        return pooled

    return [float(v) for v in result]


async def embed_chunks(
    chunks: list[dict],
    client: Any,
    model: str,
    concurrency: int = 4,
) -> list[dict]:
    """
    Embed every chunk's 'text' field concurrently and attach the result as
    an 'embedding' key.  Returns the same list with embeddings added in-place.

    Args:
        chunks:      List of chunk dicts (must each have a 'text' key).
        client:      Initialized huggingface_hub.InferenceClient.
        model:       HuggingFace model ID for feature extraction.
        concurrency: Max parallel HF calls (default 4).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _embed_one(chunk: dict) -> None:
        async with semaphore:
            text = chunk.get("text", "")
            if not text:
                chunk["embedding"] = []
                return
            try:
                embedding = await asyncio.to_thread(_embed_sync, text, client, model)
                chunk["embedding"] = embedding
            except Exception as exc:
                logger.error(
                    "Embedding failed for chunk '%s': %s",
                    chunk.get("chunk_id", "?"),
                    exc,
                )
                raise

    await asyncio.gather(*[_embed_one(chunk) for chunk in chunks])
    return chunks
