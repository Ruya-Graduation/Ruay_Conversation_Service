"""
app/database.py
───────────────
Async MongoDB helpers (Motor driver).

Connection lifecycle is managed by FastAPI's lifespan context; the Motor
client is created once at startup and closed on shutdown.
"""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import BulkWriteError

logger = logging.getLogger(__name__)


def create_motor_client(uri: str) -> AsyncIOMotorClient:
    """Create and return a Motor client. Call once at startup."""
    return AsyncIOMotorClient(uri)


# # ── Ingestion Helpers [DISABLED for Ingestion] ─────────────────────────────
# async def ensure_indexes(collection: AsyncIOMotorCollection) -> None:
#     """
#     Ensure a partial unique index exists on chunk_id.
#     """
#     await collection.create_index(
#         "chunk_id",
#         unique=True,
#         partialFilterExpression={"chunk_id": {"$exists": True, "$type": "string"}},
#     )
#     logger.info("MongoDB partial unique index on 'chunk_id' (string values only) is ready.")
#
#
# async def check_article_exists(
#     collection: AsyncIOMotorCollection,
#     article_id: str,
# ) -> bool:
#     """
#     Return True if any chunk with the given article_id already exists.
#     Used to reject duplicate uploads before attempting any inserts.
#     """
#     doc = await collection.find_one({"article_id": article_id}, projection={"_id": 1})
#     return doc is not None
#
#
# async def insert_chunks(
#     collection: AsyncIOMotorCollection,
#     chunks: list[dict[str, Any]],
# ) -> int:
#     """
#     Insert a batch of chunks.
#     """
#     if not chunks:
#         return 0
#
#     try:
#         result = await collection.insert_many(chunks, ordered=False)
#         return len(result.inserted_ids)
#     except BulkWriteError as bwe:
#         inserted = bwe.details.get("nInserted", 0)
#         n_errors = len(bwe.details.get("writeErrors", []))
#         logger.warning(
#             "BulkWriteError: %d inserted, %d duplicates/errors skipped.",
#             inserted,
#             n_errors,
#         )
#         return inserted


async def vector_search(
    collection: AsyncIOMotorCollection,
    query_embedding: list[float],
    index_name: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Run a MongoDB Atlas $vectorSearch against the 'embedding' field.

    Returns up to `top_k` chunks ordered by cosine similarity score
    (highest first).  Each result contains only the fields needed by
    the prompt builder: chunk_id, title, text, page_start, page_end, score.

    Requirements:
        - Your Atlas cluster must have a Vector Search index named `index_name`
          defined on the 'embedding' field of this collection.
        - The index must use the same dimension and similarity metric
          as the embeddings produced by the HuggingFace model.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                # Atlas recommendation: numCandidates = limit * 10 for good recall
                "numCandidates": top_k * 10,
                "limit": top_k,
            }
        },
        {
            # Return only fields the prompt builder needs; exclude raw embedding
            # to keep the payload small.
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "title": 1,
                "text": 1,
                "page_start": 1,
                "page_end": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    cursor = collection.aggregate(pipeline)
    results = await cursor.to_list(length=top_k)
    logger.info("Vector search returned %d chunks (index='%s').", len(results), index_name)
    return results
