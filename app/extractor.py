"""
app/extractor.py
────────────────
Thin async wrapper around extract_uee_articles_v3.py.

The CPU-heavy PDF extraction + Chonkie chunking runs in a thread-pool
executor so it never blocks the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

# Re-use the existing extraction logic directly.
# We import only the functions we need; the script's __main__ block is never run.
import sys
import os

# Make sure the repo root is importable as a module.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from extract_uee_articles_v3 import extract_pdf  # noqa: E402


def _make_article_id_from_name(file_name: str) -> str:
    """
    Derive a stable article_id from the uploaded file name.

    Mirrors the logic in extract_uee_articles_v3.make_article_id but works
    with just a file name string (no Path.relative_to() needed).
    """
    stem = Path(file_name).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80] or "article"
    digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def _extract_sync(
    pdf_bytes: bytes,
    file_name: str,
    chunker: Any,
    min_chunk_chars: int,
    chunk_scope: str,
) -> tuple[dict, list[dict]]:
    """
    Synchronous extraction: write bytes to a temp file, run extract_pdf,
    clean up.  Called inside asyncio.to_thread.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        # input_dir is the temp directory; article_id will be derived from the
        # temp path but we'll overwrite it below with the real file name.
        article, chunks = extract_pdf(
            pdf_path=tmp_path,
            input_dir=tmp_path.parent,
            chunker=chunker,
            min_chunk_chars=min_chunk_chars,
            chunk_scope=chunk_scope,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # ── Patch identifiers to reflect the original uploaded file name ──────────
    real_article_id = _make_article_id_from_name(file_name)
    article["article_id"] = real_article_id
    article["source_file"] = file_name
    article["file_name"] = file_name

    for idx, chunk in enumerate(chunks):
        chunk["article_id"] = real_article_id
        chunk["chunk_id"] = f"{real_article_id}:chunk-{idx:04d}"
        chunk["source_file"] = file_name
        chunk["file_name"] = file_name

    return article, chunks


async def extract_chunks(
    pdf_bytes: bytes,
    file_name: str,
    chunker: Any,
    min_chunk_chars: int,
    chunk_scope: str,
) -> tuple[dict, list[dict]]:
    """
    Async entry-point for PDF extraction + chunking.

    Returns:
        (article_dict, list_of_chunk_dicts)
        Chunk dicts do NOT yet contain the 'embedding' field.
    """
    return await asyncio.to_thread(
        _extract_sync,
        pdf_bytes,
        file_name,
        chunker,
        min_chunk_chars,
        chunk_scope,
    )
