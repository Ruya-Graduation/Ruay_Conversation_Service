#!/usr/bin/env python3
"""
Extract cleaned article text and semantic RAG chunks from PDF files.

Uses:
- PyMuPDF for PDF extraction
- Chonkie SemanticChunker for semantic chunking

Outputs:
  output/uee_articles_clean.json
  output/uee_chunks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing dependency. Install with: pip install PyMuPDF")

try:
    from chonkie import SemanticChunker
    CHONKIE_IMPORT_ERROR = None
except Exception as exc:
    SemanticChunker = None
    CHONKIE_IMPORT_ERROR = str(exc)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

PAGE_SEP = "\n\n"

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SOFT_HYPHEN_RE = re.compile(r"\u00ad")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
SINGLE_LINEBREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")


# -------------------------------------------------------------------
# Cleaning helpers
# -------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Remove invisible characters
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u00a0", " ")

    # Remove soft hyphen
    text = SOFT_HYPHEN_RE.sub("", text)

    # Remove control chars
    text = CONTROL_CHARS_RE.sub("", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphen linebreaks
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Normalize spaces
    text = MULTI_SPACE_RE.sub(" ", text)

    # Trim spaces around newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse single linebreaks into spaces
    text = SINGLE_LINEBREAK_RE.sub(" ", text)

    # Normalize paragraph spacing
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    # Final whitespace cleanup
    text = MULTI_SPACE_RE.sub(" ", text)

    return text.strip()


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove empty metadata values.
    """
    cleaned: Dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue

        cleaned[str(key)] = value

    return cleaned


def normalize_title(raw_title: Any, fallback: str) -> str:
    """
    Normalize article title.
    """
    if isinstance(raw_title, str):
        title = clean_pdf_text(raw_title)
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title

    return re.sub(r"\s+", " ", fallback).strip()


# -------------------------------------------------------------------
# Chonkie helpers
# -------------------------------------------------------------------

def load_sentence_transformer_embeddings_class():
    """
    Try to import Chonkie's SentenceTransformer embedding class.
    """
    try:
        from chonkie.embeddings import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings
    except Exception:
        pass

    try:
        from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings
    except Exception:
        pass

    try:
        from chonkie.embeddings.st import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings
    except Exception:
        pass

    return None


def create_embedding_model(model_name: str):
    """
    Create a Chonkie embedding model.

    Returns None if initialization fails, allowing Chonkie to fall back
    to its default embedding model.
    """
    embedding_class = load_sentence_transformer_embeddings_class()

    if embedding_class is None:
        print(
            "Warning: Could not import Chonkie SentenceTransformer embeddings. "
            "Falling back to Chonkie default embedding model if available.",
            file=sys.stderr,
        )
        return None

    attempts = [
        ((), {"model_name": model_name}),
        ((), {"model": model_name}),
        ((model_name,), {}),
    ]

    last_error = None

    for args, kwargs in attempts:
        try:
            return embedding_class(*args, **kwargs)
        except Exception as exc:
            last_error = exc

    print(
        f"Warning: Could not initialize embedding model '{model_name}'. "
        f"Falling back to Chonkie default if available. Error: {last_error}",
        file=sys.stderr,
    )

    return None


def create_chonkie_semantic_chunker(
    embedding_model_name: str,
    similarity_level: str,
    min_chunk_chars: int,
    max_chunk_chars: int,
):
    """
    Create Chonkie SemanticChunker.

    The script tries a few common constructor signatures because
    Chonkie versions may differ slightly.
    """
    if SemanticChunker is None:
        sys.exit(
            "Chonkie is not installed or failed to import.\n"
            f"Import error: {CHONKIE_IMPORT_ERROR}\n"
            "Install with:\n"
            "  pip install PyMuPDF \"chonkie[semantic]\" sentence-transformers"
        )

    embedding_model = create_embedding_model(embedding_model_name)

    size_options = [
        {
            "min_characters_per_chunk": min_chunk_chars,
            "max_characters_per_chunk": max_chunk_chars,
        },
        {
            "min_chunk_size": min_chunk_chars,
            "max_chunk_size": max_chunk_chars,
        },
        {
            # Approximate token values, in case the installed Chonkie version
            # expects token-based limits.
            "min_split_tokens": max(1, int(min_chunk_chars / 4)),
            "max_split_tokens": max(1, int(max_chunk_chars / 4)),
        },
        {},
    ]

    embedding_options: List[Any] = []

    if embedding_model is not None:
        embedding_options.append(embedding_model)

    # Some Chonkie versions may accept model name strings directly.
    embedding_options.append(embedding_model_name)

    # Also allow Chonkie default embedding model.
    embedding_options.append(None)

    attempts: List[Dict[str, Any]] = []

    for embedding_option in embedding_options:
        for size_option in size_options:
            kwargs: Dict[str, Any] = {
                "similarity_level": similarity_level,
            }

            if embedding_option is not None:
                kwargs["embedding_model"] = embedding_option

            kwargs.update(size_option)
            attempts.append(kwargs)

    # Also try without similarity_level, in case version differs.
    for kwargs in list(attempts):
        no_similarity = dict(kwargs)
        no_similarity.pop("similarity_level", None)
        attempts.append(no_similarity)

    last_error = None

    for kwargs in attempts:
        try:
            return SemanticChunker(**kwargs)
        except Exception as exc:
            last_error = exc
            continue

    # Final fallback: default SemanticChunker
    try:
        return SemanticChunker()
    except Exception as exc:
        sys.exit(
            "Could not create Chonkie SemanticChunker.\n"
            f"Last error: {last_error}\n"
            f"Final fallback error: {exc}"
        )


def run_chonkie_chunker(chunker: Any, text: str) -> List[Any]:
    """
    Run Chonkie chunker safely.
    """
    if not text or not text.strip():
        return []

    if hasattr(chunker, "chunk"):
        result = chunker.chunk(text)
    elif callable(chunker):
        result = chunker(text)
    else:
        raise RuntimeError("Chonkie chunker has no callable chunk method.")

    if result is None:
        return []

    if isinstance(result, dict):
        return [result]

    if isinstance(result, (list, tuple)):
        return list(result)

    return list(result)


def get_any_attr(obj: Any, names: List[str], default: Any = None) -> Any:
    """
    Get attribute from object or dict.
    """
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value

        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]

    return default


# -------------------------------------------------------------------
# Page mapping helpers
# -------------------------------------------------------------------

def build_full_text_and_spans(
    page_units: List[Tuple[int, str]],
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Build full article text and character spans for each page.

    page_units:
        [(page_number, cleaned_page_text), ...]

    Returns:
        full_text
        page_spans = [(start_char, end_char, page_number), ...]
    """
    parts: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    offset = 0

    for i, (page_number, text) in enumerate(page_units):
        start = offset
        end = start + len(text)

        spans.append((start, end, page_number))
        parts.append(text)

        if i < len(page_units) - 1:
            offset = end + len(PAGE_SEP)
        else:
            offset = end

    full_text = PAGE_SEP.join(parts)

    return full_text, spans


def pages_for_chunk(
    page_spans: List[Tuple[int, int, int]],
    start: Optional[int],
    end: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Determine page_start and page_end from character offsets.
    """
    if start is None or end is None:
        return None, None

    try:
        start = int(start)
        end = int(end)
    except Exception:
        return None, None

    if end <= start:
        end = start + 1

    pages: List[int] = []

    for span_start, span_end, page_number in page_spans:
        # Overlap check
        if start < span_end and end > span_start:
            pages.append(page_number)

    if not pages:
        return None, None

    return min(pages), max(pages)


def locate_chunk_text(
    full_text: str,
    chunk_text: str,
    search_start: int = 0,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to locate chunk text inside full text.

    Used when Chonkie does not expose character offsets.
    """
    if not chunk_text:
        return None, None

    idx = full_text.find(chunk_text, max(0, search_start))

    if idx == -1:
        return None, None

    return idx, idx + len(chunk_text)


# -------------------------------------------------------------------
# Semantic chunk builder
# -------------------------------------------------------------------

def build_semantic_chunks(
    page_units: List[Tuple[int, str]],
    full_text: str,
    page_spans: List[Tuple[int, int, int]],
    chunker: Any,
    min_chunk_chars: int,
    chunk_scope: str = "full",
) -> List[Dict[str, Any]]:
    """
    Build semantic chunks using Chonkie.

    chunk_scope:
        "full" -> semantic chunk across whole article
        "page" -> semantic chunk each page separately
    """
    chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Full-document semantic chunking
    # ------------------------------------------------------------
    if chunk_scope == "full":
        try:
            raw_chunks = run_chonkie_chunker(chunker, full_text)
        except Exception as exc:
            print(
                f"Warning: Full-document Chonkie chunking failed. "
                f"Falling back to page-wise chunking. Error: {exc}",
                file=sys.stderr,
            )
            raw_chunks = []

        search_start = 0

        for raw_chunk in raw_chunks:
            text = get_any_attr(raw_chunk, ["text"], "")
            text = str(text or "").strip()

            if len(text) < min_chunk_chars:
                continue

            start = get_any_attr(
                raw_chunk,
                [
                    "start_index",
                    "start",
                    "char_start",
                    "begin",
                    "start_char",
                ],
            )

            end = get_any_attr(
                raw_chunk,
                [
                    "end_index",
                    "end",
                    "char_end",
                    "end_char",
                ],
            )

            # If Chonkie does not provide offsets, try locating text manually.
            if start is None or end is None:
                located_start, located_end = locate_chunk_text(
                    full_text=full_text,
                    chunk_text=text,
                    search_start=search_start,
                )

                if located_start is not None and located_end is not None:
                    start = located_start
                    end = located_end
                    search_start = max(search_start, located_start + 1)

            page_start, page_end = pages_for_chunk(
                page_spans=page_spans,
                start=start,
                end=end,
            )

            token_count = get_any_attr(
                raw_chunk,
                [
                    "token_count",
                    "tokens",
                    "token_size",
                ],
            )

            chunks.append(
                {
                    "text": text,
                    "page_start": page_start,
                    "page_end": page_end,
                    "char_count": len(text),
                    "token_count": token_count,
                }
            )

        # If we successfully got chunks and at least some page metadata,
        # return full-document semantic chunks.
        if chunks and any(chunk["page_start"] is not None for chunk in chunks):
            return chunks

    # ------------------------------------------------------------
    # Page-wise fallback / explicit page-wise mode
    # ------------------------------------------------------------
    chunks = []

    for page_number, page_text in page_units:
        try:
            raw_chunks = run_chonkie_chunker(chunker, page_text)
        except Exception as exc:
            print(
                f"Warning: Chonkie page-wise chunking failed for page {page_number}. "
                f"Error: {exc}",
                file=sys.stderr,
            )
            continue

        for raw_chunk in raw_chunks:
            text = get_any_attr(raw_chunk, ["text"], "")
            text = str(text or "").strip()

            if len(text) < min_chunk_chars:
                continue

            token_count = get_any_attr(
                raw_chunk,
                [
                    "token_count",
                    "tokens",
                    "token_size",
                ],
            )

            chunks.append(
                {
                    "text": text,
                    "page_start": page_number,
                    "page_end": page_number,
                    "char_count": len(text),
                    "token_count": token_count,
                }
            )

    return chunks


# -------------------------------------------------------------------
# PDF extraction
# -------------------------------------------------------------------

def make_article_id(pdf_path: Path, input_dir: Path) -> str:
    """
    Create article ID.
    """
    try:
        identity = str(pdf_path.relative_to(input_dir))
    except ValueError:
        identity = str(pdf_path)

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem)[:80] or "article"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]

    return f"{slug}-{digest}"


def extract_pdf(
    pdf_path: Path,
    input_dir: Path,
    chunker: Any,
    min_chunk_chars: int,
    chunk_scope: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract one PDF into article + semantic chunks.
    """
    doc = fitz.open(str(pdf_path))

    try:
        if getattr(doc, "needs_pass", False):
            raise RuntimeError("Password-protected PDF.")

        metadata_raw = doc.metadata or {}
        metadata = clean_metadata(metadata_raw)

        title = normalize_title(
            raw_title=metadata.get("title"),
            fallback=pdf_path.stem,
        )

        page_units: List[Tuple[int, str]] = []

        for page_number, page in enumerate(doc, start=1):
            raw_page_text = page.get_text("text", sort=True)
            cleaned_page_text = clean_pdf_text(raw_page_text)

            if cleaned_page_text:
                page_units.append((page_number, cleaned_page_text))

        full_text, page_spans = build_full_text_and_spans(page_units)

        article_id = make_article_id(pdf_path, input_dir)

        article = {
            "article_id": article_id,
            "source_file": pdf_path.as_posix(),
            "file_name": pdf_path.name,
            "title": title,
            "metadata": metadata,
            "page_count": doc.page_count,
            "char_count": len(full_text),
            "text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "chunking_method": "chonkie-semantic",
            "text": full_text,
        }

        raw_chunks = build_semantic_chunks(
            page_units=page_units,
            full_text=full_text,
            page_spans=page_spans,
            chunker=chunker,
            min_chunk_chars=min_chunk_chars,
            chunk_scope=chunk_scope,
        )

        # Fallback for tiny documents
        if not raw_chunks and full_text:
            raw_chunks = [
                {
                    "text": full_text,
                    "page_start": page_units[0][0] if page_units else 1,
                    "page_end": page_units[-1][0] if page_units else doc.page_count,
                    "char_count": len(full_text),
                    "token_count": None,
                }
            ]

        chunks: List[Dict[str, Any]] = []

        for chunk_index, chunk in enumerate(raw_chunks):
            chunk_id = f"{article_id}:chunk-{chunk_index:04d}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "source_file": pdf_path.as_posix(),
                    "file_name": pdf_path.name,
                    "title": title,
                    "chunk_index": chunk_index,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "char_count": chunk.get("char_count"),
                    "token_count": chunk.get("token_count"),
                    "text": chunk.get("text", ""),
                }
            )

        return article, chunks

    finally:
        doc.close()


# -------------------------------------------------------------------
# File discovery
# -------------------------------------------------------------------

def find_pdf_files(
    input_path: Path,
    pattern: str,
    recursive: bool,
) -> List[Path]:
    """
    Find PDF files.
    """
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".pdf" else []

    if not input_path.exists():
        sys.exit(f"Input path does not exist: {input_path}")

    if recursive:
        pdf_files = input_path.rglob(pattern)
    else:
        pdf_files = input_path.glob(pattern)

    return sorted(
        path
        for path in pdf_files
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cleaned text and semantic chunks from PDF articles using Chonkie."
    )

    parser.add_argument(
        "--input-dir",
        default="input",
        help="Directory containing PDF files. Default: input",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where JSON files are saved. Default: output",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="PDF file glob pattern. Default: *.pdf",
    )
    parser.add_argument(
        "--recursive",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Search input directory recursively. Default: True",
    )

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help=(
            "Sentence Transformer model used by Chonkie semantic chunking. "
            "Default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )

    parser.add_argument(
        "--similarity-level",
        default="high",
        choices=["low", "medium", "high"],
        help="Chonkie semantic similarity level. Default: high",
    )

    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1500,
        help="Maximum characters per semantic chunk. Default: 1500",
    )

    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=100,
        help="Minimum characters for a chunk to be kept. Default: 100",
    )

    parser.add_argument(
        "--chunk-scope",
        default="full",
        choices=["full", "page"],
        help=(
            "full = semantic chunks across the whole article. "
            "page = semantic chunks per page. "
            "Default: full"
        ),
    )

    args = parser.parse_args()

    if args.min_chunk_chars > args.max_chunk_chars:
        parser.error("--min-chunk-chars must be <= --max-chunk-chars")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Chonkie semantic chunker...")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Similarity level: {args.similarity_level}")
    print(f"Max chunk chars: {args.max_chunk_chars}")
    print(f"Min chunk chars: {args.min_chunk_chars}")
    print(f"Chunk scope: {args.chunk_scope}")

    chunker = create_chonkie_semantic_chunker(
        embedding_model_name=args.embedding_model,
        similarity_level=args.similarity_level,
        min_chunk_chars=args.min_chunk_chars,
        max_chunk_chars=args.max_chunk_chars,
    )

    pdf_files = find_pdf_files(
        input_path=input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    articles: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    if not pdf_files:
        print(f"No PDF files found in {input_dir} with pattern '{args.pattern}'")

    for pdf_path in pdf_files:
        try:
            article, article_chunks = extract_pdf(
                pdf_path=pdf_path,
                input_dir=input_dir,
                chunker=chunker,
                min_chunk_chars=args.min_chunk_chars,
                chunk_scope=args.chunk_scope,
            )

            articles.append(article)
            chunks.extend(article_chunks)

            print(
                f"[OK] {pdf_path}: "
                f"{article['char_count']} chars, "
                f"{len(article_chunks)} semantic chunks"
            )

        except Exception as exc:
            errors.append(
                {
                    "source_file": str(pdf_path),
                    "error": str(exc),
                }
            )
            print(f"[ERROR] {pdf_path}: {exc}", file=sys.stderr)

    articles_path = output_dir / "uee_articles_clean.json"
    chunks_path = output_dir / "uee_chunks.json"

    with articles_path.open("w", encoding="utf-8") as f:
        json.dump(
            articles,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(f"Saved {len(articles)} articles to {articles_path}")
    print(f"Saved {len(chunks)} semantic chunks to {chunks_path}")

    if errors:
        print(f"{len(errors)} PDFs failed. See errors above.", file=sys.stderr)


if __name__ == "__main__":
    main()