#!/usr/bin/env python3
"""
Extract cleaned article text and RAG chunks from PDF files.

Outputs:
  output/uee_articles_clean.json
  output/uee_chunks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing dependency. Install with: pip install PyMuPDF")


# -------------------------------------------------------------------
# Regex / constants
# -------------------------------------------------------------------

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SOFT_HYPHEN_RE = re.compile(r"\u00ad")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
SINGLE_LINEBREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

SEP_LEN = len("\n\n")


# -------------------------------------------------------------------
# Cleaning helpers
# -------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    Steps:
    - Unicode normalize
    - Remove invisible/control characters
    - Remove soft hyphens
    - Fix hyphenated linebreaks
    - Collapse single linebreaks into spaces
    - Preserve blank lines as paragraph separators
    - Normalize whitespace
    """
    if not text:
        return ""

    # Unicode normalize
    text = unicodedata.normalize("NFC", text)

    # Remove common invisible characters
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u00a0", " ")

    # Remove soft hyphen
    text = SOFT_HYPHEN_RE.sub("", text)

    # Remove control chars except newline/tab
    text = CONTROL_CHARS_RE.sub("", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix words split by hyphen at end of line
    # Example:
    #   normal-
    #   ization
    # becomes:
    #   normalization
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Collapse multiple spaces/tabs
    text = MULTI_SPACE_RE.sub(" ", text)

    # Trim spaces before/after newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse single linebreaks into spaces.
    # Double linebreaks are kept as paragraph separators.
    text = SINGLE_LINEBREAK_RE.sub(" ", text)

    # Normalize 3+ newlines to 2 newlines
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    # Final whitespace cleanup
    text = MULTI_SPACE_RE.sub(" ", text)

    return text.strip()


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean PDF metadata by removing empty values.
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
    If PDF metadata title is missing/empty, use filename stem.
    """
    if isinstance(raw_title, str):
        title = clean_pdf_text(raw_title)
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title

    return re.sub(r"\s+", " ", fallback).strip()


# -------------------------------------------------------------------
# Chunking helpers
# -------------------------------------------------------------------

def hard_split(text: str, max_chars: int) -> List[str]:
    """
    Hard split text into max_chars pieces.
    Used when sentence splitting is not enough.
    """
    return [
        text[i : i + max_chars]
        for i in range(0, len(text), max_chars)
    ]


def split_long_text(text: str, max_chars: int) -> List[str]:
    """
    Split long text into smaller pieces.

    Priority:
    1. Keep under max_chars
    2. Prefer sentence boundaries
    3. Hard split if necessary
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = SENTENCE_SPLIT_RE.split(text)

    if len(sentences) <= 1:
        return hard_split(text, max_chars)

    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            pieces.extend(hard_split(sentence, max_chars))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return [piece.strip() for piece in pieces if piece.strip()]


def make_segments(
    cleaned_page_text: str,
    page_number: int,
    max_unit_chars: int,
) -> List[Dict[str, Any]]:
    """
    Convert cleaned page text into segments.

    Each segment is usually a paragraph or part of a long paragraph.
    Segments keep their page number for later chunk metadata.
    """
    if not cleaned_page_text:
        return []

    segments: List[Dict[str, Any]] = []

    # Paragraphs are separated by blank lines after cleaning.
    paragraphs = re.split(r"\n{2,}", cleaned_page_text)

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        if len(paragraph) <= max_unit_chars:
            segments.append(
                {
                    "text": paragraph,
                    "page": page_number,
                }
            )
        else:
            for piece in split_long_text(paragraph, max_unit_chars):
                segments.append(
                    {
                        "text": piece,
                        "page": page_number,
                    }
                )

    return segments


def build_chunks(
    segments: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> List[Dict[str, Any]]:
    """
    Build RAG chunks from segments.

    Each chunk contains:
    - text
    - page_start
    - page_end
    - char_count
    """
    chunks: List[Dict[str, Any]] = []

    if not segments:
        return chunks

    idx = 0
    n = len(segments)

    while idx < n:
        current_segments: List[Dict[str, Any]] = []
        current_len = 0
        j = idx

        # Add segments until reaching chunk_size
        while j < n:
            seg_len = len(segments[j]["text"])
            add_len = seg_len + (SEP_LEN if current_segments else 0)

            if current_segments and current_len + add_len > chunk_size:
                break

            current_segments.append(segments[j])
            current_len += add_len
            j += 1

        # Safety fallback, normally not needed
        if not current_segments:
            current_segments = [segments[idx]]
            j = idx + 1

        chunk_text = "\n\n".join(
            segment["text"] for segment in current_segments
        ).strip()

        if len(chunk_text) >= min_chunk_chars:
            chunks.append(
                {
                    "text": chunk_text,
                    "page_start": current_segments[0]["page"],
                    "page_end": current_segments[-1]["page"],
                    "char_count": len(chunk_text),
                }
            )

        if j >= n:
            break

        # No overlap
        if chunk_overlap <= 0:
            idx = j
            continue

        # Determine overlap by keeping trailing segments from current chunk
        overlap_len = 0
        overlap_start = j
        k = j - 1

        while k >= idx:
            seg_len = len(segments[k]["text"])
            add_len = seg_len + (SEP_LEN if overlap_len > 0 else 0)

            if overlap_len + add_len > chunk_overlap:
                break

            overlap_len += add_len
            overlap_start = k
            k -= 1

        # Always move forward at least one segment to avoid infinite loops
        idx = max(idx + 1, overlap_start)

    return chunks


# -------------------------------------------------------------------
# PDF extraction
# -------------------------------------------------------------------

def make_article_id(pdf_path: Path, input_dir: Path) -> str:
    """
    Create a stable-ish article ID from filename/path.
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
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract one PDF into:
    1. article record
    2. chunk records
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

        page_texts: List[str] = []
        segments: List[Dict[str, Any]] = []

        max_unit_chars = chunk_size

        for page_number, page in enumerate(doc, start=1):
            raw_page_text = page.get_text("text", sort=True)
            cleaned_page_text = clean_pdf_text(raw_page_text)

            if cleaned_page_text:
                page_texts.append(cleaned_page_text)
                segments.extend(
                    make_segments(
                        cleaned_page_text=cleaned_page_text,
                        page_number=page_number,
                        max_unit_chars=max_unit_chars,
                    )
                )

        full_text = "\n\n".join(page_texts).strip()

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
            "text": full_text,
        }

        raw_chunks = build_chunks(
            segments=segments,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )

        # Fallback for very small PDFs or edge cases
        if not raw_chunks and full_text:
            raw_chunks = [
                {
                    "text": full_text,
                    "page_start": 1,
                    "page_end": doc.page_count,
                    "char_count": len(full_text),
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
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "char_count": chunk["char_count"],
                    "text": chunk["text"],
                }
            )

        return article, chunks

    finally:
        doc.close()


# -------------------------------------------------------------------
# File discovery / main
# -------------------------------------------------------------------

def find_pdf_files(
    input_path: Path,
    pattern: str,
    recursive: bool,
) -> List[Path]:
    """
    Find PDF files in input path.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cleaned text and RAG chunks from PDF articles."
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
        "--chunk-size",
        type=int,
        default=1200,
        help="Target chunk size in characters. Default: 1200",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters. Default: 200",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=50,
        help="Minimum characters for a chunk to be kept. Default: 50",
    )

    args = parser.parse_args()

    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    if args.min_chunk_chars > args.chunk_size:
        parser.error("--min-chunk-chars must be <= --chunk-size")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

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
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chunk_chars=args.min_chunk_chars,
            )

            articles.append(article)
            chunks.extend(article_chunks)

            print(
                f"[OK] {pdf_path}: "
                f"{article['char_count']} chars, "
                f"{len(article_chunks)} chunks"
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
    print(f"Saved {len(chunks)} chunks to {chunks_path}")

    if errors:
        print(f"{len(errors)} PDFs failed. See errors above.", file=sys.stderr)


if __name__ == "__main__":
    main()