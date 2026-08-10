#!/usr/bin/env python3
"""
Extract cleaned article text and RAG chunks from PDF files.

Outputs:
  output/uee_articles_clean.json
  output/uee_chunks.json

Upgrades:
  - RecursiveCharacterTextSplitter (respects sentences/paragraphs)
  - Layout-aware text extraction (fixes multi-column PDFs)
  - Auto-injection of section headers into chunks
  - Automatic footer stripping
  - Accurate page mapping for each chunk
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

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    sys.exit("Missing dependency. Install with: pip install langchain-text-splitters")


# -------------------------------------------------------------------
# Regex / constants
# -------------------------------------------------------------------

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SOFT_HYPHEN_RE = re.compile(r"\u00ad")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
SINGLE_LINEBREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")
FOOTER_RE = re.compile(
    r"Block Statue,\s*Schulz,\s*UEE\s*2011\s*\d+",
    re.IGNORECASE,
)
SECTION_HEADER_RE = re.compile(
    r"^(?=[A-Z][a-z]+\s+[A-Z])?([A-Z][a-z].*?)\s*$",
    re.MULTILINE,
)


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
    """Clean PDF metadata by removing empty values."""
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
    """Normalize article title. If missing, use filename stem."""
    if isinstance(raw_title, str):
        title = clean_pdf_text(raw_title)
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title

    return re.sub(r"\s+", " ", fallback).strip()


# -------------------------------------------------------------------
# Layout-aware PDF text extraction (fixes multi-column issues)
# -------------------------------------------------------------------

def extract_text_in_read_order(page: fitz.Page) -> str:
    """
    Extract text from a PyMuPDF page by sorting text spans
    in reading order (top-to-bottom, left-to-right).

    This fixes garbled text from multi-column layouts where
    the default 'sort=True' still reads left-column then right-column.
    """
    blocks = page.get_text("dict")
    words: List[Tuple[float, float, str]] = []

    for block in blocks.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                # Bbox: (x0, y0, x1, y1)
                bbox = span["bbox"]
                y0 = bbox[1]   # vertical position (top)
                x0 = bbox[0]   # horizontal position (left)
                text = span["text"].strip()
                if text:
                    words.append((y0, x0, text))

    # Sort by vertical (y0) then horizontal (x0)
    words.sort(key=lambda w: (w[0], w[1]))

    # Join with spaces, but preserve some paragraph spacing by adding newlines
    # where vertical gaps are large (heuristic).
    if not words:
        return ""

    result = [words[0][2]]
    prev_y = words[0][0]

    for y, x, text in words[1:]:
        # If vertical gap > 1.5x the average line height, treat as paragraph break
        # Here we just use a rough 12pt threshold (can be tuned per document)
        if y - prev_y > 15.0:  # ~1.5 lines at 12pt
            result.append("\n\n")
        else:
            result.append(" ")
        result.append(text)
        prev_y = y

    return "".join(result)


# -------------------------------------------------------------------
# Section injection and footer stripping
# -------------------------------------------------------------------

def preprocess_for_chunking(text: str) -> str:
    """
    Clean text specifically for chunking:
      - Strip academic footers
      - Inject section headers as explicit context markers
    """
    # 1. Remove footers
    text = FOOTER_RE.sub("", text)

    # 2. Inject section headers: find lines that look like section titles
    #    and prepend them with a marker.
    #    This regex looks for lines that start with a capital letter,
    #    followed by lowercase, and are not too long (likely headers).
    #    We'll use a simpler approach: find all uppercase/lowercase headers
    #    common in academic papers.

    def add_section_context(match: re.Match) -> str:
        header = match.group(1).strip()
        # Avoid injecting if it's just a single word or common false-positive
        if len(header.split()) < 2 and not header.endswith(":"):
            return match.group(0)
        return f"\n\n[SECTION: {header}]\n"

    # This pattern detects lines that are standalone, start with a capital letter,
    # contain multiple words, and are followed by a newline.
    section_pattern = re.compile(
        r"^(?=[A-Z][a-z]+\s+[A-Z])?([A-Z][a-z].*?)\s*$",
        re.MULTILINE,
    )
    text = section_pattern.sub(add_section_context, text)

    # Clean up multiple injected newlines
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    return text


# -------------------------------------------------------------------
# Chunking engine (Recursive Character Splitter)
# -------------------------------------------------------------------

def chunk_document(
    full_text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    page_mapping: List[Tuple[int, int, int]],
) -> List[Dict[str, Any]]:
    """
    Split a full document into semantic RAG chunks.

    Args:
        full_text: The entire cleaned document text.
        chunk_size: Target max characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        min_chunk_chars: Minimum characters to keep a chunk.
        page_mapping: List of (start_char, end_char, page_number) for each page.

    Returns:
        List of chunk dicts with 'text', 'page_start', 'page_end', 'char_count'.
    """
    if not full_text:
        return []

    # Preprocess: strip footers and inject section markers
    processed_text = preprocess_for_chunking(full_text)

    # Use LangChain's RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",   # Paragraph breaks (highest priority)
            "\n",     # Line breaks
            ". ",     # Sentence boundaries
            ", ",     # Clauses
            " ",      # Words
            "",       # Characters (last resort)
        ],
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(processed_text)

    # Build final chunks with page metadata
    final_chunks: List[Dict[str, Any]] = []

    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        if len(chunk_text) < min_chunk_chars:
            continue

        # Find where this chunk roughly starts in the original text
        # Use the first 50 chars as a search key to locate it.
        search_key = chunk_text[:50]
        start_pos = processed_text.find(search_key)
        if start_pos == -1:
            # Fallback: use the previous end position (hard to track perfectly)
            # We'll just approximate using the first mapping.
            start_pos = 0

        end_pos = start_pos + len(chunk_text)

        # Determine which pages this chunk spans
        page_start = 1
        page_end = 1

        for p_start, p_end, p_num in page_mapping:
            # If chunk starts on or after this page's start and before its end
            if start_pos >= p_start and start_pos < p_end:
                page_start = p_num
            # If chunk ends after this page's start and within/at its end
            if end_pos > p_start and end_pos <= p_end:
                page_end = p_num

        final_chunks.append(
            {
                "text": chunk_text,
                "page_start": page_start,
                "page_end": page_end,
                "char_count": len(chunk_text),
            }
        )

    # Safety fallback: if no chunks were created but we have text, push the whole thing
    if not final_chunks and full_text:
        final_chunks.append(
            {
                "text": full_text,
                "page_start": 1,
                "page_end": page_mapping[-1][2] if page_mapping else 1,
                "char_count": len(full_text),
            }
        )

    return final_chunks


# -------------------------------------------------------------------
# PDF extraction
# -------------------------------------------------------------------

def make_article_id(pdf_path: Path, input_dir: Path) -> str:
    """Create a stable-ish article ID from filename/path."""
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
      1. article record (with full text)
      2. chunk records (RAG-ready)
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
        page_mapping: List[Tuple[int, int, int]] = []
        current_char_pos = 0

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # Use layout-aware extraction to fix columns
            raw_page_text = extract_text_in_read_order(page)
            cleaned_page_text = clean_pdf_text(raw_page_text)

            if cleaned_page_text:
                page_start = current_char_pos
                page_end = current_char_pos + len(cleaned_page_text)
                page_mapping.append((page_start, page_end, page_num + 1))
                current_char_pos = page_end
                page_texts.append(cleaned_page_text)

        # Full cleaned text
        full_text = "\n\n".join(page_texts).strip()

        # Generate article ID
        article_id = make_article_id(pdf_path, input_dir)

        # Build the article record
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

        # Create chunks using the new recursive splitter
        raw_chunks = chunk_document(
            full_text=full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
            page_mapping=page_mapping,
        )

        # Build final chunk records with IDs and article metadata
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
    """Find PDF files in input path."""
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

    # Save outputs
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