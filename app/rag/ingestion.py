import os
import re
import hashlib
from pathlib import Path

from app.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.rag.vectorstore import add_documents, get_collection, clear_collection


def load_text_file(file_path: str) -> str:
    """Load a plain text or markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """Load a PDF file and extract text."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return ""


def load_document(file_path: str) -> str:
    """Load a document based on its file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf_file(file_path)
    elif ext in (".md", ".txt", ".text", ".rst"):
        return load_text_file(file_path)
    else:
        print(f"Unsupported file type: {ext} ({file_path})")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks with sentence-aware boundaries.
    Tries to split on paragraph boundaries first, then sentences, then by size.
    """
    if not text.strip():
        return []

    # Split into paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, finalize the current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of the current chunk
            if chunk_overlap > 0:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

        # If a single paragraph is longer than chunk_size, split it further
        while len(current_chunk) > chunk_size * 1.5:
            # Try to split at sentence boundary
            split_point = _find_sentence_boundary(current_chunk, chunk_size)
            chunks.append(current_chunk[:split_point].strip())
            if chunk_overlap > 0:
                overlap_start = max(0, split_point - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
            else:
                current_chunk = current_chunk[split_point:]

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _find_sentence_boundary(text: str, target_pos: int) -> int:
    """Find the nearest sentence boundary to the target position."""
    # Look for sentence-ending punctuation near the target position
    search_start = max(0, target_pos - 100)
    search_end = min(len(text), target_pos + 100)
    search_region = text[search_start:search_end]

    # Find all sentence boundaries in the search region
    boundaries = []
    for match in re.finditer(r"[.!?]\s+", search_region):
        abs_pos = search_start + match.end()
        boundaries.append(abs_pos)

    if boundaries:
        # Return the boundary closest to target_pos
        return min(boundaries, key=lambda x: abs(x - target_pos))

    # Fallback: split at word boundary near target
    space_pos = text.rfind(" ", 0, target_pos + 50)
    if space_pos > target_pos - 100:
        return space_pos + 1

    return target_pos


def generate_chunk_id(source: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk based on source and index."""
    hash_input = f"{source}:{chunk_index}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def ingest_all_documents() -> dict:
    """
    Scan the documents directory and ingest all supported files.
    Returns stats about the ingestion.
    """
    docs_dir = Path(DOCUMENTS_DIR)
    if not docs_dir.exists():
        print(f"Documents directory not found: {DOCUMENTS_DIR}")
        return {"files_processed": 0, "total_chunks": 0, "errors": []}

    # Supported extensions
    supported_exts = {".md", ".txt", ".text", ".rst", ".pdf"}

    # Find all supported files
    files = []
    for ext in supported_exts:
        files.extend(docs_dir.rglob(f"*{ext}"))

    if not files:
        print("No supported documents found.")
        return {"files_processed": 0, "total_chunks": 0, "errors": []}

    print(f"Found {len(files)} documents to process.")

    # Clear existing collection and re-ingest
    clear_collection()

    all_chunks = []
    all_metadatas = []
    all_ids = []
    errors = []
    files_processed = 0

    for file_path in sorted(files):
        try:
            print(f"Processing: {file_path.name}")
            text = load_document(str(file_path))

            if not text.strip():
                print(f"  Skipped (empty): {file_path.name}")
                continue

            chunks = chunk_text(text)
            print(f"  Generated {len(chunks)} chunks from {file_path.name}")

            for i, chunk in enumerate(chunks):
                chunk_id = generate_chunk_id(file_path.name, i)
                metadata = {
                    "source": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)

            files_processed += 1

        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            print(f"  {error_msg}")
            errors.append(error_msg)

    # Add all chunks to the vector store
    total_added = 0
    if all_chunks:
        total_added = add_documents(all_chunks, all_metadatas, all_ids)

    result = {
        "files_processed": files_processed,
        "total_chunks": total_added,
        "errors": errors,
    }

    print(f"Ingestion complete: {files_processed} files, {total_added} chunks.")
    return result
