"""
Document chunking for the Airport Operations AI Copilot.

This module:
1. Loads cleaned policy documents.
2. Splits them into meaningful policy sections.
3. Preserves document metadata.
4. Saves the resulting chunks to data/chunks/policy_chunks.json.
"""

from pathlib import Path
import json
import re

from document_loader import load_documents


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
CHUNK_FILE = CHUNK_DIR / "policy_chunks.json"


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

# Target maximum size for a chunk.
MAX_CHUNK_SIZE = 800

# Minimum size before we consider combining content.
MIN_CHUNK_SIZE = 200


# ============================================================
# SECTION SPLITTING
# ============================================================

def split_into_sections(text: str) -> list[dict]:
    """
    Split a Markdown policy document into sections.

    Sections are identified using Markdown headings such as:

    ## 1. Purpose
    ## 2. Scope
    ## 3. Pickup Rules

    Returns:
        List of dictionaries containing:
        - section_title
        - text
    """

    pattern = r"(?m)^##\s+(.+)$"

    matches = list(re.finditer(pattern, text))

    sections = []

    # Content before the first ## heading
    if matches:
        preamble = text[:matches[0].start()].strip()

        if preamble:
            sections.append(
                {
                    "section_title": "Document Header",
                    "text": preamble,
                }
            )

    # Extract each section
    for index, match in enumerate(matches):

        section_title = match.group(1).strip()

        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        section_text = text[start:end].strip()

        if section_text:
            sections.append(
                {
                    "section_title": section_title,
                    "text": section_text,
                }
            )

    # If no sections were found, treat the entire document as one section
    if not sections:
        sections.append(
            {
                "section_title": "Full Document",
                "text": text.strip(),
            }
        )

    return sections


# ============================================================
# LARGE SECTION SPLITTING
# ============================================================

def split_large_section(
    section_text: str,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[str]:
    """
    Split a large section into smaller chunks.

    The function tries to split on paragraph boundaries first,
    then falls back to sentence boundaries.

    Args:
        section_text: Section content.
        max_size: Maximum approximate chunk size.

    Returns:
        List of chunk strings.
    """

    if len(section_text) <= max_size:
        return [section_text.strip()]

    paragraphs = [
        paragraph.strip()
        for paragraph in section_text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if not current_chunk:
            current_chunk = paragraph
            continue

        candidate = current_chunk + "\n\n" + paragraph

        if len(candidate) <= max_size:
            current_chunk = candidate
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Handle paragraphs that are themselves too large
    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= max_size:
            final_chunks.append(chunk)
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            chunk,
        )

        current = ""

        for sentence in sentences:

            if not current:
                current = sentence
                continue

            candidate = current + " " + sentence

            if len(candidate) <= max_size:
                current = candidate
            else:
                final_chunks.append(current.strip())
                current = sentence

        if current:
            final_chunks.append(current.strip())

    return final_chunks


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def chunk_document(
    document: dict,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[dict]:
    """
    Convert one policy document into chunks.

    Metadata from the original document is copied into every chunk.
    """

    sections = split_into_sections(document["content"])

    chunks = []

    chunk_number = 1

    for section in sections:

        section_chunks = split_large_section(
            section["text"],
            max_size=max_size,
        )

        for section_chunk in section_chunks:

            chunk_text = (
                f"Section: {section['section_title']}\n\n"
                f"{section_chunk}"
            )

            chunk = {
                "chunk_id": (
                    f"{document['document_id']}_chunk_{chunk_number}"
                ),
                "document_id": document["document_id"],
                "file_name": document["file_name"],
                "airport": document["metadata"]["airport"],
                "policy_id": document["metadata"]["policy_id"],
                "policy_type": document["metadata"]["policy_type"],
                "version": document["metadata"]["version"],
                "effective_date": document["metadata"]["effective_date"],
                "section": section["section_title"],
                "text": chunk_text,
            }

            chunks.append(chunk)

            chunk_number += 1

    return chunks


# ============================================================
# ALL DOCUMENTS
# ============================================================

def create_all_chunks(
    documents: list[dict],
    max_size: int = MAX_CHUNK_SIZE,
) -> list[dict]:
    """
    Create chunks for all loaded policy documents.
    """

    all_chunks = []

    for document in documents:

        document_chunks = chunk_document(
            document,
            max_size=max_size,
        )

        all_chunks.extend(document_chunks)

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(
    chunks: list[dict],
    output_file: Path = CHUNK_FILE,
) -> None:
    """
    Save chunks as JSON.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_chunk_summary(chunks: list[dict]) -> None:
    """
    Print chunking statistics.
    """

    print("=" * 70)
    print("POLICY CHUNKING SUMMARY")
    print("=" * 70)

    print(f"Total chunks created: {len(chunks)}")

    print()

    # Count chunks by airport
    airport_counts = {}

    for chunk in chunks:

        airport = chunk["airport"]

        airport_counts[airport] = (
            airport_counts.get(airport, 0) + 1
        )

    print("Chunks by airport:")

    for airport, count in sorted(airport_counts.items()):
        print(f"  {airport}: {count}")

    print()

    # Show first five chunks
    print("Sample chunks:")
    print()

    for chunk in chunks[:5]:

        print("-" * 70)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Airport: {chunk['airport']}")
        print(f"Policy Type: {chunk['policy_type']}")
        print(f"Section: {chunk['section']}")
        print(f"Characters: {len(chunk['text'])}")
        print()
        print(chunk["text"][:400])

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading policy documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")
    print()

    print("Creating policy chunks...")

    chunks = create_all_chunks(documents)

    save_chunks(chunks)

    print_chunk_summary(chunks)

    print()
    print(f"Chunks saved to: {CHUNK_FILE}")