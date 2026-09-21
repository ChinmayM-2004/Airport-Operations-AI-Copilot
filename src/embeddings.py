"""
Embedding generation for the Airport Operations AI Copilot.

This module:
1. Loads policy chunks from JSON.
2. Generates embeddings using Sentence Transformers.
3. Saves embeddings as a NumPy array.
4. Saves the corresponding chunk metadata separately.
"""

from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

import os


# ============================================================
# SYSTEM SSL CERTIFICATE CONFIGURATION
# ============================================================

SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"

os.environ["REQUESTS_CA_BUNDLE"] = SYSTEM_CA
os.environ["SSL_CERT_FILE"] = SYSTEM_CA
os.environ["CURL_CA_BUNDLE"] = SYSTEM_CA

print("Using certificate bundle:", SYSTEM_CA)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_FILE = PROJECT_ROOT / "data" / "chunks" / "policy_chunks.json"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_FILE = PROCESSED_DIR / "policy_embeddings.npy"

EMBEDDING_METADATA_FILE = (
    PROCESSED_DIR / "embedding_metadata.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(
    chunk_file: Path = CHUNK_FILE,
) -> list[dict]:
    """
    Load policy chunks from JSON.
    """

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_file}\n"
            "Run document_chunker.py first."
        )

    with chunk_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)

    if not chunks:
        raise ValueError("No policy chunks found.")

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model() -> SentenceTransformer:
    """
    Load the Sentence Transformer embedding model.
    """

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    return model


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Generate embeddings for all policy chunks.

    Only the chunk text is embedded.
    Metadata is stored separately.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    embeddings: np.ndarray,
    output_file: Path = EMBEDDINGS_FILE,
) -> None:
    """
    Save embeddings as a NumPy binary file.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_file,
        embeddings,
    )

    print(
        f"Embeddings saved to: {output_file}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_embedding_metadata(
    chunks: list[dict],
    output_file: Path = EMBEDDING_METADATA_FILE,
) -> None:
    """
    Save chunk metadata in the same order as the embeddings.

    The order is important:

        embeddings[0] <-> metadata[0]
        embeddings[1] <-> metadata[1]
        ...
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []

    for index, chunk in enumerate(chunks):

        metadata.append(
            {
                "embedding_index": index,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "airport": chunk["airport"],
                "policy_id": chunk["policy_id"],
                "policy_type": chunk["policy_type"],
                "version": chunk["version"],
                "effective_date": chunk["effective_date"],
                "section": chunk["section"],
                "text": chunk["text"],
            }
        )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Embedding metadata saved to: {output_file}"
    )


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:
    """
    Validate the generated embeddings.
    """

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numerical values."
        )

    print()
    print("=" * 70)
    print("EMBEDDING VALIDATION")
    print("=" * 70)

    print(
        f"Number of chunks: {len(chunks)}"
    )

    print(
        f"Number of embeddings: {len(embeddings)}"
    )

    print(
        f"Embedding dimensions: {embeddings.shape[1]}"
    )

    print(
        f"Data type: {embeddings.dtype}"
    )

    print("Validation: PASSED")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AIRPORT POLICY EMBEDDING PIPELINE")
    print("=" * 70)
    print()

    # Load chunks
    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} policy chunks."
    )
    print()

    # Load model
    model = load_embedding_model()

    print()

    # Generate embeddings
    embeddings = generate_embeddings(
        chunks,
        model,
    )

    print()

    # Validate
    validate_embeddings(
        embeddings,
        chunks,
    )

    print()

    # Save embeddings
    save_embeddings(
        embeddings,
    )

    # Save metadata
    save_embedding_metadata(
        chunks,
    )

    print()
    print("=" * 70)
    print("EMBEDDING PIPELINE COMPLETE")
    print("=" * 70)