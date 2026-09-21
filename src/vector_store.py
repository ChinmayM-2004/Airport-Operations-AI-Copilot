import json
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "policy_embeddings.npy"
)

EMBEDDING_METADATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embedding_metadata.json"
)

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

INDEX_FILE = VECTOR_STORE_DIR / "policy_index.faiss"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(file_path=EMBEDDINGS_FILE):
    """
    Load policy embeddings from a NumPy file.

    Returns:
        numpy.ndarray: Embedding matrix.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {file_path}"
        )

    embeddings = np.load(file_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings array, got shape {embeddings.shape}"
        )

    embeddings = embeddings.astype("float32")

    return embeddings


# ============================================================
# LOAD EMBEDDING METADATA
# ============================================================

def load_embedding_metadata(file_path=EMBEDDING_METADATA_FILE):
    """
    Load metadata corresponding to the embeddings.

    The order of this metadata must match the order of
    embeddings in the NumPy file.

    Returns:
        list: Embedding metadata.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Embedding metadata file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            "Embedding metadata must be a list."
        )

    return metadata


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):
    """
    Build a FAISS index using inner product similarity.

    Because the embeddings were normalized during the
    embedding stage, inner product is equivalent to
    cosine similarity.

    Args:
        embeddings (numpy.ndarray): Normalized embeddings.

    Returns:
        faiss.Index: FAISS vector index.
    """

    embedding_dimension = embeddings.shape[1]

    print(
        f"Creating FAISS index with dimension: "
        f"{embedding_dimension}"
    )

    # Inner Product works as cosine similarity
    # when embeddings are normalized.
    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings)

    return index


# ============================================================
# SAVE FAISS INDEX
# ============================================================

def save_faiss_index(index, file_path=INDEX_FILE):
    """
    Save FAISS index to disk.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(index, str(file_path))

    print(f"FAISS index saved to: {file_path}")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_faiss_index(file_path=INDEX_FILE):
    """
    Load a previously saved FAISS index.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {file_path}"
        )

    index = faiss.read_index(str(file_path))

    return index


# ============================================================
# VALIDATE VECTOR STORE
# ============================================================

def validate_vector_store(index, embeddings, metadata):
    """
    Validate that the FAISS index, embeddings, and metadata
    contain the same number of vectors.
    """

    print("\nVECTOR STORE VALIDATION")
    print("-" * 40)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"FAISS vectors:    {index.ntotal}")
    print(f"Metadata records: {len(metadata)}")
    print(f"FAISS dimension:  {index.d}")

    if index.ntotal != len(embeddings):
        raise ValueError(
            "Mismatch between FAISS vectors and embeddings."
        )

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Mismatch between metadata and embeddings."
        )

    if index.d != embeddings.shape[1]:
        raise ValueError(
            "Mismatch between FAISS dimension and embedding dimension."
        )

    print("Validation: PASSED")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("FAISS VECTOR STORE PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load embeddings
    # --------------------------------------------------------

    print("\nLoading embeddings...")

    embeddings = load_embeddings()

    print(
        f"Loaded embeddings: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # 2. Load metadata
    # --------------------------------------------------------

    print("\nLoading embedding metadata...")

    metadata = load_embedding_metadata()

    print(
        f"Loaded metadata records: {len(metadata)}"
    )

    # --------------------------------------------------------
    # 3. Build FAISS index
    # --------------------------------------------------------

    print("\nBuilding FAISS index...")

    index = build_faiss_index(embeddings)

    print(
        f"FAISS index contains {index.ntotal} vectors."
    )

    # --------------------------------------------------------
    # 4. Validate
    # --------------------------------------------------------

    validate_vector_store(
        index,
        embeddings,
        metadata
    )

    # --------------------------------------------------------
    # 5. Save index
    # --------------------------------------------------------

    print("\nSaving FAISS index...")

    save_faiss_index(index)

    # --------------------------------------------------------
    # 6. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VECTOR STORE PIPELINE COMPLETE")
    print("=" * 60)

    print(f"Vectors stored: {index.ntotal}")
    print(f"Dimensions:      {index.d}")
    print(f"Index type:      IndexFlatIP")
    print(f"Index file:      {INDEX_FILE}")


if __name__ == "__main__":
    main()