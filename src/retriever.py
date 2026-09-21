import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# SSL CERTIFICATE CONFIGURATION
# ============================================================

SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"

os.environ["REQUESTS_CA_BUNDLE"] = SYSTEM_CA
os.environ["SSL_CERT_FILE"] = SYSTEM_CA
os.environ["CURL_CA_BUNDLE"] = SYSTEM_CA

print(f"Using certificate bundle: {SYSTEM_CA}")


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    PROJECT_ROOT
    / "vector_store"
    / "policy_index.faiss"
)

EMBEDDING_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_metadata.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# RETRIEVER CLASS
# ============================================================

class PolicyRetriever:
    """
    Retrieves the most relevant airport policy chunks
    using semantic similarity search with FAISS.
    """

    def __init__(
        self,
        index_file=INDEX_FILE,
        metadata_file=EMBEDDING_METADATA_FILE,
        model_name=MODEL_NAME,
    ):
        self.index_file = Path(index_file)
        self.metadata_file = Path(metadata_file)
        self.model_name = model_name

        self.index = self._load_index()
        self.metadata = self._load_metadata()
        self.model = self._load_model()

        self._validate()

    # ========================================================
    # LOAD FAISS INDEX
    # ========================================================

    def _load_index(self):
        """
        Load the saved FAISS index.
        """

        if not self.index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_file}"
            )

        print("Loading FAISS index...")

        index = faiss.read_index(
            str(self.index_file)
        )

        print(
            f"FAISS index loaded: "
            f"{index.ntotal} vectors"
        )

        return index

    # ========================================================
    # LOAD METADATA
    # ========================================================

    def _load_metadata(self):
        """
        Load metadata corresponding to the vectors
        stored in the FAISS index.
        """

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_file}"
            )

        print("Loading embedding metadata...")

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        print(
            f"Metadata records loaded: "
            f"{len(metadata)}"
        )

        return metadata

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    def _load_model(self):
        """
        Load the same Sentence Transformer model used
        during document embedding.
        """

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        model = SentenceTransformer(
            self.model_name
        )

        print(
            "Embedding model loaded successfully."
        )

        return model

    # ========================================================
    # VALIDATE RETRIEVER
    # ========================================================

    def _validate(self):
        """
        Ensure FAISS vectors and metadata are aligned.
        """

        embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "Mismatch between FAISS vectors and "
                "metadata records."
            )

        if self.index.d != embedding_dimension:
            raise ValueError(
                "Mismatch between FAISS dimension and "
                "embedding model dimension."
            )

        print("Retriever validation: PASSED")

    # ========================================================
    # EMBED QUERY
    # ========================================================

    def embed_query(self, query):
        """
        Convert a user query into a normalized embedding.

        Args:
            query (str): User's policy question.

        Returns:
            numpy.ndarray: Query embedding.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")

    # ========================================================
    # DETECT AIRPORT
    # ========================================================

    @staticmethod
    def detect_airport(query):
        """
        Detect an airport mentioned explicitly in the query.

        This helps prevent a question about one airport from
        being answered using another airport's policy.
        """

        query_upper = query.upper()

        airport_codes = {
            "SFO": "San Francisco International Airport (SFO)",
            "LAX": "Los Angeles International Airport (LAX)",
            "JFK": "John F. Kennedy International Airport (JFK)",
        }

        for code, airport_name in airport_codes.items():

            if code in query_upper:
                return airport_name

        return None

    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, query, top_k=3):
        """
        Search for the most relevant policy chunks.

        If the query explicitly mentions SFO, LAX, or JFK,
        results are restricted to that airport's policies.

        Args:
            query (str): User's policy question.
            top_k (int): Number of chunks to retrieve.

        Returns:
            list: Retrieved policy chunks.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # ----------------------------------------------------
        # Convert query into embedding
        # ----------------------------------------------------

        query_embedding = self.embed_query(
            query
        )

        # ----------------------------------------------------
        # Search more candidates than needed.
        #
        # This gives us enough candidates to apply the
        # airport-specific filtering when required.
        # ----------------------------------------------------

        candidate_k = min(
            max(top_k * 5, 10),
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_embedding,
            candidate_k
        )

        # ----------------------------------------------------
        # Detect explicitly mentioned airport
        # ----------------------------------------------------

        requested_airport = self.detect_airport(
            query
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            metadata = self.metadata[index]

            # ------------------------------------------------
            # Airport-specific filtering
            # ------------------------------------------------

            if (
                requested_airport is not None
                and metadata.get("airport")
                != requested_airport
            ):
                continue

            result = {
                "score": float(score),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "document_id": metadata.get(
                    "document_id"
                ),
                "file_name": metadata.get(
                    "file_name"
                ),
                "airport": metadata.get(
                    "airport"
                ),
                "policy_id": metadata.get(
                    "policy_id"
                ),
                "policy_type": metadata.get(
                    "policy_type"
                ),
                "version": metadata.get(
                    "version"
                ),
                "effective_date": metadata.get(
                    "effective_date"
                ),
                "section": metadata.get(
                    "section"
                ),
                "text": metadata.get(
                    "text"
                ),
            }

            results.append(result)

            if len(results) == top_k:
                break

        # ----------------------------------------------------
        # Add final ranking after filtering
        # ----------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1
        ):
            result["rank"] = rank

        return results

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    @staticmethod
    def display_results(query, results):
        """
        Display retrieved policy chunks in a readable format.
        """

        print("\n" + "=" * 70)
        print("POLICY RETRIEVAL RESULTS")
        print("=" * 70)

        print(f"\nQuery:\n{query}")

        if not results:
            print(
                "\nNo relevant policy chunks found."
            )
            return

        for result in results:

            print("\n" + "-" * 70)

            print(
                f"Rank:       {result['rank']}"
            )

            print(
                f"Similarity: {result['score']:.4f}"
            )

            print(
                f"Source:     {result['file_name']}"
            )

            print(
                f"Airport:    {result['airport']}"
            )

            print(
                f"Policy:     {result['policy_id']}"
            )

            print(
                f"Type:       {result['policy_type']}"
            )

            print(
                f"Section:    {result['section']}"
            )

            print(
                f"\nText:\n{result['text']}"
            )

        print("\n" + "=" * 70)

    # ========================================================
    # TEST ALL QUESTIONS
    # ========================================================

    def run_test_questions(self, questions):
        """
        Run multiple retrieval test questions and display
        the top result for each.
        """

        print("\n" + "=" * 70)
        print("RETRIEVAL TEST RESULTS")
        print("=" * 70)

        for number, query in enumerate(
            questions,
            start=1
        ):

            results = self.search(
                query,
                top_k=3
            )

            print(
                f"\nTest Question {number}:"
            )
            print(query)

            if not results:
                print("  No results found.")
                continue

            for result in results:

                print(
                    f"  {result['rank']}. "
                    f"{result['file_name']} | "
                    f"Score: "
                    f"{result['score']:.4f}"
                )


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUERIES = [
    "What is the maximum surge multiplier allowed at SFO?",
    "Can drivers abandon the airport queue at SFO?",
    "What approval is required before increasing surge at SFO?",
    "What is the maximum surge multiplier allowed at LAX?",
    "What is the maximum surge multiplier allowed at JFK?",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AIRPORT POLICY RETRIEVER")
    print("=" * 70)

    retriever = PolicyRetriever()

    # --------------------------------------------------------
    # Detailed test for the first question
    # --------------------------------------------------------

    query = TEST_QUERIES[0]

    results = retriever.search(
        query,
        top_k=3
    )

    retriever.display_results(
        query,
        results
    )

    # --------------------------------------------------------
    # Test all questions
    # --------------------------------------------------------

    retriever.run_test_questions(
        TEST_QUERIES
    )


if __name__ == "__main__":
    main()